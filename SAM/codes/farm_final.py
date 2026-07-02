# -*- coding: utf-8 -*-
"""
Farm Boundary Detection — Sharper Boundaries + Better Sub-Farm Splitting
=========================================================================
Fixes vs farm_detection_green_quality.py:

  Problem 1: Farms divided by thin bunds detected as one polygon
  Fix: CLAHE + unsharp-mask preprocessing
       → amplifies local contrast at every boundary before SAM sees the tile
       → thin paths / bunds become clearly visible → SAM splits them correctly

  Problem 2: Over-smoothing rounded away actual farm corners
  Fix: SMOOTH_TOLERANCE 3.0 → 1.8  (less simplification)
       CHAIKIN_ITERS    5   → 2    (less corner-cutting, keeps real corners)

  Problem 3: Gap-closing merged farms across thin bunds
  Fix: GAP_TOLERANCE 1.5 → 0.8 m  (only fills genuine SAM boundary gaps)

  Problem 4: Some farms still missed
  Fix: PRED_IOU_THRESH  0.78 → 0.75
       STABILITY_THRESH 0.82 → 0.80
       (lower thresholds = more farm mask candidates)

Resumes from checkpoint.pkl automatically.
Estimated runtime: ~4.5 hours on RTX 3050 4 GB.
"""

import gc
import os
import pickle
import threading
import time
import warnings

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid

os.environ['PROJ_LIB']  = r'C:\Users\Dishan\anaconda3\envs\sam_env\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\Dishan\anaconda3\envs\sam_env\Library\share\gdal'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
IMAGE_PATH      = r"D:\BISAG\GUJ_235182318471_3_JAN_2021_C2EM.tif"
SAM_CHECKPOINT  = r"D:\BISAG\sam_vit_b_01ec64.pth"
OUTPUT_FILE     = r"D:\BISAG\farm_final.shp"
CHECKPOINT_FILE = r"D:\BISAG\checkpoint.pkl"


# ══════════════════════════════════════════════════════════════════════════════
#  SAM — RTX 3050 4 GB limits (do not change)
# ══════════════════════════════════════════════════════════════════════════════
POINTS_PER_SIDE  = 16
POINTS_PER_BATCH = 64
CROP_N_LAYERS    = 0
PRED_IOU_THRESH  = 0.75   # ↓ from 0.78 → catch more farm boundaries
STABILITY_THRESH = 0.80   # ↓ from 0.82 → catch more farm boundaries
MIN_MASK_AREA    = 50     # px


# ══════════════════════════════════════════════════════════════════════════════
#  TILING  (must match existing checkpoint stride — do not change)
# ══════════════════════════════════════════════════════════════════════════════
BLOCK_SIZE = 512
OVERLAP    = 128
STRIDE     = BLOCK_SIZE - OVERLAP   # 384


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING — new, key improvement
# ══════════════════════════════════════════════════════════════════════════════
#
#  WHY CLAHE:
#  SAM detects boundaries by seeing contrast transitions.
#  Thin bunds between farms are often only 1-3 pixels wide and low-contrast
#  after global normalization. CLAHE (Contrast Limited Adaptive Histogram
#  Equalization) enhances contrast LOCALLY in 8×8 pixel tiles — so a thin
#  bund that was invisible at global scale becomes a clear dark line.
#  Result: SAM splits what it previously saw as one farm into 2–4 farms.
#
#  WHY UNSHARP MASK:
#  Sharpens edges without amplifying noise. Works by subtracting a blurred
#  version from the original. Makes every farm boundary crisper.
#
CLAHE_CLIP_LIMIT   = 3.0   # higher = more contrast enhancement (2–4 typical)
CLAHE_TILE_GRID    = 8     # 8×8 pixel grid for local contrast
UNSHARP_STRENGTH   = 0.6   # blend factor for sharpening (0=none, 1=full)
UNSHARP_BLUR_SIGMA = 1.5   # blur radius for unsharp mask


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
MIN_FARM_AREA      = 600     # sqm ↓ from 800 — catches slightly smaller farms
MAX_FARM_AREA      = 80_000  # sqm
MAX_MASK_COVERAGE  = 0.85
NMS_OVERLAP_THRESH = 0.50

# Smoothing — CORRECTED from previous version
# 3.0m + 5 iters was over-smoothing: rounding real corners, merging adjacent farms
# 1.8m + 2 iters preserves actual farm shape while removing pixel noise
SMOOTH_TOLERANCE = 1.8    # metres Douglas-Peucker (was 3.0 — too aggressive)
CHAIKIN_ITERS    = 2      # corner-cutting passes (was 5 — too round)

# Gap closing — reduced to avoid crossing thin bunds
GAP_TOLERANCE = 0.8       # metres (was 1.5 — was merging farms across bunds)

CHECKPOINT_EVERY = 20


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def normalize_image(tile_hwc: np.ndarray) -> np.ndarray:
    """Per-band 2-98 percentile stretch → uint8."""
    out = np.zeros_like(tile_hwc, dtype=np.uint8)
    for i in range(tile_hwc.shape[2]):
        b = tile_hwc[:, :, i].astype(np.float32)
        v = b[b > 0]
        if v.size > 100:
            p2, p98 = np.percentile(v, [2, 98])
            b = np.clip((b - p2) / max(p98 - p2, 1e-5), 0, 1)
        else:
            b = b / max(b.max(), 1.0)
        out[:, :, i] = (b * 255).astype(np.uint8)
    return out


def enhance_for_sam(rgb: np.ndarray) -> np.ndarray:
    """
    CLAHE + Unsharp Mask enhancement before SAM inference.

    Step 1 — CLAHE per channel:
      Enhances local contrast so thin bunds/paths between adjacent farms
      become visible as clear dark lines. This is the primary fix for
      'multiple farms detected as one polygon'.

    Step 2 — Unsharp mask:
      Sharpens every boundary. Formula: output = original + strength × (original − blur)
      Makes farm edges crisper without amplifying noise.

    Step 3 — Blend with original:
      We blend 70% enhanced + 30% original to avoid over-processing.
      Pure CLAHE can sometimes create false boundaries in uniform areas.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_GRID, CLAHE_TILE_GRID)
    )

    # Step 1: CLAHE per channel
    enhanced = np.zeros_like(rgb, dtype=np.uint8)
    for i in range(3):
        enhanced[:, :, i] = clahe.apply(rgb[:, :, i])

    # Step 2: Unsharp mask on enhanced image
    blurred  = cv2.GaussianBlur(enhanced, (0, 0), UNSHARP_BLUR_SIGMA)
    sharpened = cv2.addWeighted(
        enhanced, 1.0 + UNSHARP_STRENGTH,
        blurred,  -UNSHARP_STRENGTH,
        0
    )
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Step 3: Blend enhanced + original (70/30)
    result = cv2.addWeighted(sharpened, 0.7, rgb, 0.3, 0)
    return result.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  TILING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_core_bounds(tx, ty, tw, th, img_w, img_h):
    h   = OVERLAP // 2
    cx0 = tx        if tx == 0               else tx + h
    cy0 = ty        if ty == 0               else ty + h
    cx1 = (tx + tw) if (tx + BLOCK_SIZE >= img_w) else (tx + tw - h)
    cy1 = (ty + th) if (ty + BLOCK_SIZE >= img_h) else (ty + th - h)
    return cx0, cy0, cx1, cy1


def centroid_in_core(cx, cy, core):
    return core[0] <= cx < core[2] and core[1] <= cy < core[3]


def extract_polygons(geom, min_area=0):
    if geom is None or geom.is_empty:
        return []
    t = geom.geom_type
    if t == "Polygon":
        return [geom] if geom.area > min_area else []
    if t == "MultiPolygon":
        return [g for g in geom.geoms if g.area > min_area]
    if t == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(extract_polygons(g, min_area))
        return out
    return []


def chaikin_smooth(coords, iters: int = CHAIKIN_ITERS):
    """
    Vectorised Chaikin corner-cutting.
    2 iterations: removes pixel staircase, keeps real farm corners.
    (5 iterations was too aggressive — was rounding genuine 90° field corners)
    """
    pts = np.asarray(list(coords), dtype=np.float64)
    if np.allclose(pts[0], pts[-1]) and len(pts) > 1:
        pts = pts[:-1]
    n = len(pts)
    for _ in range(iters):
        if n < 3:
            break
        p0 = pts
        p1 = np.roll(pts, -1, axis=0)
        q        = 0.75 * p0 + 0.25 * p1
        r        = 0.25 * p0 + 0.75 * p1
        new      = np.empty((n * 2, 2), dtype=np.float64)
        new[0::2] = q
        new[1::2] = r
        pts = new
        n   = len(pts)
    out = pts.tolist()
    out.append(out[0])
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint_async(polygons: list, tile_idx: int):
    snapshot = list(polygons)
    def _worker():
        tmp = CHECKPOINT_FILE + ".tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump({"polygons": snapshot, "tile_idx": tile_idx}, f)
            os.replace(tmp, CHECKPOINT_FILE)
        except Exception as e:
            print(f"  [CKPT] Failed: {e}")
    threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  IoU-NMS
# ══════════════════════════════════════════════════════════════════════════════

def remove_overlaps_nms(polys: list) -> list:
    """Larger farm wins. Smaller removed if >50% covered. O(n log n)."""
    if not polys:
        return []
    n = len(polys)
    print(f"  NMS: {n:,} polygons ...")
    t0 = time.time()

    order    = sorted(range(n), key=lambda i: polys[i].area, reverse=True)
    sp       = [polys[i] for i in order]
    suppress = [False] * n
    tree     = STRtree(sp)

    import shapely
    v2 = int(shapely.__version__.split(".")[0]) >= 2

    for i in range(n):
        if suppress[i]:
            continue
        pi    = sp[i]
        cands = tree.query(pi, predicate="intersects") if v2 else tree.query(pi)
        for j in cands:
            if j <= i or suppress[j]:
                continue
            pj = sp[j]
            try:
                inter = pi.intersection(pj)
                if not inter.is_empty and (inter.area / pj.area) > NMS_OVERLAP_THRESH:
                    suppress[j] = True
            except Exception:
                suppress[j] = True
        if (i + 1) % 20_000 == 0:
            kept = sum(1 for s in suppress[:i+1] if not s)
            print(f"    {i+1:,}/{n:,} — {kept:,} kept ...")

    result = [p for p, s in zip(sp, suppress) if not s]
    print(f"  NMS: {n:,} → {len(result):,} "
          f"({n-len(result):,} suppressed)  [{time.time()-t0:.1f}s]")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  GAP CLOSING
# ══════════════════════════════════════════════════════════════════════════════

def close_gaps(polys: list, tolerance: float = GAP_TOLERANCE) -> list:
    """
    Adjacent farms → shared boundary, no gap.
    tolerance=0.8m: only fills genuine SAM boundary uncertainty gaps.
    Does NOT cross thin bunds (which are typically 1–3m wide).
    """
    if not polys:
        return []

    import shapely
    v2 = int(shapely.__version__.split(".")[0]) >= 2

    n = len(polys)
    print(f"  Gap closing: {n:,} polygons, tolerance={tolerance} m ...")
    t0 = time.time()

    order    = sorted(range(n), key=lambda i: polys[i].area, reverse=True)
    sorted_p = [polys[i] for i in order]
    tree     = STRtree(sorted_p)
    placed   = {}
    result   = [None] * n

    for rank, p in enumerate(sorted_p):
        try:
            expanded = p.buffer(tolerance, join_style=2, mitre_limit=2.0)
        except Exception:
            result[rank] = p
            placed[rank] = p
            continue

        cands = list(
            tree.query(expanded, predicate="intersects") if v2
            else tree.query(expanded)
        )

        for j in cands:
            if j >= rank or j not in placed:
                continue
            try:
                expanded = expanded.difference(placed[j])
                if expanded.is_empty:
                    break
            except Exception:
                try:
                    expanded = make_valid(expanded).difference(make_valid(placed[j]))
                except Exception:
                    break

        if expanded is None or expanded.is_empty:
            expanded = p

        parts = extract_polygons(expanded, min_area=MIN_FARM_AREA * 0.3)
        if parts:
            orig_c = p.centroid
            best   = min(parts, key=lambda x: x.centroid.distance(orig_c))
        else:
            best = p

        placed[rank] = best
        result[rank] = best

        if (rank + 1) % 2_000 == 0:
            print(f"    {rank+1:,}/{n:,} ({(rank+1)/n*100:.0f}%) ...")

    out = [r for r in result if r is not None]
    print(f"  Gap closing done: {len(out):,} polygons  [{time.time()-t0:.1f}s]")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def post_process(polygons: list, crs):
    from pyproj import CRS

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  POST-PROCESSING — {len(polygons):,} raw polygons")
    print(f"  Area    : {MIN_FARM_AREA}–{MAX_FARM_AREA} sqm")
    print(f"  Smooth  : DP={SMOOTH_TOLERANCE}m + Chaikin×{CHAIKIN_ITERS}")
    print(f"  Gap     : {GAP_TOLERANCE} m")
    print(bar)

    native_crs = CRS.from_user_input(crs)
    if native_crs.is_geographic:
        with rasterio.open(IMAGE_PATH) as src:
            l, b, r, t2 = src.bounds
        cx   = (l + r) / 2;  cy = (b + t2) / 2
        zone = int((cx + 180) / 6) + 1
        epsg = 32600 + zone if cy > 0 else 32700 + zone
        utm_crs = CRS.from_epsg(epsg)
        print(f"  UTM: EPSG:{epsg}")
    else:
        utm_crs = native_crs

    # 1. Area filter
    print(f"\n[1/5] Area filter ...")
    t0      = time.time()
    gdf     = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf_utm = gdf.to_crs(utm_crs)
    areas   = gdf_utm.geometry.area
    mask    = (areas >= MIN_FARM_AREA) & (areas <= MAX_FARM_AREA)
    gdf_utm = gdf_utm[mask].reset_index(drop=True)
    print(f"  → {len(gdf_utm):,} remain  "
          f"({int((~mask).sum()):,} removed)  [{time.time()-t0:.1f}s]")

    if len(gdf_utm) == 0:
        print("  [WARNING] No polygons after area filter!")
        return [], utm_crs

    # 2. Validate + NMS
    print("\n[2/5] Validation + IoU-NMS ...")
    t0    = time.time()
    valid = []
    for p in gdf_utm.geometry:
        if not p.is_valid:
            p = make_valid(p)
        valid.extend(extract_polygons(p, MIN_FARM_AREA))
    print(f"  → {len(valid):,} valid polygons")
    kept = remove_overlaps_nms(valid)

    # 3. Gap closing
    print("\n[3/5] Gap closing ...")
    gapless = close_gaps(kept, tolerance=GAP_TOLERANCE)

    # 4. Smooth
    print(f"\n[4/5] Smoothing (DP={SMOOTH_TOLERANCE}m, Chaikin×{CHAIKIN_ITERS}) ...")
    t0       = time.time()
    smoothed = []
    fb       = 0
    for p in gapless:
        try:
            s = p.simplify(SMOOTH_TOLERANCE, preserve_topology=True)
            if s.is_empty or s.area < MIN_FARM_AREA:
                continue
            for part in extract_polygons(s, MIN_FARM_AREA):
                try:
                    ext    = chaikin_smooth(list(part.exterior.coords))
                    holes  = [chaikin_smooth(list(h.coords))
                               for h in part.interiors]
                    result = Polygon(ext, holes)
                    if result.is_valid and result.area >= MIN_FARM_AREA:
                        smoothed.append(result)
                    elif part.is_valid:
                        smoothed.append(part)
                except Exception:
                    if part.is_valid:
                        smoothed.append(part)
                    fb += 1
        except Exception:
            fb += 1
            if p.is_valid and p.area >= MIN_FARM_AREA:
                smoothed.append(p)
    if fb:
        print(f"    ({fb} used simplified-only fallback)")
    print(f"  → {len(smoothed):,} after smoothing  [{time.time()-t0:.1f}s]")

    # 5. Final validity
    print("\n[5/5] Final validity ...")
    t0    = time.time()
    final = []
    for p in smoothed:
        if not p.is_valid:
            p = make_valid(p)
        final.extend(extract_polygons(p, MIN_FARM_AREA))
    print(f"  → FINAL: {len(final):,} clean polygons  [{time.time()-t0:.1f}s]")
    print(bar)
    return final, utm_crs


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 62)
    print("  FARM DETECTION — Sharper Boundaries + Better Splitting")
    print("=" * 62)
    print(f"  Device      : {DEVICE}")
    print(f"  Tile/Stride : {BLOCK_SIZE}/{STRIDE}  overlap={OVERLAP}")
    print(f"  SAM IOU     : {PRED_IOU_THRESH}  Stability: {STABILITY_THRESH}")
    print(f"  Preprocess  : CLAHE(clip={CLAHE_CLIP_LIMIT}, grid={CLAHE_TILE_GRID})"
          f" + Unsharp(σ={UNSHARP_BLUR_SIGMA}, s={UNSHARP_STRENGTH})")
    print(f"  Smooth      : DP={SMOOTH_TOLERANCE}m  Chaikin×{CHAIKIN_ITERS}")
    print(f"  Gap         : {GAP_TOLERANCE} m")
    print("=" * 62)

    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"\n[GPU] {p.name}  |  {p.total_memory/1e9:.1f} GB VRAM")

    # ── Resume ────────────────────────────────────────────────────────────────
    all_polygons   = []
    start_tile_idx = 0

    if os.path.exists(CHECKPOINT_FILE):
        print("\n[RESUME] Loading checkpoint ...")
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                data = pickle.load(f)
            all_polygons   = data["polygons"]
            start_tile_idx = data["tile_idx"]
            print(f"  → {len(all_polygons):,} polygons, resuming from tile {start_tile_idx}")
        except Exception as e:
            print(f"  → Load failed ({e}), starting fresh")

    # ── Load SAM ──────────────────────────────────────────────────────────────
    print("\n[MODEL] Loading SAM vit_b ...")
    torch.cuda.empty_cache()
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    mask_gen = SamAutomaticMaskGenerator(
        model                          = sam,
        points_per_side                = POINTS_PER_SIDE,
        points_per_batch               = POINTS_PER_BATCH,
        pred_iou_thresh                = PRED_IOU_THRESH,
        stability_score_thresh         = STABILITY_THRESH,
        crop_n_layers                  = CROP_N_LAYERS,
        crop_n_points_downscale_factor = 2,
        min_mask_region_area           = MIN_MASK_AREA,
    )
    print("[MODEL] SAM ready\n")

    # ── Tile loop ─────────────────────────────────────────────────────────────
    with rasterio.open(IMAGE_PATH) as src:
        transform    = src.transform
        crs          = src.crs
        img_w, img_h = src.width, src.height

        y_steps     = list(range(0, img_h, STRIDE))
        x_steps     = list(range(0, img_w, STRIDE))
        total_tiles = len(y_steps) * len(x_steps)

        print(f"[TILES] {len(x_steps)} × {len(y_steps)} = {total_tiles:,} total")
        print(f"        Resuming at tile {start_tile_idx} "
              f"({total_tiles - start_tile_idx:,} remaining)\n")

        tile_idx = 0
        active   = 0
        recent_t : list[float] = []

        for y in y_steps:
            for x in x_steps:

                if tile_idx < start_tile_idx:
                    tile_idx += 1
                    continue

                curr_w   = min(BLOCK_SIZE, img_w - x)
                curr_h   = min(BLOCK_SIZE, img_h - y)
                tile_chw = src.read([1, 2, 3], window=Window(x, y, curr_w, curr_h))
                tile_hwc = np.transpose(tile_chw, (1, 2, 0))

                if tile_hwc.std() < 8 or tile_hwc.mean() < 10:
                    tile_idx += 1
                    continue

                # ── Preprocess: normalize → CLAHE → unsharp ──────────────────
                rgb      = normalize_image(tile_hwc)
                rgb_enh  = enhance_for_sam(rgb)   # ← KEY new step

                core = get_core_bounds(x, y, curr_w, curr_h, img_w, img_h)

                torch.cuda.empty_cache()
                t0 = time.time()
                try:
                    with torch.inference_mode():
                        masks = mask_gen.generate(rgb_enh)  # ← use enhanced image
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n  [OOM] Tile {tile_idx} skipped")
                        torch.cuda.empty_cache()
                        gc.collect()
                        tile_idx += 1
                        continue
                    raise

                recent_t.append(time.time() - t0)
                if len(recent_t) > 30:
                    recent_t.pop(0)

                n_this = 0
                for m in masks:
                    seg = m["segmentation"].astype(np.uint8)
                    if seg.sum() / (curr_w * curr_h) > MAX_MASK_COVERAGE:
                        continue

                    contours, _ = cv2.findContours(
                        seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
                    )
                    for cnt in contours:
                        if len(cnt) < 4:
                            continue
                        M = cv2.moments(cnt)
                        if M["m00"] == 0:
                            continue
                        cxg = M["m10"] / M["m00"] + x
                        cyg = M["m01"] / M["m00"] + y
                        if not centroid_in_core(cxg, cyg, core):
                            continue

                        coords = []
                        for pt in cnt.reshape(-1, 2):
                            gx, gy = rasterio.transform.xy(
                                transform, int(pt[1]) + y, int(pt[0]) + x
                            )
                            coords.append((gx, gy))
                        if len(coords) < 4:
                            continue

                        try:
                            poly = Polygon(coords)
                            if not poly.is_valid:
                                poly = make_valid(poly)
                                for piece in extract_polygons(poly):
                                    all_polygons.append(piece)
                                    n_this += 1
                            elif poly.area > 0:
                                all_polygons.append(poly)
                                n_this += 1
                        except Exception:
                            continue

                del masks, rgb, rgb_enh, tile_hwc, tile_chw
                gc.collect()

                tile_idx += 1
                active   += 1

                if active % CHECKPOINT_EVERY == 0:
                    save_checkpoint_async(all_polygons, tile_idx)
                    print(f"  [CKPT] {len(all_polygons):,} polygons @ tile {tile_idx}")

                avg_t     = float(np.mean(recent_t)) if recent_t else 0
                remaining = total_tiles - tile_idx
                eta_sec   = remaining * avg_t
                eta_clock = time.strftime("%H:%M", time.localtime(time.time() + eta_sec))
                print(
                    f"Tile {tile_idx:5d}/{total_tiles} | "
                    f"+{n_this:3d} | "
                    f"Total: {len(all_polygons):7,} | "
                    f"{avg_t:5.1f}s/tile | "
                    f"ETA {eta_sec/60:.0f} min  (by {eta_clock})"
                )

    sam_min = (time.time() - t_start) / 60
    print(f"\n{'='*62}")
    print(f"  SAM COMPLETE — {len(all_polygons):,} raw polygons  [{sam_min:.1f} min]")
    print(f"{'='*62}")

    if not all_polygons:
        print("[ERROR] Zero polygons detected.")
        return

    print("\n[BACKUP] Saving raw polygons ...")
    try:
        gpd.GeoDataFrame(geometry=all_polygons).to_file(
            OUTPUT_FILE.replace(".shp", "_backup_raw.shp")
        )
        print("  → _backup_raw.shp saved")
    except Exception as e:
        print(f"  → Backup failed: {e}")

    del mask_gen, sam
    torch.cuda.empty_cache()
    gc.collect()

    result = post_process(all_polygons, crs)
    if not result:
        print("[ERROR] Post-processing returned nothing.")
        return
    final, utm_crs = result

    print(f"\n[SAVE] Writing {len(final):,} polygons → {OUTPUT_FILE}")
    gdf = gpd.GeoDataFrame(geometry=final, crs=utm_crs)
    gdf["area_sqm"] = gdf.geometry.area.round(2)
    gdf.to_crs(crs).to_file(OUTPUT_FILE)

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*62}")
    print(f"  ✅  DONE!")
    print(f"  Output   : {OUTPUT_FILE}")
    print(f"  Polygons : {len(final):,}")
    print(f"  Runtime  : {total_min:.1f} min  ({total_min/60:.2f} h)")
    print(f"{'='*62}")
    print("⚠️  Delete checkpoint.pkl after verifying output.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Run again to resume from last checkpoint.")
    except Exception as e:
        import traceback
        err = OUTPUT_FILE.replace(".shp", "_error.txt")
        with open(err, "w") as f:
            f.write(traceback.format_exc())
        print(f"\n[FATAL] {e}\nTraceback → {err}")
        raise