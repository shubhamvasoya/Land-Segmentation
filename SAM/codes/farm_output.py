# -*- coding: utf-8 -*-
"""
Farm Boundary Detection — Clean Start
======================================
RTX 3050 4 GB  |  22,908 × 19,271 px image  |  ~4.5 hour runtime

Goals:
  • Detect as many farms as possible      → CLAHE preprocessing + low SAM thresholds
  • Clean polygons shaped like real farms → Douglas-Peucker + Chaikin smoothing
  • One polygon per farm                  → IoU-NMS (no merging, no clipping)
  • Thin boundaries → separate polygons   → CLAHE makes thin bunds visible to SAM
  • Adjacent farms share one boundary     → Gap-closing step
"""

import gc, os, pickle, threading, time, warnings
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

# ─────────────────────────────────────────────────────────────
#  PATHS  — only edit these
# ─────────────────────────────────────────────────────────────
IMAGE_PATH      = r"D:\BISAG\GUJ_235182318471_3_JAN_2021_C2EM.tif"
SAM_CHECKPOINT  = r"D:\BISAG\sam_vit_b_01ec64.pth"
OUTPUT_FILE     = r"D:\BISAG\farm_output.shp"
CHECKPOINT_FILE = r"D:\BISAG\farm_checkpoint.pkl"

# ─────────────────────────────────────────────────────────────
#  SAM  — locked for RTX 3050 4 GB
#  DO NOT raise points_per_batch above 64
#  DO NOT raise crop_n_layers above 0  (= 105 s/tile → too slow)
# ─────────────────────────────────────────────────────────────
POINTS_PER_SIDE  = 16     # 16×16 = 256 prompt points per tile
POINTS_PER_BATCH = 64     # max safe for 4 GB VRAM
CROP_N_LAYERS    = 0      # our tiling loop IS the chunking
IOU_THRESHOLD    = 0.75   # lower = SAM keeps more farm masks
STABILITY        = 0.80   # lower = SAM keeps more farm masks
MIN_MASK_PX      = 50     # minimum mask size inside SAM (pixels)

# ─────────────────────────────────────────────────────────────
#  TILING
#  512 px blocks, 96 px overlap → stride 416
#  ~2,632 tiles × ~6 s/tile = ~264 min SAM
#  + ~20 min post-processing  = ~4.7 hours total  ✅
# ─────────────────────────────────────────────────────────────
TILE   = 512
OVERLAP = 96
STRIDE  = TILE - OVERLAP    # 416 px

# ─────────────────────────────────────────────────────────────
#  PREPROCESSING
#  CLAHE makes thin bunds/paths visible to SAM so it splits
#  adjacent farms correctly instead of merging them.
# ─────────────────────────────────────────────────────────────
CLAHE_CLIP   = 3.0   # contrast amplification limit
CLAHE_GRID   = 8     # local tile size for CLAHE (pixels)
SHARP_WEIGHT = 0.5   # unsharp mask strength (0 = off, 1 = full)
SHARP_SIGMA  = 1.5   # blur radius for unsharp mask

# ─────────────────────────────────────────────────────────────
#  POST-PROCESSING
# ─────────────────────────────────────────────────────────────
MIN_AREA_SQM   = 500    # drop polygons smaller than this (noise)
MAX_AREA_SQM   = 80_000 # drop polygons larger than this (background)
NMS_THRESHOLD  = 0.50   # suppress smaller polygon if this fraction is covered
SIMPLIFY_M     = 1.5    # Douglas-Peucker tolerance in metres
CHAIKIN_PASSES = 2      # corner-cutting passes (2 = clean, not over-rounded)
GAP_M          = 0.8    # expand each polygon this many metres to touch neighbour

SAVE_EVERY = 25   # checkpoint every N tiles


# ═════════════════════════════════════════════════════════════
#  STEP 1 — PREPROCESSING
# ═════════════════════════════════════════════════════════════

def to_uint8(bands: np.ndarray) -> np.ndarray:
    """
    Satellite bands (any count) → uint8 HWC RGB.
    Uses 2-98 percentile stretch so bright/dark tiles look consistent.
    """
    if   bands.shape[0] > 3: bands = bands[:3]
    elif bands.shape[0] == 1: bands = np.repeat(bands, 3, axis=0)
    elif bands.shape[0] == 2: bands = np.concatenate([bands, bands[:1]], axis=0)

    out = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        b = bands[i].astype(np.float32)
        v = b[b > 0]
        if v.size > 100:
            lo, hi = np.percentile(v, [2, 98])
            b = np.clip((b - lo) / max(hi - lo, 1e-5), 0, 1)
        else:
            b = b / max(b.max(), 1.0)
        out[:, :, i] = (b * 255).astype(np.uint8)
    return out


def enhance(rgb: np.ndarray) -> np.ndarray:
    """
    CLAHE + unsharp mask → makes thin field boundaries visible to SAM.

    Without this, a 1-2 pixel bund between two farms often looks
    like a texture variation to SAM, so it draws one polygon over both.
    CLAHE computes contrast locally (in 8×8 px windows) so that thin
    dark lines stay dark relative to their immediate surroundings —
    SAM then detects them as boundaries and creates separate polygons.

    Final blend: 65 % enhanced + 35 % original to avoid false edges
    in uniform areas like water bodies.
    """
    clahe  = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                              tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    enh = np.zeros_like(rgb)
    for i in range(3):
        enh[:, :, i] = clahe.apply(rgb[:, :, i])

    blur     = cv2.GaussianBlur(enh.astype(np.float32), (0, 0), SHARP_SIGMA)
    sharp    = np.clip(enh.astype(np.float32) * (1 + SHARP_WEIGHT)
                       - blur * SHARP_WEIGHT, 0, 255).astype(np.uint8)

    return cv2.addWeighted(sharp, 0.65, rgb, 0.35, 0)


# ═════════════════════════════════════════════════════════════
#  STEP 2 — TILING HELPERS
# ═════════════════════════════════════════════════════════════

def core_of(tx, ty, tw, th, img_w, img_h):
    """
    Every tile owns a non-overlapping core strip (OVERLAP/2 px inside each edge).
    A polygon is kept only if its centroid is inside this core.
    Adjacent cores tile the full image with zero gaps and zero duplicates.
    This is what prevents the grid-pattern artifact in the output.
    """
    m   = OVERLAP // 2
    x0  = tx       if tx == 0             else tx + m
    y0  = ty       if ty == 0             else ty + m
    x1  = tx + tw  if tx + TILE >= img_w  else tx + tw - m
    y1  = ty + th  if ty + TILE >= img_h  else ty + th - m
    return x0, y0, x1, y1


def in_core(cx, cy, core):
    return core[0] <= cx < core[2] and core[1] <= cy < core[3]


def as_polygons(geom, min_area=0):
    """Extract all Polygon parts from any Shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom] if geom.area > min_area else []
    if geom.geom_type == "MultiPolygon":
        return [g for g in geom.geoms if g.area > min_area]
    if geom.geom_type == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(as_polygons(g, min_area))
        return out
    return []


# ═════════════════════════════════════════════════════════════
#  STEP 3 — POST-PROCESSING FUNCTIONS
# ═════════════════════════════════════════════════════════════

def nms(polys: list) -> list:
    """
    Non-Maximum Suppression — keeps one polygon per farm.

    Sorts by area (largest first). For each polygon, checks all
    smaller candidates via STRtree. If more than NMS_THRESHOLD of
    a smaller polygon's area is covered by the current polygon,
    the smaller one is removed entirely — never clipped, never merged.

    One intact polygon per farm. O(n log n).
    """
    if not polys:
        return []

    n      = len(polys)
    order  = sorted(range(n), key=lambda i: polys[i].area, reverse=True)
    sp     = [polys[i] for i in order]
    kill   = [False] * n
    tree   = STRtree(sp)

    import shapely
    v2 = int(shapely.__version__.split(".")[0]) >= 2

    for i in range(n):
        if kill[i]:
            continue
        pi    = sp[i]
        hits  = tree.query(pi, predicate="intersects") if v2 else tree.query(pi)
        for j in hits:
            if j <= i or kill[j]:
                continue
            pj = sp[j]
            try:
                inter = pi.intersection(pj)
                if not inter.is_empty and inter.area / pj.area > NMS_THRESHOLD:
                    kill[j] = True
            except Exception:
                kill[j] = True

    return [p for p, k in zip(sp, kill) if not k]


def close_gaps(polys: list) -> list:
    """
    Expand each polygon by GAP_M metres to touch its neighbours,
    then subtract already-committed polygons so farms never overlap.
    Result: two adjacent farms share exactly one boundary line.

    join_style=2 (mitre) keeps sharp agricultural corners.
    Largest farm commits first — consistent with NMS priority.
    """
    if not polys:
        return []

    import shapely
    v2 = int(shapely.__version__.split(".")[0]) >= 2

    order  = sorted(range(len(polys)), key=lambda i: polys[i].area, reverse=True)
    sp     = [polys[i] for i in order]
    tree   = STRtree(sp)
    done   = {}
    out    = [None] * len(sp)

    for rank, p in enumerate(sp):
        try:
            exp = p.buffer(GAP_M, join_style=2, mitre_limit=2.0)
        except Exception:
            done[rank] = p;  out[rank] = p;  continue

        hits = list(tree.query(exp, predicate="intersects") if v2
                    else tree.query(exp))
        for j in hits:
            if j >= rank or j not in done:
                continue
            try:
                exp = exp.difference(done[j])
                if exp.is_empty:
                    break
            except Exception:
                try:
                    exp = make_valid(exp).difference(make_valid(done[j]))
                except Exception:
                    break

        if exp is None or exp.is_empty:
            exp = p

        parts = as_polygons(exp, min_area=MIN_AREA_SQM * 0.3)
        best  = min(parts, key=lambda x: x.centroid.distance(p.centroid)) \
                if parts else p

        done[rank] = best
        out[rank]  = best

    return [o for o in out if o is not None]


def smooth(poly: Polygon) -> Polygon:
    """
    Douglas-Peucker simplify → Chaikin corner-cutting.
    DP removes pixel staircase.  Chaikin softens remaining jagged edges.
    2 Chaikin passes = clean without over-rounding real farm corners.
    """
    s = poly.simplify(SIMPLIFY_M, preserve_topology=True)
    if s.is_empty or s.geom_type != "Polygon":
        return poly

    def chaikin(coords):
        pts = np.array(list(coords), dtype=np.float64)
        if np.allclose(pts[0], pts[-1]) and len(pts) > 1:
            pts = pts[:-1]
        for _ in range(CHAIKIN_PASSES):
            if len(pts) < 3:
                break
            p1 = np.roll(pts, -1, axis=0)
            q  = 0.75 * pts + 0.25 * p1
            r  = 0.25 * pts + 0.75 * p1
            new       = np.empty((len(pts) * 2, 2))
            new[0::2] = q
            new[1::2] = r
            pts = new
        return pts.tolist() + [pts[0].tolist()]

    try:
        ext    = chaikin(s.exterior.coords)
        holes  = [chaikin(h.coords) for h in s.interiors]
        result = Polygon(ext, holes)
        return result if result.is_valid and result.area >= MIN_AREA_SQM else poly
    except Exception:
        return poly


# ═════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ═════════════════════════════════════════════════════════════

def save(polys, tile_idx):
    """Save checkpoint in background thread — never blocks inference."""
    snap = list(polys)
    def _go():
        tmp = CHECKPOINT_FILE + ".tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump({"polys": snap, "tile": tile_idx}, f)
            os.replace(tmp, CHECKPOINT_FILE)
        except Exception as e:
            print(f"  [CKPT] {e}")
    threading.Thread(target=_go, daemon=True).start()


def load():
    if not os.path.exists(CHECKPOINT_FILE):
        return [], 0
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            d = pickle.load(f)
        # support both old key ("tile_idx") and new key ("tile")
        tile = d.get("tile", d.get("tile_idx", 0))
        polys = d.get("polys", d.get("polygons", []))
        print(f"[RESUME] {len(polys):,} polygons, resuming from tile {tile}")
        return polys, tile
    except Exception as e:
        print(f"[RESUME] Failed to load checkpoint: {e}")
        return [], 0


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 60)
    print("  FARM BOUNDARY DETECTION")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Tile    : {TILE}px  overlap={OVERLAP}px  stride={STRIDE}px")
    print(f"  SAM     : pts={POINTS_PER_SIDE}  batch={POINTS_PER_BATCH}"
          f"  iou={IOU_THRESHOLD}  stab={STABILITY}")
    print(f"  CLAHE   : clip={CLAHE_CLIP}  grid={CLAHE_GRID}px")
    print(f"  Smooth  : DP={SIMPLIFY_M}m  Chaikin×{CHAIKIN_PASSES}")
    print(f"  Gap     : {GAP_M} m")
    print(f"  Area    : {MIN_AREA_SQM}–{MAX_AREA_SQM} sqm")
    print("=" * 60)

    if DEVICE == "cuda":
        g = torch.cuda.get_device_properties(0)
        print(f"\n[GPU] {g.name}  {g.total_memory/1e9:.1f} GB VRAM")

    # Image metadata
    with rasterio.open(IMAGE_PATH) as src:
        W, H       = src.width, src.height
        native_crs = src.crs
        transform  = src.transform

        if native_crs.is_geographic:
            from pyproj import CRS
            l, b, r, t2 = src.bounds
            cx   = (l + r) / 2;  cy = (b + t2) / 2
            zone = int((cx + 180) / 6) + 1
            epsg = 32600 + zone if cy > 0 else 32700 + zone
            utm_crs = CRS.from_epsg(epsg)
            print(f"\n[IMAGE] {W:,} × {H:,} px  |  UTM EPSG:{epsg}")
        else:
            utm_crs = native_crs
            print(f"\n[IMAGE] {W:,} × {H:,} px")

    xs     = list(range(0, W, STRIDE))
    ys     = list(range(0, H, STRIDE))
    total  = len(xs) * len(ys)
    spt    = 6.5   # seconds per tile (empirical for RTX 3050, pts=16, batch=64)
    est_h  = (total * spt + 1200) / 3600
    print(f"[TILES] {len(xs)} × {len(ys)} = {total:,}  |  Est. {est_h:.1f} h\n")

    # Resume
    all_polys, start = load()

    # Load SAM
    print("[SAM] Loading model ...")
    torch.cuda.empty_cache()
    sam  = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    gen  = SamAutomaticMaskGenerator(
        model                          = sam,
        points_per_side                = POINTS_PER_SIDE,
        points_per_batch               = POINTS_PER_BATCH,
        pred_iou_thresh                = IOU_THRESHOLD,
        stability_score_thresh         = STABILITY,
        crop_n_layers                  = CROP_N_LAYERS,
        crop_n_points_downscale_factor = 2,
        min_mask_region_area           = MIN_MASK_PX,
    )
    print("[SAM] Ready\n")

    # ── Inference loop ────────────────────────────────────────
    with rasterio.open(IMAGE_PATH) as src:

        idx      = 0
        active   = 0
        recent   = []

        for y in ys:
            for x in xs:

                if idx < start:
                    idx += 1
                    continue

                th = min(TILE, H - y)
                tw = min(TILE, W - x)
                raw = src.read(window=Window(x, y, tw, th))

                # Skip blank tiles
                if raw.max() < 5 or raw.std() < 5:
                    idx += 1
                    continue

                # Preprocess
                rgb = to_uint8(raw)
                inp = enhance(rgb)

                core = core_of(x, y, tw, th, W, H)

                # SAM
                torch.cuda.empty_cache()
                t1 = time.time()
                try:
                    with torch.inference_mode():
                        masks = gen.generate(inp)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  [OOM] tile {idx} — skipped")
                        torch.cuda.empty_cache();  gc.collect()
                        idx += 1;  continue
                    raise

                recent.append(time.time() - t1)
                if len(recent) > 40:
                    recent.pop(0)

                # Masks → polygons
                n_new = 0
                for m in masks:
                    seg = m["segmentation"].astype(np.uint8)

                    # Skip masks that cover most of the tile (background)
                    if seg.sum() / (tw * th) > 0.85:
                        continue

                    cnts, _ = cv2.findContours(
                        seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
                    )
                    for cnt in cnts:
                        if len(cnt) < 4:
                            continue

                        M = cv2.moments(cnt)
                        if M["m00"] == 0:
                            continue

                        # Only keep polygon if its centroid is in this tile's core
                        cxg = M["m10"] / M["m00"] + x
                        cyg = M["m01"] / M["m00"] + y
                        if not in_core(cxg, cyg, core):
                            continue

                        # Pixel coordinates → geo coordinates
                        pts    = cnt.reshape(-1, 2)
                        coords = [
                            rasterio.transform.xy(transform, int(p[1]) + y, int(p[0]) + x)
                            for p in pts
                        ]
                        if len(coords) < 4:
                            continue

                        try:
                            poly = Polygon(coords)
                            if not poly.is_valid:
                                poly = make_valid(poly)
                                all_polys.extend(as_polygons(poly))
                                n_new += len(as_polygons(poly))
                            elif poly.area > 0:
                                all_polys.append(poly)
                                n_new += 1
                        except Exception:
                            continue

                del masks, inp, rgb, raw
                gc.collect()

                idx    += 1
                active += 1

                # Checkpoint
                if active % SAVE_EVERY == 0:
                    save(all_polys, idx)

                # Progress
                avg  = np.mean(recent) if recent else spt
                eta  = (total - idx) * avg
                done = time.strftime("%H:%M", time.localtime(time.time() + eta))
                print(
                    f"  tile {idx:5d}/{total}"
                    f"  +{n_new:3d}"
                    f"  total={len(all_polys):7,}"
                    f"  {avg:.1f}s/tile"
                    f"  ETA {eta/60:.0f}min ({done})"
                )

    print(f"\n[SAM done] {len(all_polys):,} raw polygons"
          f"  in {(time.time()-t0)/60:.0f} min")

    if not all_polys:
        print("[ERROR] No polygons detected — check image path and band values.")
        return

    # Backup
    try:
        gpd.GeoDataFrame(geometry=all_polys).to_file(
            OUTPUT_FILE.replace(".shp", "_raw_backup.shp"))
        print("[BACKUP] Raw polygons saved.")
    except Exception as e:
        print(f"[BACKUP] Failed: {e}")

    del gen, sam
    torch.cuda.empty_cache();  gc.collect()

    # ── Post-processing ───────────────────────────────────────
    bar = "=" * 60
    print(f"\n{bar}\n  POST-PROCESSING\n{bar}")

    # 1. Project to UTM + area filter
    print("\n[1/5] Area filter ...")
    t1      = time.time()
    gdf     = gpd.GeoDataFrame(geometry=all_polys, crs=native_crs)
    gdf_utm = gdf.to_crs(utm_crs)
    a       = gdf_utm.geometry.area
    mask    = (a >= MIN_AREA_SQM) & (a <= MAX_AREA_SQM)
    gdf_utm = gdf_utm[mask].reset_index(drop=True)
    print(f"     {len(gdf_utm):,} remain  ({(~mask).sum():,} removed)"
          f"  [{time.time()-t1:.1f}s]")

    if len(gdf_utm) == 0:
        print("     No polygons after area filter!");  return

    # 2. Validate
    print("\n[2/5] Geometry validation ...")
    t1    = time.time()
    valid = []
    for p in gdf_utm.geometry:
        if not p.is_valid:
            p = make_valid(p)
        valid.extend(as_polygons(p, MIN_AREA_SQM))
    print(f"     {len(valid):,} valid polygons  [{time.time()-t1:.1f}s]")

    # 3. NMS — one polygon per farm
    print("\n[3/5] NMS overlap removal ...")
    t1   = time.time()
    kept = nms(valid)
    print(f"     {len(valid):,} → {len(kept):,} polygons  [{time.time()-t1:.1f}s]")

    # 4. Gap closing — shared boundaries
    print(f"\n[4/5] Gap closing ({GAP_M} m) ...")
    t1      = time.time()
    gapless = close_gaps(kept)
    print(f"     Done  [{time.time()-t1:.1f}s]")

    # 5. Smooth — clean farm shapes
    print(f"\n[5/5] Smoothing (DP={SIMPLIFY_M}m, Chaikin×{CHAIKIN_PASSES}) ...")
    t1       = time.time()
    smoothed = []
    for p in gapless:
        s = smooth(p)
        smoothed.extend(as_polygons(s, MIN_AREA_SQM))
    print(f"     {len(smoothed):,} final polygons  [{time.time()-t1:.1f}s]")

    # Final validity pass
    final = []
    for p in smoothed:
        if not p.is_valid:
            p = make_valid(p)
        final.extend(as_polygons(p, MIN_AREA_SQM))

    # Save
    print(f"\n[SAVE] {len(final):,} polygons → {OUTPUT_FILE}")
    gdf_out = gpd.GeoDataFrame(geometry=final, crs=utm_crs)
    gdf_out["area_sqm"] = gdf_out.geometry.area.round(1)
    gdf_out.to_crs(native_crs).to_file(OUTPUT_FILE)

    elapsed = (time.time() - t0) / 60
    print(f"\n{bar}")
    print(f"  ✅  DONE")
    print(f"  Output   : {OUTPUT_FILE}")
    print(f"  Polygons : {len(final):,}")
    print(f"  Runtime  : {elapsed:.0f} min  ({elapsed/60:.2f} h)")
    print(f"{bar}")
    print("\n⚠️  Delete farm_checkpoint.pkl after you have verified the output.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] — run again to resume from last checkpoint.")
    except Exception as e:
        import traceback
        err = OUTPUT_FILE.replace(".shp", "_error.txt")
        with open(err, "w") as f:
            f.write(traceback.format_exc())
        print(f"\n[FATAL] {e}\nFull traceback → {err}")
        raise