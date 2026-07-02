# -*- coding: utf-8 -*-
"""
Farm Boundary Detection - Dual-Pass Fusion Pipeline
=====================================================
FIXES APPLIED:
  [1] PROJ_DATA set at top before any imports → fixes proj.db error
  [2] Resume looks for .gpkg not .shp
  [3] else branch never deletes tile log if .gpkg already exists
  [4] Checkpoint load tries multiple fallback strategies for corrupted .gpkg
  [5] _save_chk strips CRS if write fails (proj.db fallback)
"""

# ── Fix PROJ_DATA FIRST before any geo imports ──────────────────
import os, sys
_PROJ_CANDIDATES = [
    r"C:\Users\Dishan\anaconda3\envs\sam_env\Library\share\proj",
    r"C:\Users\Dishan\anaconda3\pkgs\proj\Library\share\proj",
    r"C:\Users\Dishan\anaconda3\Library\share\proj",
]
if "PROJ_DATA" not in os.environ:
    for _p in _PROJ_CANDIDATES:
        if os.path.exists(os.path.join(_p, "proj.db")):
            os.environ["PROJ_DATA"] = _p
            os.environ["PROJ_LIB"]  = _p
            print(f"[PROJ] Set PROJ_DATA = {_p}")
            break
    else:
        print("[PROJ] WARNING: proj.db not found in known locations. "
              "Set PROJ_DATA manually if CRS errors occur.")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import rasterio
import cv2
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.strtree import STRtree
from rasterio.windows import Window
import gc
import time
import warnings

warnings.filterwarnings("ignore")

# ── Prefer pyogrio over fiona ───────────────────────────────────
try:
    import pyogrio
    gpd.options.io_engine = "pyogrio"
    _SAVE_ENGINE = "pyogrio"
    print("[IO] Using pyogrio engine for all file I/O.")
except ImportError:
    _SAVE_ENGINE = None
    print("[IO] pyogrio not found – falling back to fiona.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("[FATAL] segment-anything not installed.")
    sys.exit(1)

# ============================================================
# ▶ CONFIGURATION
# ============================================================
IMAGE_PATH     = r"D:\BISAG\GUJ_235182318471_3_JAN_2021_C2EM.tif"
SAM_CHECKPOINT = r"D:\BISAG\sam_vit_b_01ec64.pth"
OUTPUT_FILE    = r"D:\BISAG\farm_dual_pass.shp"

SAM_MODEL_TYPE = "vit_b"

BLOCK_SIZE    = 512
BLOCK_OVERLAP = 128
STRIDE        = BLOCK_SIZE - BLOCK_OVERLAP  # 384

# Pass 1 – Quality
Q_POINTS_PER_SIDE      = 32
Q_POINTS_PER_BATCH     = 64
Q_PRED_IOU_THRESH      = 0.86
Q_STABILITY_THRESH     = 0.90
Q_CROP_N_LAYERS        = 1
Q_CROP_DOWNSCALE       = 2
Q_MIN_MASK_AREA        = 100

# Pass 2 – Quantity
P_POINTS_PER_SIDE      = 48
P_POINTS_PER_BATCH     = 32
P_PRED_IOU_THRESH      = 0.76
P_STABILITY_THRESH     = 0.82
P_CROP_N_LAYERS        = 1
P_CROP_DOWNSCALE       = 2
P_MIN_MASK_AREA        = 60

# Post-processing
MIN_FARM_AREA_SQM  = 300
MAX_FARM_AREA_SQM  = 80000
MAX_MASK_COVERAGE  = 0.85
SMOOTH_TOLERANCE   = 1.5
CHAIKIN_ITERS      = 3

# Fusion
QUANTITY_OVERLAP_THRESH = 0.50

# Gap filling
FILL_GAPS    = True
GAP_MIN_AREA = 50

# Checkpointing
CHECKPOINT_INTERVAL = 50
PROGRESS_INTERVAL   = 10


# ============================================================
# UTILITIES
# ============================================================

def normalize_image(img_bands):
    if img_bands.shape[0] > 3:
        img_bands = img_bands[:3]
    elif img_bands.shape[0] == 1:
        img_bands = np.repeat(img_bands, 3, axis=0)
    out = np.zeros_like(img_bands, dtype=np.uint8)
    for i in range(img_bands.shape[0]):
        band = img_bands[i].astype(np.float32)
        valid = band[band > 0]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            band = np.clip((band - p2) / (p98 - p2 + 1e-5), 0, 1)
        out[i] = (band * 255).astype(np.uint8)
    return np.transpose(out, (1, 2, 0))


def chaikin_smooth(coords, iterations=CHAIKIN_ITERS):
    coords = list(coords)
    for _ in range(iterations):
        if len(coords) < 3:
            return coords
        new_coords = []
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i + 1]
            new_coords.append((0.75*p0[0]+0.25*p1[0], 0.75*p0[1]+0.25*p1[1]))
            new_coords.append((0.25*p0[0]+0.75*p1[0], 0.25*p0[1]+0.75*p1[1]))
        new_coords.append(new_coords[0])
        coords = new_coords
    return coords


def get_core_bounds(tile_x, tile_y, tile_w, tile_h, img_w, img_h):
    half = BLOCK_OVERLAP // 2
    cx0 = tile_x if tile_x == 0 else tile_x + half
    cy0 = tile_y if tile_y == 0 else tile_y + half
    cx1 = (tile_x + tile_w) if (tile_x + BLOCK_SIZE >= img_w) else (tile_x + tile_w - half)
    cy1 = (tile_y + tile_h) if (tile_y + BLOCK_SIZE >= img_h) else (tile_y + tile_h - half)
    return cx0, cy0, cx1, cy1


def centroid_in_core(cx, cy, core):
    return core[0] <= cx < core[2] and core[1] <= cy < core[3]


def extract_polygons(geom, min_area=0):
    out = []
    if geom is None or geom.is_empty:
        return out
    if geom.geom_type == 'Polygon':
        if geom.area > min_area:
            out.append(geom)
    elif geom.geom_type == 'MultiPolygon':
        for g in geom.geoms:
            if g.area > min_area:
                out.append(g)
    elif geom.geom_type == 'GeometryCollection':
        for g in geom.geoms:
            out.extend(extract_polygons(g, min_area))
    return out


def aspect_ratio(poly):
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    if h == 0:
        return 999
    return max(w / h, h / w)


# ============================================================
# CHECKPOINT SAVE  –  robust against proj.db issues
# ============================================================

def _save_chk(polys, crs, path, count, label):
    """Save checkpoint .gpkg with multiple CRS fallback strategies."""
    chk_path = path.replace(".shp", ".gpkg")
    try:
        gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)

        # Strategy 1: normal save
        try:
            gdf.to_file(chk_path, driver="GPKG", engine=_SAVE_ENGINE)
            print(f"  [{label}] Checkpoint saved: {len(polys)} polys after {count} tiles")
            return
        except Exception as e1:
            print(f"  [{label}] Checkpoint save attempt 1 failed: {e1}")

        # Strategy 2: save with CRS as EPSG string
        try:
            epsg = crs.to_epsg() if hasattr(crs, 'to_epsg') else None
            if epsg:
                gdf2 = gpd.GeoDataFrame(geometry=polys)
                gdf2 = gdf2.set_crs(f"EPSG:{epsg}", allow_override=True)
                gdf2.to_file(chk_path, driver="GPKG", engine=_SAVE_ENGINE)
                print(f"  [{label}] Checkpoint saved (EPSG fallback): {len(polys)} polys")
                return
        except Exception as e2:
            print(f"  [{label}] Checkpoint save attempt 2 failed: {e2}")

        # Strategy 3: save without CRS (geometries still preserved)
        try:
            gdf3 = gpd.GeoDataFrame(geometry=polys)
            gdf3.to_file(chk_path, driver="GPKG", engine=_SAVE_ENGINE)
            print(f"  [{label}] Checkpoint saved (no CRS): {len(polys)} polys after {count} tiles")
            print(f"  [{label}] WARNING: CRS not saved – will be re-attached on resume.")
        except Exception as e3:
            print(f"  [{label}] All checkpoint save strategies failed: {e3}")

    except Exception as e:
        print(f"  [{label}] Checkpoint failed entirely: {e}")


# ============================================================
# CHECKPOINT LOAD  –  robust against corrupted / no-CRS .gpkg
# ============================================================

def _load_chk(chk_path, native_crs, label):
    """
    Try multiple strategies to load a checkpoint .gpkg.
    Returns list of geometries or empty list on failure.
    """
    if not os.path.exists(chk_path):
        return None  # signals: no checkpoint found

    print(f"[{label}] Found checkpoint file. Attempting to load...")

    # Strategy 1: normal read
    try:
        gdf = gpd.read_file(chk_path, engine=_SAVE_ENGINE)
        polys = list(gdf.geometry)
        print(f"[{label}] Loaded {len(polys)} polygons (strategy 1 – normal).")
        return polys
    except Exception as e1:
        print(f"[{label}] Load strategy 1 failed: {e1}")

    # Strategy 2: read with fiona directly
    try:
        import fiona
        with fiona.open(chk_path) as src:
            polys = []
            for feat in src:
                from shapely.geometry import shape
                geom = shape(feat['geometry'])
                if not geom.is_empty:
                    polys.append(geom)
        print(f"[{label}] Loaded {len(polys)} polygons (strategy 2 – fiona).")
        return polys
    except Exception as e2:
        print(f"[{label}] Load strategy 2 failed: {e2}")

    # Strategy 3: pyogrio with no CRS enforcement
    try:
        import pyogrio
        gdf = pyogrio.read_dataframe(chk_path, use_arrow=False)
        polys = list(gdf.geometry)
        print(f"[{label}] Loaded {len(polys)} polygons (strategy 3 – pyogrio raw).")
        return polys
    except Exception as e3:
        print(f"[{label}] Load strategy 3 failed: {e3}")

    print(f"[{label}] All load strategies failed. Starting fresh for this pass.")
    return []


# ============================================================
# SAM INFERENCE
# ============================================================

def run_sam_on_tile(mask_generator, rgb, x, y, img_w, img_h,
                    curr_w, curr_h, transform):
    core = get_core_bounds(x, y, curr_w, curr_h, img_w, img_h)

    masks = None
    orig_batch = mask_generator.points_per_batch
    for b_size in [orig_batch, 32, 16, 4]:
        try:
            mask_generator.points_per_batch = b_size
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    masks = mask_generator.generate(rgb)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                del e
                torch.cuda.empty_cache()
                gc.collect()
                if b_size > 4:
                    print(f"    [OOM] batch={b_size} failed. Retrying...")
                continue
            raise
    mask_generator.points_per_batch = orig_batch

    if masks is None:
        print("    [FATAL OOM] Skipping tile.")
        return []

    polys = []
    for m in masks:
        seg = m['segmentation'].astype(np.uint8)
        if np.sum(seg) / (curr_w * curr_h) > MAX_MASK_COVERAGE:
            continue

        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 4:
                continue
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx_g = M['m10'] / M['m00'] + x
            cy_g = M['m01'] / M['m00'] + y
            if not centroid_in_core(cx_g, cy_g, core):
                continue

            coords = []
            for pt in contour:
                px, py = pt[0][0] + x, pt[0][1] + y
                wx, wy = rasterio.transform.xy(transform, py, px)
                coords.append((wx, wy))
            if len(coords) < 4:
                continue

            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                polys.extend(extract_polygons(poly))
            except Exception:
                continue

    del masks
    return polys


# ============================================================
# POST-PROCESSING
# ============================================================

def area_filter(polys_geo, utm_crs, native_crs, label=""):
    gdf = gpd.GeoDataFrame(geometry=polys_geo, crs=native_crs)
    gdf = gdf.to_crs(utm_crs)
    gdf['area'] = gdf.geometry.area
    before = len(gdf)
    gdf = gdf[(gdf['area'] >= MIN_FARM_AREA_SQM) &
              (gdf['area'] <= MAX_FARM_AREA_SQM)].copy()
    gdf['ratio'] = gdf.geometry.apply(aspect_ratio)
    gdf = gdf[gdf['ratio'] < 6].drop(columns=['ratio'])
    print(f"  [{label}] Area filter: {before} → {len(gdf)} polygons")
    return gdf.reset_index(drop=True)


def remove_overlaps_priority(polys, min_area):
    polys = sorted(polys, key=lambda p: p.area, reverse=True)
    placed = []
    tree = None
    for i, poly in enumerate(polys):
        if not poly.is_valid:
            poly = make_valid(poly)
        if placed:
            if tree is None or i % 500 == 0:
                tree = STRtree(placed)
            for idx in tree.query(poly):
                neighbor = placed[idx]
                if poly.intersects(neighbor):
                    try:
                        poly = poly.difference(neighbor)
                    except Exception:
                        try:
                            poly = make_valid(poly).difference(make_valid(neighbor))
                        except Exception:
                            break
            survivors = extract_polygons(poly, min_area)
            placed.extend(survivors)
        else:
            placed.append(poly)
        if (i + 1) % 5000 == 0:
            print(f"    ... {i+1}/{len(polys)} checked, {len(placed)} placed")
    return placed


def smooth_polygon(poly):
    try:
        s = poly.simplify(SMOOTH_TOLERANCE, preserve_topology=True)
        if s.is_empty or s.area < MIN_FARM_AREA_SQM:
            return None
        if s.geom_type == 'Polygon':
            ext = chaikin_smooth(s.exterior.coords)
            ints = [chaikin_smooth(h.coords) for h in s.interiors]
            result = Polygon(ext, ints)
            return result if result.is_valid and result.area > MIN_FARM_AREA_SQM else s
        elif s.geom_type == 'MultiPolygon':
            parts = []
            for part in s.geoms:
                ext = chaikin_smooth(part.exterior.coords)
                result = Polygon(ext)
                if result.is_valid and result.area > MIN_FARM_AREA_SQM:
                    parts.append(result)
            return parts[0] if len(parts) == 1 else (
                MultiPolygon(parts) if parts else None)
    except Exception:
        return poly if poly.is_valid else None


def fill_gaps_voronoi(polys, scene_bounds, min_gap_area):
    print("  [Gap Fill] Computing farm union and gaps...")
    farm_union = unary_union(polys)
    scene_poly = box(*scene_bounds)
    gaps = scene_poly.difference(farm_union)
    if gaps.is_empty:
        print("  [Gap Fill] No gaps found.")
        return polys
    gap_parts = extract_polygons(gaps, min_gap_area)
    print(f"  [Gap Fill] Found {len(gap_parts)} gap regions to fill")
    if not gap_parts:
        return polys

    tree = STRtree(polys)
    updated = list(polys)
    for gap in gap_parts:
        candidates = tree.query(gap.buffer(5))
        if len(candidates) == 0:
            continue
        best_idx, best_contact = None, 0.0
        for idx in candidates:
            try:
                contact = polys[idx].buffer(1).intersection(gap).area
                if contact > best_contact:
                    best_contact = contact
                    best_idx = idx
            except Exception:
                continue
        if best_idx is not None:
            try:
                merged = updated[best_idx].union(gap)
                if merged.is_valid and merged.geom_type in ('Polygon', 'MultiPolygon'):
                    updated[best_idx] = merged
            except Exception:
                pass
    print(f"  [Gap Fill] Done. Polygon count: {len(updated)}")
    return updated


# ============================================================
# FUSION
# ============================================================

def fuse_passes(quality_polys, quantity_polys, min_area):
    print(f"\n[FUSION] Quality: {len(quality_polys)}, Quantity: {len(quantity_polys)}")
    placed = list(quality_polys)
    tree = STRtree(placed)
    added = 0
    for i, qpoly in enumerate(quantity_polys):
        if not qpoly.is_valid:
            qpoly = make_valid(qpoly)
        candidates = tree.query(qpoly)
        overlap_area = 0.0
        for idx in candidates:
            try:
                overlap_area += qpoly.intersection(placed[idx]).area
            except Exception:
                continue
        if qpoly.area > 0 and overlap_area / qpoly.area > QUANTITY_OVERLAP_THRESH:
            continue
        clipped = qpoly
        for idx in candidates:
            try:
                clipped = clipped.difference(placed[idx])
            except Exception:
                try:
                    clipped = make_valid(clipped).difference(make_valid(placed[idx]))
                except Exception:
                    break
        for s in extract_polygons(clipped, min_area):
            placed.append(s)
            added += 1
        if (i + 1) % 1000 == 0:
            tree = STRtree(placed)
            print(f"    ... {i+1}/{len(quantity_polys)} quantity checked, {added} added")
    print(f"[FUSION] Added {added} quantity polygons. Total: {len(placed)}")
    return placed


# ============================================================
# PASS RUNNER
# ============================================================

def _log(tile_log, tid, processed_set):
    try:
        with open(tile_log, "a") as f:
            f.write(f"{tid}\n")
        processed_set.add(tid)
    except Exception:
        pass


def _run_pass(mask_gen, src, transform, img_w, img_h,
              x_starts, y_starts, total_tiles,
              checkpoint_shp, tile_log, label=""):

    all_polys = []
    processed = set()
    native_crs = src.crs

    # ── Resume logic (FIX: always use .gpkg) ──────────────────
    checkpoint_gpkg = checkpoint_shp.replace(".shp", ".gpkg")

    if os.path.exists(checkpoint_gpkg) and os.path.exists(tile_log):
        # Load polygons from checkpoint
        loaded = _load_chk(checkpoint_gpkg, native_crs, label)
        if loaded is not None:
            all_polys = loaded

        # Load processed tile IDs
        try:
            with open(tile_log, "r") as f:
                processed = set(l.strip() for l in f if l.strip())
            print(f"[{label}] Skipping {len(processed)} processed tiles.")
        except Exception:
            print(f"[{label}] Could not read tile log – will reprocess all tiles.")

    elif os.path.exists(checkpoint_gpkg) and not os.path.exists(tile_log):
        # Checkpoint exists but tile log was lost — load polygons, reprocess all tiles
        print(f"[{label}] Tile log missing but checkpoint found. "
              f"Loading polygons, reprocessing all tiles (duplicates will be cleaned later).")
        loaded = _load_chk(checkpoint_gpkg, native_crs, label)
        if loaded is not None:
            all_polys = loaded

    else:
        # Fresh start — clean up any stale .shp fragments only
        print(f"[{label}] No checkpoint found. Starting fresh.")
        for ext in [".cpg", ".dbf", ".prj", ".shp", ".shx"]:
            p = checkpoint_shp.replace(".shp", ext)
            if os.path.exists(p):
                os.remove(p)
        # Do NOT remove tile_log here if checkpoint_gpkg already exists
        if not os.path.exists(checkpoint_gpkg) and os.path.exists(tile_log):
            os.remove(tile_log)

    tile_count = active = skipped = 0
    t0 = time.time()

    for y in y_starts:
        for x in x_starts:
            tile_count += 1
            tid = f"{x}_{y}"

            if tid in processed:
                skipped += 1
                continue

            curr_w = min(BLOCK_SIZE, img_w - x)
            curr_h = min(BLOCK_SIZE, img_h - y)
            window = Window(x, y, curr_w, curr_h)
            img = src.read(window=window)

            if np.mean(img) < 1.0:
                _log(tile_log, tid, processed)
                continue

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            rgb = normalize_image(img)
            del img

            tile_polys = run_sam_on_tile(
                mask_gen, rgb, x, y, img_w, img_h,
                curr_w, curr_h, transform
            )
            all_polys.extend(tile_polys)
            del rgb
            gc.collect()

            _log(tile_log, tid, processed)
            active += 1

            if active % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - t0
                rate = elapsed / active
                remaining = (total_tiles - tile_count) * rate
                print(f"  [{label}] [{tile_count}/{total_tiles}] "
                      f"+{len(tile_polys)} | Total: {len(all_polys)} | "
                      f"ETA: {remaining/60:.1f} min")

            if active % CHECKPOINT_INTERVAL == 0:
                _save_chk(all_polys, native_crs, checkpoint_shp, active, label)

    print(f"\n[{label}] Raw polygons collected: {len(all_polys)}")
    return all_polys, processed


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print("=" * 65)
    print("  FARM BOUNDARY DETECTION – Dual-Pass Fusion Pipeline")
    print("=" * 65)
    print(f"  Device      : {DEVICE.upper()}")
    print(f"  Block Size  : {BLOCK_SIZE}px  Stride: {STRIDE}px")
    print(f"  Pass 1 (Quality)  : pts/side={Q_POINTS_PER_SIDE}, "
          f"IoU={Q_PRED_IOU_THRESH}, batch={Q_POINTS_PER_BATCH}")
    print(f"  Pass 2 (Quantity) : pts/side={P_POINTS_PER_SIDE}, "
          f"IoU={P_PRED_IOU_THRESH}, batch={P_POINTS_PER_BATCH}")
    print(f"  Area filter : {MIN_FARM_AREA_SQM}–{MAX_FARM_AREA_SQM} sqm")
    print(f"  Gap fill    : {FILL_GAPS}")
    print("=" * 65)

    print("\n[MODEL] Loading SAM ViT-B...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    print("[MODEL] SAM loaded.\n")

    chk_quality  = OUTPUT_FILE.replace(".shp", "_chk_quality.shp")
    chk_quantity = OUTPUT_FILE.replace(".shp", "_chk_quantity.shp")
    tile_log_q   = OUTPUT_FILE.replace(".shp", "_tiles_quality.txt")
    tile_log_p   = OUTPUT_FILE.replace(".shp", "_tiles_quantity.txt")

    with rasterio.open(IMAGE_PATH) as src:
        transform  = src.transform
        native_crs = src.crs
        img_w, img_h = src.width, src.height
        scene_bounds = src.bounds

        if native_crs.is_geographic:
            from pyproj import CRS
            cx = (src.bounds.left + src.bounds.right) / 2
            cy = (src.bounds.bottom + src.bounds.top) / 2
            zone = int((cx + 180) / 6) + 1
            epsg = 32600 + zone if cy > 0 else 32700 + zone
            utm_crs = CRS.from_epsg(epsg)
            print(f"[CRS] Geographic → UTM EPSG:{epsg}")
        else:
            utm_crs = native_crs

        print(f"[IMAGE] {img_w} × {img_h} pixels")
        y_starts = list(range(0, img_h, STRIDE))
        x_starts = list(range(0, img_w, STRIDE))
        total_tiles = len(y_starts) * len(x_starts)
        print(f"[TILES] {len(x_starts)} × {len(y_starts)} = {total_tiles} tiles\n")

        # PASS 1 – QUALITY
        print("=" * 65)
        print("  PASS 1: QUALITY  (high IoU, precise boundaries)")
        print("=" * 65)
        q_gen = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=Q_POINTS_PER_SIDE,
            points_per_batch=Q_POINTS_PER_BATCH,
            pred_iou_thresh=Q_PRED_IOU_THRESH,
            stability_score_thresh=Q_STABILITY_THRESH,
            crop_n_layers=Q_CROP_N_LAYERS,
            crop_n_points_downscale_factor=Q_CROP_DOWNSCALE,
            min_mask_region_area=Q_MIN_MASK_AREA,
        )
        quality_raw, processed_q = _run_pass(
            q_gen, src, transform, img_w, img_h,
            x_starts, y_starts, total_tiles,
            chk_quality, tile_log_q, label="Quality"
        )
        del q_gen
        torch.cuda.empty_cache()
        gc.collect()

        # PASS 2 – QUANTITY
        print("\n" + "=" * 65)
        print("  PASS 2: QUANTITY  (low IoU, maximum farm detection)")
        print("=" * 65)
        p_gen = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=P_POINTS_PER_SIDE,
            points_per_batch=P_POINTS_PER_BATCH,
            pred_iou_thresh=P_PRED_IOU_THRESH,
            stability_score_thresh=P_STABILITY_THRESH,
            crop_n_layers=P_CROP_N_LAYERS,
            crop_n_points_downscale_factor=P_CROP_DOWNSCALE,
            min_mask_region_area=P_MIN_MASK_AREA,
        )
        quantity_raw, processed_p = _run_pass(
            p_gen, src, transform, img_w, img_h,
            x_starts, y_starts, total_tiles,
            chk_quantity, tile_log_p, label="Quantity"
        )
        del p_gen, sam
        torch.cuda.empty_cache()
        gc.collect()

    # POST-PROCESSING
    print("\n" + "=" * 65)
    print("  POST-PROCESSING")
    print("=" * 65)

    print("\n[Step 1/6] Area filtering – Quality pass...")
    gdf_q = area_filter(quality_raw, utm_crs, native_crs, "Quality")

    print("\n[Step 2/6] Area filtering – Quantity pass...")
    gdf_p = area_filter(quantity_raw, utm_crs, native_crs, "Quantity")

    print("\n[Step 3/6] Fusing passes...")
    fused_polys = fuse_passes(
        list(gdf_q.geometry), list(gdf_p.geometry),
        min_area=MIN_FARM_AREA_SQM
    )
    del gdf_q, gdf_p, quality_raw, quantity_raw
    gc.collect()

    print(f"\n[Step 4/6] Resolving remaining overlaps ({len(fused_polys)} polys)...")
    no_overlap = remove_overlaps_priority(fused_polys, MIN_FARM_AREA_SQM)
    print(f"  After overlap removal: {len(no_overlap)} polygons")
    del fused_polys
    gc.collect()

    if FILL_GAPS:
        print(f"\n[Step 5/6] Filling gaps...")
        no_overlap = fill_gaps_voronoi(
            no_overlap,
            scene_bounds=(scene_bounds.left, scene_bounds.bottom,
                          scene_bounds.right, scene_bounds.top),
            min_gap_area=GAP_MIN_AREA
        )
    else:
        print("\n[Step 5/6] Gap filling skipped.")

    print(f"\n[Step 6/6] Smoothing {len(no_overlap)} polygon boundaries...")
    smoothed = []
    failed = 0
    for p in no_overlap:
        result = smooth_polygon(p)
        if result is None:
            failed += 1
            continue
        if isinstance(result, list):
            smoothed.extend(result)
        else:
            smoothed.extend(extract_polygons(result, MIN_FARM_AREA_SQM))
    print(f"  Smoothed: {len(smoothed)} polygons ({failed} failed → discarded)")

    final = []
    for p in smoothed:
        if not p.is_valid:
            p = make_valid(p)
        final.extend(extract_polygons(p, MIN_FARM_AREA_SQM))
    print(f"  Final validated: {len(final)} clean farm polygons")

    # SAVE
    out_path = OUTPUT_FILE.replace(".shp", ".gpkg")
    print(f"\n[SAVE] Writing {len(final)} polygons to:\n  {out_path}")
    gdf_out = gpd.GeoDataFrame(geometry=final, crs=utm_crs)
    gdf_out['area_sqm'] = gdf_out.geometry.area
    gdf_out['area_ha']  = gdf_out['area_sqm'] / 10000.0
    gdf_out = gdf_out.to_crs(native_crs)
    gdf_out.to_file(out_path, driver="GPKG", engine=_SAVE_ENGINE)

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*65}")
    print(f"  ✅  DONE!")
    print(f"  Output   : {out_path}")
    print(f"  Polygons : {len(final)}")
    print(f"  Runtime  : {elapsed:.1f} min ({elapsed/60:.2f} hours)")
    print(f"{'='*65}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Run the script again to resume from checkpoint.")
    except Exception as e:
        import traceback
        err_file = OUTPUT_FILE.replace(".shp", "_error.txt")
        with open(err_file, "w") as f:
            f.write(traceback.format_exc())
        print(f"\n[FATAL] {e}")
        print(f"Full traceback → {err_file}")
        raise