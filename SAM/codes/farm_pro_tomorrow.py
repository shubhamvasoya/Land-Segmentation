# -*- coding: utf-8 -*-
"""
Farm Boundary Detection - Tomorrow Edition
===========================================
Grid-free architecture using Core Region Clipping:
- 1024px tiles with 256px overlap
- Only keeps polygons whose centroid is in the tile's inner core
- Eliminates grid artifacts and duplicate detections
- Optimized for ~1 hour on 4GB VRAM
"""

import numpy as np
import torch
import rasterio
import cv2
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.strtree import STRtree
from rasterio.windows import Window
import gc
import time
import warnings
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("[ERROR] segment-anything not installed.")
    exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_PATH = r"D:\BISAG\GUJ_235182318471_3_JAN_2021_C2EM.tif"
SAM_CHECKPOINT = r"D:\BISAG\sam_vit_b_01ec64.pth"
OUTPUT_FILE = r"D:\BISAG\farm_pro_tomorrow.shp"

SAM_MODEL_TYPE = "vit_b"
POINTS_PER_SIDE = 32
POINTS_PER_BATCH = 64
CROP_N_LAYERS = 0            # We handle tiling; SAM skips internal cropping. 2x speed.
PRED_IOU_THRESH = 0.86
STABILITY_THRESH = 0.90
MIN_MASK_AREA = 100

BLOCK_SIZE = 1024
BLOCK_OVERLAP = 256          # 128px context on each side of core

SMOOTH_TOLERANCE = 1.0
MIN_FARM_AREA = 500          # sqm - removes noise
MAX_FARM_AREA = 50000        # sqm - removes background masks
MERGE_DISTANCE = 1.0         # meters - heals split farms at tile boundaries


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


def chaikin_smooth(coords, iterations=2):
    coords = list(coords)
    for _ in range(iterations):
        if len(coords) < 3:
            return coords
        new_coords = []
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i + 1]
            new_coords.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            new_coords.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        new_coords.append(new_coords[0])
        coords = new_coords
    return coords


def get_core_bounds(tile_x, tile_y, tile_w, tile_h, img_w, img_h):
    """
    Core region = the non-overlapping center of each tile.
    Edge tiles extend their core to the image boundary.
    Adjacent cores are perfectly flush - no gaps, no overlaps.
    """
    half_ov = BLOCK_OVERLAP // 2
    core_x_min = tile_x if tile_x == 0 else tile_x + half_ov
    core_y_min = tile_y if tile_y == 0 else tile_y + half_ov
    core_x_max = (tile_x + tile_w) if (tile_x + BLOCK_SIZE >= img_w) else (tile_x + tile_w - half_ov)
    core_y_max = (tile_y + tile_h) if (tile_y + BLOCK_SIZE >= img_h) else (tile_y + tile_h - half_ov)
    return core_x_min, core_y_min, core_x_max, core_y_max


def is_in_core(cx, cy, bounds):
    return bounds[0] <= cx < bounds[2] and bounds[1] <= cy < bounds[3]


def batched_union(geoms, batch_size=2000):
    """Perform unary_union in batches for speed on large sets."""
    if len(geoms) <= batch_size:
        return unary_union(geoms)
    results = []
    for i in range(0, len(geoms), batch_size):
        results.append(unary_union(geoms[i:i + batch_size]))
    return unary_union(results)


def extract_polygons(geom, min_area=0):
    """Extract all Polygon objects from any geometry type."""
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


def process_polygons(polygons, utm_crs):
    if not polygons:
        return []

    print(f"  [POST] Starting with {len(polygons)} raw polygons...")
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=utm_crs)

    # Step 1: Area filter
    gdf = gdf[gdf.geometry.area >= MIN_FARM_AREA]
    gdf = gdf[gdf.geometry.area <= MAX_FARM_AREA]
    print(f"  [POST] After area filter: {len(gdf)} polygons")

    if len(gdf) == 0:
        return []

    # Step 2: Merge split farms (buffer-dissolve-unbuffer)
    print("  [POST] Merging split farms at tile boundaries...")
    buffered = list(gdf.geometry.buffer(MERGE_DISTANCE))
    merged = batched_union(buffered)
    pieces = extract_polygons(merged)

    unbuffered = []
    for p in pieces:
        shrunk = p.buffer(-MERGE_DISTANCE)
        unbuffered.extend(extract_polygons(shrunk, MIN_FARM_AREA))
    print(f"  [POST] After merge: {len(unbuffered)} polygons")

    # Step 3: Overlap removal (smallest-first priority)
    print("  [POST] Removing overlaps...")
    polys = sorted(unbuffered, key=lambda p: p.area)
    tree = STRtree(polys)
    result = [None] * len(polys)

    for i in range(len(polys)):
        poly = polys[i]
        neighbors_idx = tree.query(poly)
        for nid in neighbors_idx:
            if nid != i and result[nid] is not None:
                try:
                    poly = poly.difference(result[nid])
                except Exception:
                    poly = make_valid(poly)
                    try:
                        poly = poly.difference(result[nid])
                    except Exception:
                        continue
        if not poly.is_empty and poly.area > MIN_FARM_AREA:
            if not poly.is_valid:
                poly = make_valid(poly)
            result[i] = poly

    final_polys = []
    for r in result:
        final_polys.extend(extract_polygons(r, MIN_FARM_AREA))
    print(f"  [POST] After overlap removal: {len(final_polys)} polygons")

    # Step 4: Smoothing
    print("  [POST] Smoothing boundaries...")
    smoothed = []
    for p in final_polys:
        s = p.simplify(SMOOTH_TOLERANCE, preserve_topology=True)
        try:
            ext = chaikin_smooth(s.exterior.coords)
            ints = [chaikin_smooth(h.coords) for h in s.interiors]
            rp = Polygon(ext, ints)
            if rp.is_valid and rp.area > MIN_FARM_AREA:
                smoothed.append(rp)
            else:
                smoothed.append(s)
        except Exception:
            smoothed.append(s)

    print(f"  [POST] Final count: {len(smoothed)} polygons")
    return smoothed


# ============================================================
# MAIN
# ============================================================

def main():
    start_all = time.time()
    print(f"[MODEL] Initializing SAM {SAM_MODEL_TYPE} on {DEVICE}...")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=POINTS_PER_BATCH,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        crop_n_layers=CROP_N_LAYERS,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=MIN_MASK_AREA,
    )

    all_polygons = []
    stride = BLOCK_SIZE - BLOCK_OVERLAP  # 768

    # --- RESUME LOGIC ---
    checkpoint_file = OUTPUT_FILE.replace(".shp", "_checkpoint.shp")
    tile_log_file = OUTPUT_FILE.replace(".shp", "_processed.txt")
    processed_tiles = set()

    if os.path.exists(checkpoint_file):
        print(f"[RESUME] Found checkpoint: {checkpoint_file}")
        try:
            chk_gdf = gpd.read_file(checkpoint_file)
            all_polygons = list(chk_gdf.geometry)
            print(f"[RESUME] Loaded {len(all_polygons)} polygons.")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}")

    if os.path.exists(tile_log_file):
        try:
            with open(tile_log_file, "r") as f:
                processed_tiles = set(line.strip() for line in f if line.strip())
            print(f"[RESUME] Skipping {len(processed_tiles)} already processed tiles.")
        except Exception as e:
            print(f"[WARNING] Could not read tile log: {e}")

    with rasterio.open(IMAGE_PATH) as src:
        transform = src.transform
        crs = src.crs

        if crs.is_geographic:
            from pyproj import CRS
            left, bottom, right, top = src.bounds
            cx, cy = (left + right) / 2, (bottom + top) / 2
            utm_zone = int((cx + 180) / 6) + 1
            utm_epsg = 32600 + utm_zone if cy > 0 else 32700 + utm_zone
            utm_crs = CRS.from_epsg(utm_epsg)
            print(f"[IMAGE] Geographic CRS. Using UTM EPSG:{utm_epsg}")
        else:
            utm_crs = crs

        w, h = src.width, src.height
        print(f"[IMAGE] Size: {w}x{h} pixels")

        y_steps = list(range(0, h, stride))
        x_steps = list(range(0, w, stride))
        total_blocks = len(y_steps) * len(x_steps)
        count = 0
        skipped = 0
        tile_start = time.time()

        for y in y_steps:
            for x in x_steps:
                count += 1
                tile_id = f"{x}_{y}"
                if tile_id in processed_tiles:
                    skipped += 1
                    continue

                curr_w = min(BLOCK_SIZE, w - x)
                curr_h = min(BLOCK_SIZE, h - y)

                window = Window(x, y, curr_w, curr_h)
                img = src.read(window=window)
                if np.mean(img) < 1:
                    try:
                        with open(tile_log_file, "a") as f:
                            f.write(f"{tile_id}\n")
                        processed_tiles.add(tile_id)
                    except Exception:
                        pass
                    continue

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                rgb = normalize_image(img)
                core_bounds = get_core_bounds(x, y, curr_w, curr_h, w, h)

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        masks = mask_generator.generate(rgb)

                tile_poly_count = 0
                for m in masks:
                    seg = m['segmentation'].astype(np.uint8)
                    if np.sum(seg) > curr_w * curr_h * 0.80:
                        continue

                    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) < 4:
                            continue
                        moments = cv2.moments(contour)
                        if moments['m00'] == 0:
                            continue
                        cx_local = moments['m10'] / moments['m00']
                        cy_local = moments['m01'] / moments['m00']
                        cx_global = cx_local + x
                        cy_global = cy_local + y

                        if not is_in_core(cx_global, cy_global, core_bounds):
                            continue

                        coords = []
                        for pt in contour:
                            px = pt[0][0] + x
                            py = pt[0][1] + y
                            wx, wy = rasterio.transform.xy(transform, py, px)
                            coords.append((wx, wy))
                        try:
                            poly = Polygon(coords)
                            if poly.is_valid and poly.area > 0:
                                all_polygons.append(poly)
                                tile_poly_count += 1
                        except Exception:
                            continue

                del masks, rgb, img
                gc.collect()

                try:
                    with open(tile_log_file, "a") as f:
                        f.write(f"{tile_id}\n")
                    processed_tiles.add(tile_id)
                except Exception:
                    pass

                active_count = count - skipped
                if active_count > 0 and active_count % 10 == 0:
                    elapsed = time.time() - tile_start
                    remaining = total_blocks - count
                    rate = elapsed / active_count
                    eta_min = (remaining * rate) / 60
                    print(f"  [PROGRESS] Block {count}/{total_blocks} | "
                          f"+{tile_poly_count} polys | "
                          f"Total: {len(all_polygons)} | "
                          f"ETA: {eta_min:.1f} min")

                if active_count > 0 and active_count % 50 == 0:
                    try:
                        temp_gdf = gpd.GeoDataFrame(geometry=all_polygons, crs=crs)
                        temp_gdf.to_file(checkpoint_file)
                        print(f"  [SAVED] Checkpoint: {len(all_polygons)} polygons")
                    except Exception:
                        pass

    print(f"\n[SAM] Detection complete: {len(all_polygons)} raw polygons.")

    if not all_polygons:
        print("[ERROR] No polygons detected.")
        return

    print("[POST] Converting to UTM for post-processing...")
    gdf_raw = gpd.GeoDataFrame(geometry=all_polygons, crs=crs)
    gdf_utm = gdf_raw.to_crs(utm_crs)

    final_polys = process_polygons(list(gdf_utm.geometry), utm_crs)

    print(f"[SAVE] Writing {len(final_polys)} polygons to {OUTPUT_FILE}...")
    gdf_final = gpd.GeoDataFrame(geometry=final_polys, crs=utm_crs).to_crs(crs)
    gdf_final['area_sqm'] = gdf_final.to_crs(utm_crs).geometry.area
    gdf_final.to_file(OUTPUT_FILE)

    end_all = time.time()
    print(f"[DONE] Total Time: {(end_all - start_all)/60:.1f} minutes")
    print(f"[DONE] Output: {OUTPUT_FILE}")
    print(f"[DONE] Polygons: {len(final_polys)}")


if __name__ == "__main__":
    main()