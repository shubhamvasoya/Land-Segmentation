# -*- coding: utf-8 -*-
"""
Farm Boundary Detection - Final Pipeline
=========================================
Requirements satisfied:
  1. Detect as many farms as possible        -> points_per_side=32, lower thresholds
  2. Solve overlapping polygons              -> STRtree-based overlap removal
  3. One polygon per farm (no merging)       -> No buffer-dissolve merge; keep separate
  4. Clean, non-zigzag boundaries            -> Douglas-Peucker + Chaikin smoothing
  5. SAM automatic chunking                  -> crop_n_layers=1
  6. Batch size 512                          -> BLOCK_SIZE=512

Architecture:
  - 512px tiles, stride=384 (128px overlap)
  - SAM crop_n_layers=1 for internal sub-cropping
  - Core-region clipping to avoid duplicates
  - Post-processing: area filter -> overlap removal -> smoothing
  - Checkpointing every 50 processed tiles for crash recovery
  - Estimated: ~2-4 hours total on 4GB VRAM
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
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("[FATAL] segment-anything not installed.")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_PATH     = r"D:\BISAG\GUJ_235182318471_3_JAN_2021_C2EM.tif"
SAM_CHECKPOINT = r"D:\BISAG\sam_vit_b_01ec64.pth"
OUTPUT_FILE    = r"D:\BISAG\farm_final.shp"

# --- SAM Parameters ---
SAM_MODEL_TYPE = "vit_b"
POINTS_PER_SIDE = 24           # 24 avoids hard OOMs, relies on threshold for more farms
POINTS_PER_BATCH = 512         # Keep the batch size 512 as requested
CROP_N_LAYERS = 1              # SAM auto-chunks each tile into sub-crops
CROP_N_POINTS_DOWNSCALE = 2    # Fewer points in sub-crops (standard)
PRED_IOU_THRESH = 0.82         # Lowered to 0.82 for maximum farm detection
STABILITY_THRESH = 0.86        # Lowered to 0.86 for maximum farm detection
MIN_MASK_AREA = 80             # Pixel-level minimum inside SAM

# --- Tiling ---
BLOCK_SIZE = 512               # User requirement
BLOCK_OVERLAP = 64             # Minimal context buffer
STRIDE = BLOCK_SIZE - BLOCK_OVERLAP

# --- Post-processing ---
SMOOTH_TOLERANCE = 1.2         # Douglas-Peucker simplification (meters)
CHAIKIN_ITERATIONS = 3         # More iterations = smoother curves
MIN_FARM_AREA = 200            # sqm - filter noise
MAX_FARM_AREA = 60000          # sqm - filter background blobs
MAX_MASK_COVERAGE = 0.85       # Skip masks covering >85% of tile (background)

# --- Checkpointing ---
CHECKPOINT_INTERVAL = 50       # Save every N processed tiles
PROGRESS_INTERVAL = 10         # Print progress every N processed tiles


# ============================================================
# UTILITIES
# ============================================================

def normalize_image(img_bands):
    """Normalize satellite bands to 8-bit RGB with percentile stretch."""
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
    return np.transpose(out, (1, 2, 0))  # HWC


def chaikin_smooth(coords, iterations=CHAIKIN_ITERATIONS):
    """Chaikin's corner-cutting algorithm for smooth polygon boundaries."""
    coords = list(coords)
    for _ in range(iterations):
        if len(coords) < 3:
            return coords
        new_coords = []
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i + 1]
            new_coords.append((0.75 * p0[0] + 0.25 * p1[0],
                               0.75 * p0[1] + 0.25 * p1[1]))
            new_coords.append((0.25 * p0[0] + 0.75 * p1[0],
                               0.25 * p0[1] + 0.75 * p1[1]))
        new_coords.append(new_coords[0])  # close ring
        coords = new_coords
    return coords


def get_core_bounds(tile_x, tile_y, tile_w, tile_h, img_w, img_h):
    """
    Core region = the non-overlapping center of each tile.
    Edge tiles extend their core to the image boundary.
    This ensures adjacent cores are perfectly flush: no gaps, no overlaps.
    """
    half_ov = BLOCK_OVERLAP // 2  # 64px
    # Left / top edge: core starts at tile origin
    core_x_min = tile_x if tile_x == 0 else tile_x + half_ov
    core_y_min = tile_y if tile_y == 0 else tile_y + half_ov
    # Right / bottom edge: core extends to image boundary
    core_x_max = (tile_x + tile_w) if (tile_x + BLOCK_SIZE >= img_w) else (tile_x + tile_w - half_ov)
    core_y_max = (tile_y + tile_h) if (tile_y + BLOCK_SIZE >= img_h) else (tile_y + tile_h - half_ov)
    return core_x_min, core_y_min, core_x_max, core_y_max


def centroid_in_core(cx, cy, core_bounds):
    """Check if a polygon's centroid falls within the core region."""
    return (core_bounds[0] <= cx < core_bounds[2] and
            core_bounds[1] <= cy < core_bounds[3])


def extract_individual_polygons(geom, min_area=0):
    """Recursively extract all Polygon objects from any geometry type."""
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
            out.extend(extract_individual_polygons(g, min_area))
    return out


# ============================================================
# POST-PROCESSING
# ============================================================

def post_process(polygons, utm_crs, native_crs):
    """
    Full post-processing pipeline:
      1. Area filtering (remove noise + background)
      2. Overlap removal (largest-wins priority)
      3. Boundary smoothing (Douglas-Peucker + Chaikin)
      4. Final validation
    """
    if not polygons:
        return []

    print(f"\n{'='*60}")
    print(f"  POST-PROCESSING: {len(polygons)} raw polygons")
    print(f"{'='*60}")

    # --- Convert to UTM for metric area calculations ---
    print("[1/4] Converting to UTM and filtering by area...")
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=native_crs)
    gdf_utm = gdf.to_crs(utm_crs)

    areas = gdf_utm.geometry.area
    mask = (areas >= MIN_FARM_AREA) & (areas <= MAX_FARM_AREA)
    gdf_utm = gdf_utm[mask].reset_index(drop=True)
    print(f"  -> After area filter: {len(gdf_utm)} polygons "
          f"(removed {(~mask).sum()} outside {MIN_FARM_AREA}-{MAX_FARM_AREA} sqm)")

    if len(gdf_utm) == 0:
        print("  [WARNING] No polygons survived area filter!")
        return []

    # --- Overlap Removal (larger polygon wins) ---
    print("[2/4] Removing overlapping polygons...")
    polys = list(gdf_utm.geometry)

    # Sort by area DESCENDING - larger farms get priority, smaller fragments are clipped
    polys.sort(key=lambda p: p.area, reverse=True)

    placed = []       # Final non-overlapping polygons
    placed_tree = None

    for i, poly in enumerate(polys):
        if not poly.is_valid:
            poly = make_valid(poly)
            extracts = extract_individual_polygons(poly, MIN_FARM_AREA)
            if not extracts:
                continue
            poly = max(extracts, key=lambda p: p.area)

        if placed:
            # Rebuild spatial index periodically for performance
            if placed_tree is None or i % 500 == 0:
                placed_tree = STRtree(placed)

            candidates = placed_tree.query(poly)
            for idx in candidates:
                neighbor = placed[idx]
                if poly.intersects(neighbor):
                    try:
                        poly = poly.difference(neighbor)
                    except Exception:
                        poly = make_valid(poly)
                        try:
                            poly = poly.difference(make_valid(neighbor))
                        except Exception:
                            continue

            # After clipping, extract surviving pieces
            survivors = extract_individual_polygons(poly, MIN_FARM_AREA)
            for s in survivors:
                placed.append(s)
        else:
            placed.append(poly)

        if (i + 1) % 5000 == 0:
            print(f"    ... processed {i+1}/{len(polys)} overlap checks, "
                  f"{len(placed)} placed so far")

    print(f"  -> After overlap removal: {len(placed)} non-overlapping polygons")

    # --- Smoothing ---
    print("[3/4] Smoothing boundaries (Douglas-Peucker + Chaikin)...")
    smoothed = []
    failed = 0
    for p in placed:
        try:
            # Step 1: Simplify to remove pixel staircase
            s = p.simplify(SMOOTH_TOLERANCE, preserve_topology=True)
            if s.is_empty or s.area < MIN_FARM_AREA:
                continue

            # Step 2: Chaikin corner-cutting for smooth curves
            if s.geom_type == 'Polygon':
                ext = chaikin_smooth(s.exterior.coords)
                ints = [chaikin_smooth(h.coords) for h in s.interiors]
                result = Polygon(ext, ints)
                if result.is_valid and result.area > MIN_FARM_AREA:
                    smoothed.append(result)
                else:
                    # Fallback: use simplified but not Chaikin'd
                    if s.is_valid and s.area > MIN_FARM_AREA:
                        smoothed.append(s)
            elif s.geom_type == 'MultiPolygon':
                for part in s.geoms:
                    ext = chaikin_smooth(part.exterior.coords)
                    ints = [chaikin_smooth(h.coords) for h in part.interiors]
                    result = Polygon(ext, ints)
                    if result.is_valid and result.area > MIN_FARM_AREA:
                        smoothed.append(result)
        except Exception:
            failed += 1
            if p.is_valid and p.area > MIN_FARM_AREA:
                smoothed.append(p)

    if failed > 0:
        print(f"    ({failed} polygons fell back to unsmoothed)")
    print(f"  -> After smoothing: {len(smoothed)} polygons")

    # --- Final Validation ---
    print("[4/4] Final validation pass...")
    final = []
    for p in smoothed:
        if not p.is_valid:
            p = make_valid(p)
        pieces = extract_individual_polygons(p, MIN_FARM_AREA)
        final.extend(pieces)

    print(f"  -> FINAL OUTPUT: {len(final)} clean, non-overlapping farm polygons")
    return final


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    t_start = time.time()
    print("=" * 60)
    print("  FARM BOUNDARY DETECTION - Final Pipeline")
    print("=" * 60)
    print(f"  Device:       {DEVICE}")
    print(f"  Block Size:   {BLOCK_SIZE}px")
    print(f"  Overlap:      {BLOCK_OVERLAP}px (stride={STRIDE})")
    print(f"  SAM crop_n_layers: {CROP_N_LAYERS} (automatic chunking)")
    print(f"  Points/side:  {POINTS_PER_SIDE}")
    print(f"  Output:       {OUTPUT_FILE}")
    print("=" * 60)

    # ---- Load SAM ----
    print("\n[MODEL] Loading SAM...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=POINTS_PER_BATCH,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        crop_n_layers=CROP_N_LAYERS,
        crop_n_points_downscale_factor=CROP_N_POINTS_DOWNSCALE,
        min_mask_region_area=MIN_MASK_AREA,
    )
    print("[MODEL] SAM ready.\n")

    # ---- Resume Logic ----
    checkpoint_shp = OUTPUT_FILE.replace(".shp", "_checkpoint.shp")
    tile_log = OUTPUT_FILE.replace(".shp", "_tiles.txt")
    all_polygons = []
    processed_tiles = set()

    if os.path.exists(checkpoint_shp) and os.path.exists(tile_log):
        print("[RESUME] Found checkpoint files...")
        try:
            chk = gpd.read_file(checkpoint_shp)
            all_polygons = list(chk.geometry)
            print(f"[RESUME] Loaded {len(all_polygons)} polygons from checkpoint.")
        except Exception as e:
            print(f"[RESUME] Could not load checkpoint polygons: {e}")
            all_polygons = []

        try:
            with open(tile_log, "r") as f:
                processed_tiles = set(l.strip() for l in f if l.strip())
            print(f"[RESUME] Will skip {len(processed_tiles)} already-processed tiles.")
        except Exception as e:
            print(f"[RESUME] Could not load tile log: {e}")
            processed_tiles = set()
    else:
        # Clean start - remove any stale checkpoint files
        for ext in [".cpg", ".dbf", ".prj", ".shp", ".shx"]:
            f = checkpoint_shp.replace(".shp", ext)
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(tile_log):
            os.remove(tile_log)

    # ---- Open Image & Tile ----
    with rasterio.open(IMAGE_PATH) as src:
        transform = src.transform
        native_crs = src.crs
        img_w, img_h = src.width, src.height

        # Determine UTM CRS for area calculations
        if native_crs.is_geographic:
            from pyproj import CRS
            left, bottom, right, top = src.bounds
            cx = (left + right) / 2
            cy = (bottom + top) / 2
            utm_zone = int((cx + 180) / 6) + 1
            utm_epsg = 32600 + utm_zone if cy > 0 else 32700 + utm_zone
            utm_crs = CRS.from_epsg(utm_epsg)
            print(f"[IMAGE] Geographic CRS -> UTM EPSG:{utm_epsg}")
        else:
            utm_crs = native_crs

        print(f"[IMAGE] Size: {img_w} x {img_h} pixels")

        # Build tile grid
        y_starts = list(range(0, img_h, STRIDE))
        x_starts = list(range(0, img_w, STRIDE))
        total_tiles = len(y_starts) * len(x_starts)
        print(f"[TILES] Grid: {len(x_starts)} x {len(y_starts)} = {total_tiles} tiles\n")

        tile_count = 0
        active_count = 0
        skipped_count = 0
        empty_count = 0
        t_tiles_start = time.time()

        for y in y_starts:
            for x in x_starts:
                tile_count += 1
                tile_id = f"{x}_{y}"

                # Skip already-processed
                if tile_id in processed_tiles:
                    skipped_count += 1
                    continue

                # Read tile
                curr_w = min(BLOCK_SIZE, img_w - x)
                curr_h = min(BLOCK_SIZE, img_h - y)
                window = Window(x, y, curr_w, curr_h)
                img = src.read(window=window)

                # Skip empty/black tiles
                if np.mean(img) < 1.0:
                    _log_tile(tile_log, tile_id, processed_tiles)
                    empty_count += 1
                    continue

                # Free VRAM
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                rgb = normalize_image(img)
                core = get_core_bounds(x, y, curr_w, curr_h, img_w, img_h)

                # --- SAM Inference with OOM Cascading ---
                masks = None
                original_batch = mask_generator.points_per_batch
                for b_size in [original_batch, 64, 16, 4]:
                    try:
                        mask_generator.points_per_batch = b_size
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                masks = mask_generator.generate(rgb)
                        break  # Success!
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            del e
                            torch.cuda.empty_cache()
                            gc.collect()
                            if b_size > 4:
                                print(f"  [OOM] Tile {tile_id} batch={b_size} failed. Retrying smaller...")
                            continue
                        raise
                        
                mask_generator.points_per_batch = original_batch
                
                if masks is None:
                    print(f"  [FATAL OOM] Tile {tile_id} failed even at batch=4. Skipping.")
                    _log_tile(tile_log, tile_id, processed_tiles)
                    continue

                # --- Extract Polygons ---
                tile_polys = 0
                for m in masks:
                    seg = m['segmentation'].astype(np.uint8)

                    # Skip background masks (cover too much of the tile)
                    coverage = np.sum(seg) / (curr_w * curr_h)
                    if coverage > MAX_MASK_COVERAGE:
                        continue

                    contours, _ = cv2.findContours(
                        seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for contour in contours:
                        if len(contour) < 4:
                            continue

                        # Centroid check against core region
                        moments = cv2.moments(contour)
                        if moments['m00'] == 0:
                            continue
                        cx_local = moments['m10'] / moments['m00']
                        cy_local = moments['m01'] / moments['m00']
                        cx_global = cx_local + x
                        cy_global = cy_local + y

                        if not centroid_in_core(cx_global, cy_global, core):
                            continue

                        # Convert pixel contour to geo coordinates
                        coords = []
                        for pt in contour:
                            px = pt[0][0] + x
                            py = pt[0][1] + y
                            wx, wy = rasterio.transform.xy(transform, py, px)
                            coords.append((wx, wy))

                        if len(coords) < 4:
                            continue

                        try:
                            poly = Polygon(coords)
                            if not poly.is_valid:
                                poly = make_valid(poly)
                                pieces = extract_individual_polygons(poly)
                                for piece in pieces:
                                    if piece.area > 0:
                                        all_polygons.append(piece)
                                        tile_polys += 1
                            elif poly.area > 0:
                                all_polygons.append(poly)
                                tile_polys += 1
                        except Exception:
                            continue

                # Cleanup
                del masks, rgb, img
                gc.collect()

                # Log processed tile
                _log_tile(tile_log, tile_id, processed_tiles)
                active_count += 1

                # Progress reporting
                if active_count % PROGRESS_INTERVAL == 0:
                    elapsed = time.time() - t_tiles_start
                    remaining_tiles = total_tiles - tile_count
                    rate = elapsed / active_count if active_count > 0 else 1
                    eta_min = (remaining_tiles * rate) / 60

                    print(f"  [{tile_count}/{total_tiles}] "
                          f"+{tile_polys} polys | "
                          f"Total: {len(all_polygons)} | "
                          f"Rate: {rate:.2f}s/tile | "
                          f"ETA: {eta_min:.1f} min")

                # Periodic checkpoint
                if active_count % CHECKPOINT_INTERVAL == 0:
                    _save_checkpoint(all_polygons, native_crs,
                                     checkpoint_shp, active_count)

    # ---- SAM Phase Complete ----
    t_sam_end = time.time()
    sam_minutes = (t_sam_end - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  SAM DETECTION COMPLETE")
    print(f"  Raw polygons: {len(all_polygons)}")
    print(f"  Time:         {sam_minutes:.1f} minutes")
    print(f"  Tiles:        {active_count} processed, "
          f"{skipped_count} skipped, {empty_count} empty")
    print(f"{'='*60}")

    if not all_polygons:
        print("[ERROR] No polygons detected. Exiting.")
        return

    # Free SAM from GPU
    del mask_generator, sam
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Post-Processing ----
    final_polys = post_process(all_polygons, utm_crs, native_crs)

    if not final_polys:
        print("[ERROR] Post-processing produced no polygons!")
        return

    # ---- Save Final Output ----
    print(f"\n[SAVE] Writing {len(final_polys)} polygons to {OUTPUT_FILE}...")
    gdf_final = gpd.GeoDataFrame(geometry=final_polys, crs=utm_crs)

    # Add area column in square meters
    gdf_final['area_sqm'] = gdf_final.geometry.area

    # Convert back to native CRS for the shapefile
    gdf_final = gdf_final.to_crs(native_crs)
    gdf_final.to_file(OUTPUT_FILE)

    t_end = time.time()
    total_min = (t_end - t_start) / 60

    print(f"\n{'='*60}")
    print(f"  ✅ DONE!")
    print(f"  Output:    {OUTPUT_FILE}")
    print(f"  Polygons:  {len(final_polys)}")
    print(f"  Total Time: {total_min:.1f} minutes ({total_min/60:.1f} hours)")
    print(f"{'='*60}")


# ============================================================
# HELPERS
# ============================================================

def _log_tile(tile_log, tile_id, processed_set):
    """Append tile ID to the processed log file."""
    try:
        with open(tile_log, "a") as f:
            f.write(f"{tile_id}\n")
        processed_set.add(tile_id)
    except Exception:
        pass


def _save_checkpoint(polygons, crs, path, count):
    """Save intermediate checkpoint shapefile."""
    try:
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        gdf.to_file(path)
        print(f"  [CHECKPOINT] Saved {len(polygons)} polygons "
              f"(after {count} tiles)")
    except Exception as e:
        print(f"  [CHECKPOINT] Save failed: {e}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] You can resume by running this script again.")
    except Exception as e:
        import traceback
        error_file = OUTPUT_FILE.replace(".shp", "_error.txt")
        with open(error_file, "w") as f:
            f.write(traceback.format_exc())
        print(f"\n[FATAL ERROR] {e}")
        print(f"Full traceback saved to: {error_file}")
        raise
