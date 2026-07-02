# -*- coding: utf-8 -*-
"""
Farm Boundary Detection Stable Edition (V512)
==============================================
Optimized for:
- 4GB VRAM (SAM vit_b)
- Tile Size: 512 (Native Window)
- Batch Size: 64 (Safe for Memory)
- SAM Internal Chunking (crop_n_layers=1)
"""

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
import os

#  Memory Safety Config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print(" Error: segment-anything not installed. Please install via: pip install segment-anything")
    exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_PATH = r"D:\BISAG\GUJ_235182318471_3_JAN_2021_C2EM.tif"
SAM_CHECKPOINT = r"D:\BISAG\sam_vit_b_01ec64.pth"
OUTPUT_FILE = r"D:\BISAG\farm_pro_v512.shp"

#  SAM SETTINGS
SAM_MODEL_TYPE = "vit_b"
POINTS_PER_SIDE = 24         # Balanced density (576 points per tile)
POINTS_PER_BATCH = 64        #  Safe for 4GB VRAM. Prevents OOM crashes.
CROP_N_LAYERS = 1            # Let SAM perform internal sub-cropping
PRED_IOU_THRESH = 0.88       # High confidence required
STABILITY_THRESH = 0.92      # Sharp mask required
MIN_MASK_AREA = 100          # Discard tiny artifacts

#  TILING (BLOCKS)
BLOCK_SIZE = 768             # Optimized for ~5 hour finish
BLOCK_OVERLAP = 128          # Prevent artifacts at seams

#  POST-PROCESSING
SMOOTH_TOLERANCE = 0.8       # simplification in meters
GAP_CLOSE_DIST = 1.2         # Gap filling distance
MIN_FARM_AREA = 250          # Area in square meters
MAX_FARM_AREA = 50000        # Filter out background universal masks


# ============================================================
# UTILITIES
# ============================================================

def normalize_image(img_bands):
    """Normalize multi-band image to 8-bit RGB for SAM."""
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
    """Smooths a list of coordinates using Chaikin's corner-cutting."""
    coords = list(coords)
    for _ in range(iterations):
        if len(coords) < 3: return coords
        new_coords = []
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i+1]
            new_coords.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            new_coords.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        new_coords.append(new_coords[0]) 
        coords = new_coords
    return coords


def process_polygons(polygons, utm_crs):
    """Clean: Overlap removal, smoothing, gap closing."""
    if not polygons: return []
    
    print(f" Cleaning {len(polygons)} polygons...")
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=utm_crs)
    gdf = gdf[gdf.geometry.area >= MIN_FARM_AREA]
    gdf = gdf[gdf.geometry.area <= MAX_FARM_AREA]
    
    # Sort by area (smallest first)
    gdf['area_temp'] = gdf.geometry.area
    gdf = gdf.sort_values(by='area_temp', ascending=True).reset_index(drop=True)
    gdf = gdf.drop(columns=['area_temp'])
    polys = list(gdf.geometry)
    
    tree = STRtree(polys)
    result = [None] * len(polys)
    
    print("    Solving overlaps and gaps...")
    for i in range(len(polys)):
        poly = polys[i]
        
        # Expand slightly to touch neighbors (close gaps)
        poly = poly.buffer(GAP_CLOSE_DIST, join_style=2) 
        
        # Subtract all already-placed neighbors to remove overlaps
        neighbors_idx = tree.query(poly)
        for nid in neighbors_idx:
            if nid != i and result[nid] is not None:
                try:
                    poly = poly.difference(result[nid])
                except:
                    poly = make_valid(poly)
                    poly = poly.difference(result[nid])
                    
        if not poly.is_empty and poly.area > MIN_FARM_AREA:
            # Shrink back slightly
            poly = poly.buffer(-GAP_CLOSE_DIST * 0.5) 
            if not poly.is_valid: poly = make_valid(poly)
            result[i] = poly

    final_polys = []
    for r in result:
        if r is not None:
            if r.geom_type == 'Polygon': final_polys.append(r)
            elif r.geom_type == 'MultiPolygon': final_polys.extend(r.geoms)
            
    print("    Final smoothing pass...")
    smoothed = []
    for p in final_polys:
        s = p.simplify(SMOOTH_TOLERANCE, preserve_topology=True)
        try:
            ext = chaikin_smooth(s.exterior.coords)
            ints = [chaikin_smooth(h.coords) for h in s.interiors]
            smoothed.append(Polygon(ext, ints))
        except:
            smoothed.append(s)
            
    return smoothed


# ============================================================
# MAIN
# ============================================================

def main():
    start_all = time.time()
    print(f" Initializing SAM {SAM_MODEL_TYPE} on {DEVICE}...")
    
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
    
    # --- RESUME LOGIC ---
    checkpoint_file = OUTPUT_FILE.replace(".shp", "_checkpoint.shp")
    tile_log_file = OUTPUT_FILE.replace(".shp", "_processed.txt")
    processed_tiles = set()

    if os.path.exists(checkpoint_file):
        print(f" [CHECKPOINT] Found existing checkpoint: {checkpoint_file}")
        try:
            chk_gdf = gpd.read_file(checkpoint_file)
            all_polygons = list(chk_gdf.geometry)
            print(f" [CHECKPOINT] Loaded {len(all_polygons)} polygons to resume work.")
        except Exception as e:
            print(f" [WARNING] Could not load checkpoint: {e}. Starting fresh.")

    if os.path.exists(tile_log_file):
        try:
            with open(tile_log_file, "r") as f:
                processed_tiles = set(line.strip() for line in f if line.strip())
            print(f" [CHECKPOINT] Skipping {len(processed_tiles)} already processed tiles.")
        except Exception as e:
            print(f" [WARNING] Could not read tile log: {e}")
    # --------------------
    
    with rasterio.open(IMAGE_PATH) as src:
        transform = src.transform
        crs = src.crs
        
        # Geographic vs Projected correction
        if crs.is_geographic:
            from pyproj import CRS
            left, bottom, right, top = src.bounds
            center_x, center_y = (left + right) / 2, (bottom + top) / 2
            utm_zone = int((center_x + 180) / 6) + 1
            utm_epsg = 32600 + utm_zone if center_y > 0 else 32700 + utm_zone
            utm_crs = CRS.from_epsg(utm_epsg)
            print(f"   Image is Geographic. Using UTM Projection EPSG:{utm_epsg} for calculations.")
        else:
            utm_crs = crs

        w, h = src.width, src.height
        print(f" Image Size: {w}x{h} pixels")
        
        y_steps = range(0, h, BLOCK_SIZE - BLOCK_OVERLAP)
        x_steps = range(0, w, BLOCK_SIZE - BLOCK_OVERLAP)
        total_blocks = len(y_steps) * len(x_steps)
        count = 0
        
        for y in y_steps:
            for x in x_steps:
                count += 1
                
                # --- SKIP LOGIC ---
                tile_id = f"{x}_{y}"
                if tile_id in processed_tiles:
                    continue
                # ------------------

                curr_w = min(BLOCK_SIZE, w - x)
                curr_h = min(BLOCK_SIZE, h - y)
                
                window = Window(x, y, curr_w, curr_h)
                img = src.read(window=window)
                if np.mean(img) < 1: continue
                
                print(f"   Block {count}/{total_blocks} at ({x},{y})...", end="\r")
                
                # CUDA Memory Safety
                if DEVICE == "cuda": torch.cuda.empty_cache()
                
                rgb = normalize_image(img)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        masks = mask_generator.generate(rgb)
                
                for m in masks:
                    seg = m['segmentation'].astype(np.uint8)
                    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) < 4: continue
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
                        except: continue
                
                del masks, rgb, img
                gc.collect()

                # --- UPDATE LOG ---
                try:
                    with open(tile_log_file, "a") as f:
                        f.write(f"{tile_id}\n")
                    processed_tiles.add(tile_id)
                except: pass
                # ------------------

                # Periodic Checkpoint (Every 100 blocks)
                if count % 100 == 0:
                    try:
                        temp_gdf = gpd.GeoDataFrame(geometry=all_polygons, crs=crs)
                        temp_gdf.to_file(OUTPUT_FILE.replace(".shp", "_checkpoint.shp"))
                    except: pass

    print(f"\n SAM detection complete: {len(all_polygons)} raw polygons found.")
    
    if not all_polygons:
        print(" No polygons detected. Check input file and model.")
        return

    # To UTM for precision
    gdf_raw = gpd.GeoDataFrame(geometry=all_polygons, crs=crs)
    gdf_utm = gdf_raw.to_crs(utm_crs)
    
    # Cleaning
    final_polys = process_polygons(list(gdf_utm.geometry), utm_crs)
    
    # Save
    print(f" Saving to {OUTPUT_FILE}...")
    gdf_final = gpd.GeoDataFrame(geometry=final_polys, crs=utm_crs).to_crs(crs)
    gdf_final['area_sqm'] = gdf_final.to_crs(utm_crs).geometry.area
    gdf_final.to_file(OUTPUT_FILE)
    
    end_all = time.time()
    print(f" Done! Total Time: {(end_all - start_all)/60:.2f} mins")

if __name__ == "__main__":
    main()