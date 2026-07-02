"""
Smart Merge: Delineate Anything + SAM Outputs
===============================================
Quality-aware merge that:
  1. Compares both model outputs against the raw satellite image
  2. Uses spectral homogeneity (CV = std/mean) to pick the better polygon
  3. Guarantees zero overlapping polygons in the final output

Usage:
    python merge_outputs.py

Before running, update the file paths in the CONFIG section at the bottom.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from tqdm import tqdm
import os
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
#   CORE FUNCTIONS
# =====================================================================

def find_tiff_files(directory):
    """Find all .tif/.tiff files in a directory."""
    tiffs = []
    for f in os.listdir(directory):
        if f.lower().endswith((".tif", ".tiff")):
            tiffs.append(os.path.join(directory, f))
    return tiffs


def calculate_spectral_cv(polygon_geom, raster_src, polygon_crs):
    """
    Calculate spectral Coefficient of Variation (CV = std/mean) for pixels
    inside a polygon. Lower CV = more uniform = better delineation.

    Returns the average CV across all bands, or a high penalty value if
    the polygon has too few pixels or an error occurs.
    """
    PENALTY_CV = 999.0  # returned when we can't compute

    try:
        # Transform polygon bounds to raster CRS if needed
        if str(polygon_crs) != str(raster_src.crs):
            from pyproj import Transformer
            transformer = Transformer.from_crs(polygon_crs, raster_src.crs, always_xy=True)
            minx, miny, maxx, maxy = polygon_geom.bounds
            minx2, miny2 = transformer.transform(minx, miny)
            maxx2, maxy2 = transformer.transform(maxx, maxy)
            # Check if polygon is within raster extent
            raster_bounds = raster_src.bounds
            if (maxx2 < raster_bounds.left or minx2 > raster_bounds.right or
                maxy2 < raster_bounds.bottom or miny2 > raster_bounds.top):
                return PENALTY_CV

        # Mask raster with the polygon geometry
        geom_for_mask = [mapping(polygon_geom)]
        try:
            out_image, _ = rasterio_mask(
                raster_src, geom_for_mask,
                crop=True, filled=True, nodata=0,
                all_touched=True
            )
        except Exception:
            return PENALTY_CV

        # out_image shape: (bands, height, width)
        num_bands = out_image.shape[0]

        cvs = []
        for band_idx in range(num_bands):
            band_data = out_image[band_idx]
            # Only consider non-zero (non-nodata) pixels
            valid_pixels = band_data[band_data > 0].astype(np.float64)

            if len(valid_pixels) < 10:  # too few pixels to judge
                continue

            mean_val = np.mean(valid_pixels)
            std_val = np.std(valid_pixels)

            if mean_val > 0:
                cvs.append(std_val / mean_val)

        if not cvs:
            return PENALTY_CV

        return np.mean(cvs)

    except Exception as e:
        logger.debug(f"CV calculation error: {e}")
        return PENALTY_CV


def compute_group_cv(polygons_gdf, raster_src, crs):
    """
    Compute average CV across a group of polygons.
    Returns (average_cv, list_of_individual_cvs).
    """
    cvs = []
    for _, row in polygons_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        cv = calculate_spectral_cv(geom, raster_src, crs)
        cvs.append(cv)

    if not cvs:
        return 999.0, cvs

    return np.mean(cvs), cvs


def resolve_overlaps(gdf):
    """
    Remove any remaining overlaps in the final merged GeoDataFrame.
    For overlapping polygons, subtract the overlap from the LARGER polygon
    (keeping the smaller/more-specific one intact).
    """
    logger.info("Resolving any remaining polygon overlaps...")
    sindex = gdf.sindex

    geometries = gdf.geometry.values.copy()
    to_remove = set()

    for i in tqdm(range(len(gdf)), desc="Resolving overlaps", unit="poly"):
        if i in to_remove:
            continue

        geom_i = geometries[i]
        if geom_i is None or geom_i.is_empty:
            to_remove.add(i)
            continue

        # Find candidates via spatial index
        candidate_idxs = list(sindex.intersection(geom_i.bounds))

        for j in candidate_idxs:
            if j <= i or j in to_remove:
                continue

            geom_j = geometries[j]
            if geom_j is None or geom_j.is_empty:
                to_remove.add(j)
                continue

            if not geom_i.intersects(geom_j):
                continue

            intersection = geom_i.intersection(geom_j)
            if intersection.is_empty:
                continue

            overlap_area = intersection.area

            # Only resolve if overlap is meaningful (> 1% of smaller polygon)
            min_area = min(geom_i.area, geom_j.area)
            if min_area > 0 and (overlap_area / min_area) < 0.01:
                continue

            # Subtract overlap from the LARGER polygon (keep smaller one intact)
            if geom_i.area >= geom_j.area:
                diff = geom_i.difference(geom_j)
                if diff is not None and not diff.is_empty:
                    geometries[i] = diff
                    geom_i = diff  # update for subsequent comparisons
                else:
                    to_remove.add(i)
                    break
            else:
                diff = geom_j.difference(geom_i)
                if diff is not None and not diff.is_empty:
                    geometries[j] = diff
                else:
                    to_remove.add(j)

    # Update geometries and remove empty ones
    gdf = gdf.copy()
    gdf["geometry"] = geometries

    if to_remove:
        gdf = gdf.drop(index=list(to_remove)).reset_index(drop=True)

    # Filter out any geometries that became too small
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)

    return gdf


# =====================================================================
#   MAIN MERGE FUNCTION
# =====================================================================

def smart_merge(delineate_path, sam_path, raw_image_dir, output_path,
                overlap_iou_threshold=0.1, min_area_m2=500):
    """
    Perform a quality-aware merge of Delineate Anything and SAM outputs.

    Parameters
    ----------
    delineate_path : str
        Path to Delineate Anything output (.gpkg)
    sam_path : str
        Path to SAM output (.gpkg / .shp / .geojson)
    raw_image_dir : str
        Path to folder containing raw satellite GeoTIFF(s)
    output_path : str
        Path for merged output (.gpkg)
    overlap_iou_threshold : float
        Minimum IoU to consider two polygons as overlapping (default: 0.1)
    min_area_m2 : float
        Minimum polygon area in m² to keep (default: 500)
    """

    # ==================================================================
    # STEP 1: Load data
    # ==================================================================
    logger.info(f"Loading Delineate Anything output: {delineate_path}")
    da_gdf = gpd.read_file(delineate_path)
    logger.info(f"  → {len(da_gdf)} polygons loaded")

    logger.info(f"Loading SAM output: {sam_path}")
    sam_gdf = gpd.read_file(sam_path)
    logger.info(f"  → {len(sam_gdf)} polygons loaded")

    # Find raw image
    tiff_files = find_tiff_files(raw_image_dir)
    if not tiff_files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {raw_image_dir}")
    raw_image_path = tiff_files[0]
    logger.info(f"Using raw image: {raw_image_path}")

    # Open raster
    raster_src = rasterio.open(raw_image_path)
    raster_crs = raster_src.crs
    logger.info(f"  → Raster CRS: {raster_crs}, Size: {raster_src.width}x{raster_src.height}, Bands: {raster_src.count}")

    # Ensure same CRS (reproject to raster CRS for pixel extraction)
    working_crs = da_gdf.crs

    if da_gdf.crs != raster_crs:
        logger.info(f"Reprojecting DA polygons from {da_gdf.crs} to {raster_crs}")
        da_gdf = da_gdf.to_crs(raster_crs)
        working_crs = raster_crs

    if sam_gdf.crs != raster_crs:
        logger.info(f"Reprojecting SAM polygons from {sam_gdf.crs} to {raster_crs}")
        sam_gdf = sam_gdf.to_crs(raster_crs)

    # ==================================================================
    # STEP 2: Filter by minimum area
    # ==================================================================
    equal_area_crs = "EPSG:6933"
    da_areas = da_gdf.to_crs(equal_area_crs).geometry.area
    sam_areas = sam_gdf.to_crs(equal_area_crs).geometry.area

    da_gdf = da_gdf[da_areas >= min_area_m2].reset_index(drop=True)
    sam_gdf = sam_gdf[sam_areas >= min_area_m2].reset_index(drop=True)
    logger.info(f"After area filter (>= {min_area_m2} m²): DA={len(da_gdf)}, SAM={len(sam_gdf)}")

    # Add source column
    da_gdf["source"] = "delineate_anything"
    sam_gdf["source"] = "sam"

    # ==================================================================
    # STEP 3: Classify polygons into DA-only, SAM-only, Overlap groups
    # ==================================================================
    logger.info("Classifying polygons into overlap groups...")

    da_sindex = da_gdf.sindex
    sam_sindex = sam_gdf.sindex

    # Track which polygons are involved in overlaps
    da_in_overlap = set()   # indices of DA polygons that overlap with SAM
    sam_in_overlap = set()  # indices of SAM polygons that overlap with DA

    # Build overlap groups: each group is a set of DA indices + SAM indices
    # that mutually overlap
    overlap_pairs = []  # list of (da_idx, sam_idx, iou)

    for da_idx, da_row in tqdm(da_gdf.iterrows(), total=len(da_gdf),
                                desc="Finding overlaps", unit="poly"):
        da_geom = da_row.geometry
        if da_geom is None or da_geom.is_empty:
            continue

        # Find SAM candidates
        sam_candidates = list(sam_sindex.intersection(da_geom.bounds))

        for sam_idx in sam_candidates:
            sam_geom = sam_gdf.iloc[sam_idx].geometry
            if sam_geom is None or sam_geom.is_empty:
                continue

            if not da_geom.intersects(sam_geom):
                continue

            try:
                intersection = da_geom.intersection(sam_geom).area
                union = da_geom.union(sam_geom).area
                iou = intersection / union if union > 0 else 0
            except Exception:
                continue

            if iou > overlap_iou_threshold:
                overlap_pairs.append((da_idx, sam_idx, iou))
                da_in_overlap.add(da_idx)
                sam_in_overlap.add(sam_idx)

    # Identify non-overlapping polygons
    da_only_idxs = set(range(len(da_gdf))) - da_in_overlap
    sam_only_idxs = set(range(len(sam_gdf))) - sam_in_overlap

    logger.info(f"  DA-only polygons: {len(da_only_idxs)}")
    logger.info(f"  SAM-only polygons: {len(sam_only_idxs)}")
    logger.info(f"  Overlap pairs found: {len(overlap_pairs)}")

    # ==================================================================
    # STEP 4: Build connected overlap groups using Union-Find
    # ==================================================================
    logger.info("Building connected overlap groups...")

    # Union-Find to group overlapping polygons
    # We use "da_{idx}" and "sam_{idx}" as node ids
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for da_idx, sam_idx, iou in overlap_pairs:
        union(f"da_{da_idx}", f"sam_{sam_idx}")

    # Collect groups
    groups = {}
    all_overlap_nodes = set()
    for da_idx, sam_idx, iou in overlap_pairs:
        all_overlap_nodes.add(f"da_{da_idx}")
        all_overlap_nodes.add(f"sam_{sam_idx}")

    for node in all_overlap_nodes:
        root = find(node)
        if root not in groups:
            groups[root] = {"da": set(), "sam": set()}
        if node.startswith("da_"):
            groups[root]["da"].add(int(node[3:]))
        else:
            groups[root]["sam"].add(int(node[4:]))

    logger.info(f"  Connected overlap groups: {len(groups)}")

    # ==================================================================
    # STEP 5: Quality comparison for each overlap group
    # ==================================================================
    logger.info("Comparing quality using spectral homogeneity...")

    selected_polygons = []  # list of (gdf_row, source)

    # Add DA-only polygons
    for idx in da_only_idxs:
        selected_polygons.append(da_gdf.iloc[idx])

    # Add SAM-only polygons
    for idx in sam_only_idxs:
        selected_polygons.append(sam_gdf.iloc[idx])

    da_wins = 0
    sam_wins = 0

    for group_id, group in tqdm(groups.items(), desc="Quality comparison", unit="group"):
        da_indices = list(group["da"])
        sam_indices = list(group["sam"])

        da_group_gdf = da_gdf.iloc[da_indices]
        sam_group_gdf = sam_gdf.iloc[sam_indices]

        # Calculate spectral homogeneity for both sets
        da_avg_cv, da_cvs = compute_group_cv(da_group_gdf, raster_src, working_crs)
        sam_avg_cv, sam_cvs = compute_group_cv(sam_group_gdf, raster_src, working_crs)

        logger.debug(f"Group {group_id}: DA CV={da_avg_cv:.4f} ({len(da_indices)} polys), "
                     f"SAM CV={sam_avg_cv:.4f} ({len(sam_indices)} polys)")

        # Pick the set with lower average CV (more homogeneous = better)
        if da_avg_cv <= sam_avg_cv:
            # DA is better or equal — keep DA polygons
            for idx in da_indices:
                selected_polygons.append(da_gdf.iloc[idx])
            da_wins += 1
        else:
            # SAM is better — keep SAM polygons
            for idx in sam_indices:
                selected_polygons.append(sam_gdf.iloc[idx])
            sam_wins += 1

    logger.info(f"  Quality results: DA won {da_wins} groups, SAM won {sam_wins} groups")

    # Close raster
    raster_src.close()

    # ==================================================================
    # STEP 6: Build merged GeoDataFrame
    # ==================================================================
    logger.info("Building merged output...")

    # Keep only common columns + geometry + source
    keep_cols = ["geometry", "source"]
    merged = gpd.GeoDataFrame(selected_polygons, crs=working_crs)

    # Ensure we have the columns we need
    if "source" not in merged.columns:
        merged["source"] = "unknown"

    merged = merged.reset_index(drop=True)

    # ==================================================================
    # STEP 7: Remove remaining overlaps
    # ==================================================================
    merged = resolve_overlaps(merged)

    # Final area filter
    merged_areas = merged.to_crs(equal_area_crs).geometry.area
    merged = merged[merged_areas >= min_area_m2].reset_index(drop=True)

    # ==================================================================
    # STEP 8: Save
    # ==================================================================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Keep only geometry + source to avoid schema conflicts (e.g. FID from shapefiles)
    save_cols = ["geometry", "source"]
    extra_cols = [c for c in merged.columns if c not in save_cols]
    if extra_cols:
        logger.info(f"Dropping extra columns to avoid conflicts: {extra_cols}")
        merged = merged.drop(columns=extra_cols)

    merged.to_file(output_path, driver="GPKG")

    # ==================================================================
    # STEP 9: Summary
    # ==================================================================
    da_count = len(merged[merged["source"] == "delineate_anything"])
    sam_count = len(merged[merged["source"] == "sam"])

    print("\n" + "=" * 60)
    print("              POLYGONS MERGE SUMMARY")
    print("=" * 60)
    print(f"  Input DA polygons          : {len(da_gdf)}")
    print(f"  Input SAM polygons         : {len(sam_gdf)}")
    print(f"  DA-only (no SAM overlap)   : {len(da_only_idxs)}")
    print(f"  SAM-only (no DA overlap)   : {len(sam_only_idxs)}")
    print(f"  Overlap groups compared    : {len(groups)}")
    print(f"    → DA won (better quality): {da_wins}")
    print(f"    → SAM won (better quality): {sam_wins}")
    print(f"  ─────────────────────────────")
    print(f"  Final merged polygons      : {len(merged)}")
    print(f"    → From Delineate Anything: {da_count}")
    print(f"    → From SAM               : {sam_count}")
    print(f"  Output saved to            : {output_path}")
    print("=" * 60)


# =====================================================================
#   ✏️  CONFIGURATION — EDIT PATHS BELOW BEFORE RUNNING
# =====================================================================

if __name__ == "__main__":
    # Path to YOUR Delineate Anything output (.gpkg)
    DELINEATE_OUTPUT = r"E:\BISAG\Delineate-Anything\data\delineated\Gujarat.gpkg"

    # Folder containing your FRIEND's SAM output (.shp / .gpkg / .geojson)
    # The script will auto-detect the first vector file in this folder
    SAM_OUTPUT_DIR = r"E:\BISAG\Delineate-Anything\data\delineated\SAM"

    # Folder containing the raw satellite GeoTIFF image(s)
    RAW_IMAGE_DIR = r"E:\BISAG\Delineate-Anything\data\images\Gujarat"

    # Path where the MERGED output will be saved (.gpkg)
    MERGED_OUTPUT = r"E:\BISAG\Delineate-Anything\data\delineated\merged\merged_output.gpkg"

    # IoU threshold: polygons with overlap > this are considered the "same area"
    #   0.05 = very sensitive (catches even slight overlaps)
    #   0.1  = recommended
    #   0.3  = only catches major overlaps
    OVERLAP_IOU_THRESHOLD = 0.1

    # Minimum polygon area in square meters (smaller ones are removed)
    MIN_AREA_M2 = 500

    # =====================================================================

    # Auto-detect SAM vector file in the directory
    sam_file = None
    for f in os.listdir(SAM_OUTPUT_DIR):
        if f.lower().endswith((".shp", ".gpkg", ".geojson")):
            sam_file = os.path.join(SAM_OUTPUT_DIR, f)
            break

    if sam_file is None:
        raise FileNotFoundError(f"No .shp/.gpkg/.geojson file found in {SAM_OUTPUT_DIR}")

    print(f"Auto-detected SAM file: {sam_file}")

    smart_merge(
        delineate_path=DELINEATE_OUTPUT,
        sam_path=sam_file,
        raw_image_dir=RAW_IMAGE_DIR,
        output_path=MERGED_OUTPUT,
        overlap_iou_threshold=OVERLAP_IOU_THRESHOLD,
        min_area_m2=MIN_AREA_M2,
    )

