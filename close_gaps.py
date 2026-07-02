"""
close_gaps.py — Spatial-Batched Watershed Gap Closure for SAM Polygons
======================================================================
Removes gaps between adjacent farm polygons using the TRUE watershed algorithm.
To drastically reduce memory, polygons are grouped into spatial tiles/blocks based 
on their centroids. Each block is processed independently with surrounding polygons 
included as context (so there are no straight-line cutting artifacts).

Usage:
    conda run -n lanseg python close_gaps.py
"""

import os
import logging
import warnings
import math
import gc

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes as rasterio_shapes
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from scipy.ndimage import binary_closing
from skimage.segmentation import watershed
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================================
#   HELPER FUNCTIONS
# =====================================================================

def compute_gradient(image_data: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude from an image array."""
    if image_data.ndim == 3 and image_data.shape[0] > 1:
        n_bands = min(3, image_data.shape[0])
        gray = image_data[:n_bands].astype(np.float32).mean(axis=0)
    else:
        gray = image_data[0].astype(np.float32)

    g_min, g_max = gray.min(), gray.max()
    if g_max > g_min:
        gray_norm = ((gray - g_min) / (g_max - g_min) * 255).astype(np.uint8)
    else:
        gray_norm = gray.astype(np.uint8)

    del gray
    gc.collect()

    sobelx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)
    del gray_norm
    gc.collect()

    # In-place magnitude calculation to avoid numpy temporaries (OOM fix)
    cv2.multiply(sobelx, sobelx, sobelx)
    cv2.multiply(sobely, sobely, sobely)
    cv2.add(sobelx, sobely, sobelx)
    del sobely
    gc.collect()
    
    cv2.sqrt(sobelx, sobelx)

    # Normalise to [0, 1]
    grad_max = float(np.max(sobelx))
    if grad_max > 0:
        cv2.multiply(sobelx, 1.0 / grad_max, sobelx)

    return sobelx

def process_block(core_indices, context_indices, gdf, src, dilation_pixels, raster_res):
    """
    Apply watershed logic to a single tile block.
    
    Parameters
    ----------
    core_indices : set
        Indices of polygons that "belong" to this block.
    context_indices : set
        Indices of surrounding polygons included to provide seamless boundaries.
    gdf : geopandas.GeoDataFrame
        The full dataframe, containing '_label' column and geometries.
    src : rasterio.DatasetReader
        Open reference raster.
    dilation_pixels : int
        Dilation amount for the watershed mask.
    raster_res : float
        Approximate raster resolution in CRS units.
    """
    all_indices = list(core_indices.union(context_indices))
    local_gdf = gdf.iloc[all_indices]
    
    # Compute bounds with buffer
    local_bounds = local_gdf.total_bounds
    buffer_m = dilation_pixels * raster_res * 2
    bbox_buffered = (
        local_bounds[0] - buffer_m,
        local_bounds[1] - buffer_m,
        local_bounds[2] + buffer_m,
        local_bounds[3] + buffer_m,
    )
    
    raster_bounds = src.bounds
    bbox_buffered = (
        max(bbox_buffered[0], raster_bounds.left),
        max(bbox_buffered[1], raster_bounds.bottom),
        min(bbox_buffered[2], raster_bounds.right),
        min(bbox_buffered[3], raster_bounds.top),
    )
    
    # Read the image window covering this local extent
    win = window_from_bounds(*bbox_buffered, transform=src.transform)
    win = win.round_lengths().round_offsets()
    
    if win.width <= 0 or win.height <= 0:
        return []
        
    win_transform = src.window_transform(win)
    image_data = src.read(window=win)
    H, W = image_data.shape[1], image_data.shape[2]
    
    gradient = compute_gradient(image_data)
    del image_data
    gc.collect()
        
    # Rasterise markers
    shapes_with_labels = [
        (mapping(row.geometry), int(row["_label"]))
        for _, row in local_gdf.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    ]
    
    markers = rasterize(
        shapes_with_labels,
        out_shape=(H, W),
        transform=win_transform,
        fill=0,
        dtype=np.int32,
        all_touched=False,
    )
    
    # Dilate footprint and then erode (Binary Closing) to create processing mask
    # This ensures we bridge narrow gaps but DO NOT expand isolated outer boundaries!
    polygon_footprint = markers > 0
    closed_mask = binary_closing(polygon_footprint, iterations=dilation_pixels)
    
    # Ensure original footprint is fully preserved
    processing_mask = closed_mask | polygon_footprint
    
    # Watershed
    labels = watershed(
        image=gradient,
        markers=markers,
        mask=processing_mask,
        connectivity=2,
        compactness=0.0,
    )
    
    # Vectorise
    label_int32 = labels.astype(np.int32)
    result_shapes = list(rasterio_shapes(
        label_int32,
        mask=(labels > 0).astype(np.uint8),
        transform=win_transform,
        connectivity=8,
    ))
    
    # Filter output polygons based on core properties
    core_labels = set(gdf.iloc[list(core_indices)]["_label"].values)
    return_records = []
    
    for geom_dict, label_val in result_shapes:
        lv = int(label_val)
        if lv in core_labels:
            return_records.append({"geometry": shape(geom_dict), "_label": lv})
            
    return return_records

# =====================================================================
#   MAIN RUNNER
# =====================================================================

def close_gaps_watershed(
    input_gpkg: str,
    raw_image_path: str,
    output_gpkg: str,
    block_size_m: float = 3000.0,
    context_buffer_m: float = 200.0,
    dilation_pixels: int = 20,
    min_area_m2: float = 200.0,
):
    # 1. Load polygons
    logger.info(f"Loading polygons: {input_gpkg}")
    gdf = gpd.read_file(input_gpkg)
    n_input = len(gdf)
    logger.info(f"  → {n_input} polygons loaded, CRS: {gdf.crs}")
    
    if n_input == 0:
        raise ValueError("Input GeoPackage contains no polygons.")

    # Assign base labels (1-based index)
    gdf = gdf.reset_index(drop=True)
    gdf["_label"] = gdf.index + 1

    # 2. Open Reference Raster
    logger.info(f"Opening reference raster: {raw_image_path}")
    src = rasterio.open(raw_image_path)
    raster_crs = src.crs
    raster_res_x = abs(src.transform.a)
    raster_res_y = abs(src.transform.e)
    raster_res = max(raster_res_x, raster_res_y)
    
    if str(gdf.crs) != str(raster_crs):
        logger.info(f"  Reprojecting polygons from {gdf.crs} → {raster_crs}")
        gdf = gdf.to_crs(raster_crs)
        
    sindex = gdf.sindex
    gdf["centroid"] = gdf.geometry.centroid
    bounds = gdf.total_bounds # minx, miny, maxx, maxy
    
    # Determine block sizes based on CRS
    is_geographic = src.crs.is_geographic
    if is_geographic:
        logger.info("  CRS is geographic (degrees). Converting block sizes to degrees.")
        block_size_crs = block_size_m / 111320.0
        context_buffer_crs = context_buffer_m / 111320.0
    else:
        block_size_crs = block_size_m
        context_buffer_crs = context_buffer_m
        
    minx, miny, maxx, maxy = bounds
    cols = int(math.ceil((maxx - minx) / block_size_crs))
    rows = int(math.ceil((maxy - miny) / block_size_crs))
    
    logger.info(f"  → Tiling into {cols}x{rows} blocks (Block size: {block_size_crs:.4f} units)")
    
    all_results = []
    
    # 3. Process Blocks
    total_blocks = cols * rows
    with tqdm(total=total_blocks, desc="Processing Blocks") as pbar:
        for r in range(rows):
            for c in range(cols):
                b_minx = minx + c * block_size_crs
                b_miny = miny + r * block_size_crs
                b_maxx = b_minx + block_size_crs
                b_maxy = b_miny + block_size_crs
                
                # Core polygons are those whose centroid falls perfectly within the block
                core_mask = (
                    (gdf["centroid"].x >= b_minx) & (gdf["centroid"].x < b_maxx) &
                    (gdf["centroid"].y >= b_miny) & (gdf["centroid"].y < b_maxy)
                )
                core_indices = set(gdf[core_mask].index.tolist())
                
                if not core_indices:
                    pbar.update(1)
                    continue
                
                # Context polygons are those that intersect a buffered box
                buffered_box = box(
                    b_minx - context_buffer_crs, 
                    b_miny - context_buffer_crs, 
                    b_maxx + context_buffer_crs, 
                    b_maxy + context_buffer_crs
                )
                intersect_indices = list(sindex.intersection(buffered_box.bounds))
                context_indices = set(intersect_indices) - core_indices
                
                # Run the watershed on this block
                records = process_block(
                    core_indices=core_indices,
                    context_indices=context_indices,
                    gdf=gdf,
                    src=src,
                    dilation_pixels=dilation_pixels,
                    raster_res=raster_res
                )
                
                all_results.extend(records)
                pbar.update(1)

    src.close()
    
    # 4. Assemble Final DataFrame
    logger.info(f"Vectorised parts collected. Assembling final DataFrame...")
    result_gdf = gpd.GeoDataFrame(all_results, crs=raster_crs)
    
    if len(result_gdf) == 0:
        logger.warning("No polygons were generated!")
        return
        
    # Explode multi-parts
    result_gdf = result_gdf.explode(index_parts=False).reset_index(drop=True)
    
    # Filter by Area
    eq_crs = "EPSG:6933"
    areas = result_gdf.to_crs(eq_crs).geometry.area
    result_gdf = result_gdf[areas >= min_area_m2].reset_index(drop=True)
    
    # Attach Attributes
    orig_cols = [c for c in gdf.columns if c not in ("geometry", "_label", "centroid")]
    label_to_attrs = {}
    for _, row in gdf.iterrows():
        label_to_attrs[int(row["_label"])] = {c: row[c] for c in orig_cols}

    for col in orig_cols:
        result_gdf[col] = result_gdf["_label"].map(
            lambda lv, col=col: label_to_attrs.get(lv, {}).get(col, None)
        )

    result_gdf = result_gdf.drop(columns=["_label"])
    
    # Match original CRS
    logger.info("Saving results...")
    original_crs = gpd.read_file(input_gpkg, rows=1).crs
    if str(result_gdf.crs) != str(original_crs):
        result_gdf = result_gdf.to_crs(original_crs)

    # 5. Save Output
    os.makedirs(os.path.dirname(os.path.abspath(output_gpkg)), exist_ok=True)
    result_gdf.to_file(output_gpkg, driver="GPKG")
    
    print("\n" + "=" * 60)
    print("         BATCHED WATERSHED SUMMARY")
    print("=" * 60)
    print(f"  Input  polygons  : {n_input}")
    print(f"  Output polygons  : {len(result_gdf)}")
    print(f"  Block grid       : {cols} x {rows} blocks")
    print(f"  Output saved to  : {output_gpkg}")
    print("=" * 60)

# =====================================================================
#   ✏️  CONFIGURATION
# =====================================================================

if __name__ == "__main__":
    INPUT_GPKG = r"E:\BISAG\Delineate-Anything\data\delineated\SAM\farm_dual_pass_2.gpkg"
    RAW_IMAGE = r"E:\BISAG\Delineate-Anything\data\images\Gujarat\GUJ_235182318471_3_JAN_2021_C2EM.tif"
    OUTPUT_GPKG = r"E:\BISAG\Delineate-Anything\data\delineated\SAM\farm_dual_pass_2_no_gaps.gpkg"
    
    BLOCK_SIZE_M = 3000.0     # 3000x3000m blocks
    CONTEXT_BUFFER_M = 300.0  # Polygons within 300m of block are used as boundaries
    DILATION_PIXELS = 10      # Distance to bridge gaps. Lowered to 10 so it doesn't bridge across huge open lands.
    MIN_AREA_M2 = 200.0

    close_gaps_watershed(
        input_gpkg=INPUT_GPKG,
        raw_image_path=RAW_IMAGE,
        output_gpkg=OUTPUT_GPKG,
        block_size_m=BLOCK_SIZE_M,
        context_buffer_m=CONTEXT_BUFFER_M,
        dilation_pixels=DILATION_PIXELS,
        min_area_m2=MIN_AREA_M2,
    )
