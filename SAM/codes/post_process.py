# -*- coding: utf-8 -*-
"""
Post-Processing Script for farm_polygons_upgraded_v2.shp
=========================================================
1. Remove small polygons (trees, noise) using area-based filter
2. Remove border polygons touching the image edge
3. Remove overlapping polygons — one polygon per farm
4. Smooth boundaries — single clean edge between adjacent farms

Output: farm_polygons_final.shp
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.strtree import STRtree
import time
import warnings
import sys
import os

if sys.platform == "win32":
    try:
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])
    except Exception:
        pass

os.environ['PROJ_LIB'] = r'C:\Users\Dishan\anaconda3\envs\sam_env\Library\share\proj'
os.environ['GDAL_DATA'] = r'C:\Users\Dishan\anaconda3\envs\sam_env\Library\share\gdal'

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = r"D:\BISAG\farm_dual_pass.gpkg"   # was .shp
OUTPUT_FILE = r"D:\BISAG\farm_dual_pass_2.gpkg"

# Image bounds (from the GeoTIFF) — polygons touching these edges get removed
IMAGE_BOUNDS = {
    'left':   72.527026,
    'bottom': 23.383899,
    'right':  72.657127,
    'top':    23.493344,
}
BORDER_BUFFER_DEG = 0.0002  # ~22m buffer inside image edges for border detection

# Area thresholds (square meters)
MIN_FARM_AREA_SQM = 300      # Remove anything < 200 sqm (trees, tiny noise fragments)
MAX_FARM_AREA_SQM = 30000  # Remove anomalously large polygons

# Compactness (Polsby-Popper) — removes thin slivers
MIN_COMPACTNESS = 0.08

# Smoothing
SMOOTH_TOLERANCE_M = 1.0    # Douglas-Peucker tolerance (meters)
CHAIKIN_ITERATIONS = 3      # More iterations = smoother boundaries
SHARED_EDGE_SNAP_M = 1.5    # Snap distance for shared boundary consolidation


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_polygons(geom):
    """Extract all Polygon geometries from any geometry type."""
    polys = []
    if geom is None or geom.is_empty:
        return polys
    if geom.geom_type == 'Polygon':
        if geom.area > 0:
            polys.append(geom)
    elif geom.geom_type == 'MultiPolygon':
        for g in geom.geoms:
            if g.area > 0:
                polys.append(g)
    elif geom.geom_type == 'GeometryCollection':
        for g in geom.geoms:
            polys.extend(extract_polygons(g))
    return polys


def fix_geometry(geom):
    """Fix invalid geometry."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def polsby_popper(poly):
    """Polsby-Popper compactness score (0-1, circle=1)."""
    if poly.area == 0 or poly.length == 0:
        return 0
    return (4 * np.pi * poly.area) / (poly.length ** 2)


def chaikin_smooth(coords, iterations=3):
    """Chaikin corner-cutting for smooth boundaries."""
    coords = list(coords)
    for _ in range(iterations):
        if len(coords) < 3:
            return coords
        new_coords = []
        for i in range(len(coords) - 1):
            p0 = coords[i]
            p1 = coords[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_coords.append(q)
            new_coords.append(r)
        if new_coords:
            new_coords.append(new_coords[0])
        coords = new_coords
    return coords


def smooth_polygon(poly, tolerance):
    """Smooth: Douglas-Peucker + Chaikin."""
    if poly.is_empty or poly.area == 0:
        return None
    simplified = poly.simplify(tolerance, preserve_topology=True)
    if simplified.is_empty or simplified.area == 0:
        return poly
    try:
        exterior_coords = list(simplified.exterior.coords)
        smoothed_exterior = chaikin_smooth(exterior_coords, iterations=CHAIKIN_ITERATIONS)
        smoothed_interiors = []
        for interior in simplified.interiors:
            int_coords = list(interior.coords)
            smoothed_int = chaikin_smooth(int_coords, iterations=CHAIKIN_ITERATIONS)
            if len(smoothed_int) >= 4:
                smoothed_interiors.append(smoothed_int)
        result = Polygon(smoothed_exterior, smoothed_interiors)
        if not result.is_valid:
            result = make_valid(result)
            if result.geom_type == 'MultiPolygon':
                result = max(result.geoms, key=lambda g: g.area)
            elif result.geom_type != 'Polygon':
                return simplified
        return result if result.area > 0 else simplified
    except Exception:
        return simplified


# ============================================================
# STEP 1: LOAD AND AREA FILTER + BORDER REMOVAL
# ============================================================

def step1_filter(input_file):
    """Load, remove small polygons (trees/noise) and border polygons."""
    print("=" * 60)
    print("STEP 1: Area filtering + Border removal")
    print("=" * 60)


    # Load input
    gdf = gpd.read_file(input_file)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    original_crs = gdf.crs
    print(f"  Loaded: {len(gdf)} polygons (CRS: {original_crs})")

    # Project to UTM for accurate area
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    print(f"  Projected to UTM: {utm_crs}")

    # Fix all geometries first
    print("  Fixing geometries...")
    fixed_geoms = []
    for geom in gdf_utm.geometry:
        g = fix_geometry(geom)
        if g is not None:
            fixed_geoms.extend(extract_polygons(g))
    gdf_utm = gpd.GeoDataFrame(geometry=fixed_geoms, crs=utm_crs)
    print(f"    → {len(gdf_utm)} valid polygons")

    # --- Area filtering ---
    print(f"\n  Area filtering (min={MIN_FARM_AREA_SQM} sqm, max={MAX_FARM_AREA_SQM} sqm)...")
    areas = gdf_utm.geometry.area
    before = len(gdf_utm)
    mask = (areas >= MIN_FARM_AREA_SQM) & (areas <= MAX_FARM_AREA_SQM)
    gdf_utm = gdf_utm[mask].reset_index(drop=True)
    print(f"    Removed {before - len(gdf_utm)} polygons (too small/large)")
    print(f"    → {len(gdf_utm)} polygons remain")

    # --- Compactness filtering (remove slivers) ---
    print(f"\n  Compactness filtering (min={MIN_COMPACTNESS})...")
    before = len(gdf_utm)
    compactness = gdf_utm.geometry.apply(polsby_popper)
    gdf_utm = gdf_utm[compactness >= MIN_COMPACTNESS].reset_index(drop=True)
    print(f"    Removed {before - len(gdf_utm)} sliver polygons")
    print(f"    → {len(gdf_utm)} polygons remain")

    # --- Border removal ---
    print(f"\n  Border removal (buffer={BORDER_BUFFER_DEG} degrees inside image edge)...")
    border_box_geo = box(
        IMAGE_BOUNDS['left'] + BORDER_BUFFER_DEG,
        IMAGE_BOUNDS['bottom'] + BORDER_BUFFER_DEG,
        IMAGE_BOUNDS['right'] - BORDER_BUFFER_DEG,
        IMAGE_BOUNDS['top'] - BORDER_BUFFER_DEG,
    )
    border_gdf = gpd.GeoDataFrame(geometry=[border_box_geo], crs=original_crs).to_crs(utm_crs)
    inner_box = border_gdf.geometry[0]

    before = len(gdf_utm)
    gdf_utm = gdf_utm[gdf_utm.geometry.within(inner_box)].reset_index(drop=True)
    print(f"    Removed {before - len(gdf_utm)} border polygons")
    print(f"    → {len(gdf_utm)} polygons remain")

    return gdf_utm, original_crs, utm_crs




# ============================================================
# STEP 2: REMOVE OVERLAPS — ONE POLYGON PER FARM
# ============================================================

def step2_remove_overlaps(gdf_utm, utm_crs):
    """
    Remove overlapping polygons so each piece of land belongs to exactly one farm.
    Uses spatial-index for speed. Larger farms get subtracted from where smaller
    farms already exist (smaller farms get priority).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Removing overlaps — one polygon per farm")
    print("=" * 60)

    polygons = list(gdf_utm.geometry)
    n = len(polygons)
    print(f"  Input: {n} polygons")

    # Sort by area: smallest first (they get priority)
    areas = [p.area for p in polygons]
    sorted_indices = sorted(range(n), key=lambda i: areas[i])
    priority = [0] * n
    for rank, orig_idx in enumerate(sorted_indices):
        priority[orig_idx] = rank

    # Build spatial index
    tree = STRtree(polygons)
    result = [None] * n

    for progress, orig_idx in enumerate(sorted_indices):
        if progress % 5000 == 0:
            placed = sum(1 for r in result if r is not None)
            print(f"    Processing {progress}/{n}... (placed: {placed})")

        poly = polygons[orig_idx]

        # Find overlapping neighbors
        neighbor_indices = tree.query(poly)

        # Subtract all higher-priority (smaller, already-placed) neighbors
        for ni in neighbor_indices:
            if ni == orig_idx:
                continue
            if priority[ni] < priority[orig_idx] and result[ni] is not None:
                try:
                    poly = poly.difference(result[ni])
                    if poly.is_empty:
                        break
                except Exception:
                    continue

        # Fix and store
        poly = fix_geometry(poly)
        if poly is not None and not poly.is_empty and poly.area > 0:
            result[orig_idx] = poly

    # Collect results
    final = []
    for p in result:
        if p is not None:
            final.extend(extract_polygons(p))

    print(f"  → {len(final)} non-overlapping polygons")

    # Second pass: remove any tiny slivers created by the overlap removal
    print("  Removing post-overlap slivers...")
    before = len(final)
    final = [p for p in final if p.area >= MIN_FARM_AREA_SQM and polsby_popper(p) >= MIN_COMPACTNESS]
    print(f"    Removed {before - len(final)} slivers")
    print(f"    → {len(final)} clean polygons")

    return gpd.GeoDataFrame(geometry=final, crs=utm_crs)


# ============================================================
# STEP 3: SMOOTH BOUNDARIES — SINGLE EDGE BETWEEN FARMS
# ============================================================

def step3_smooth_boundaries(gdf_utm, utm_crs):
    """
    Smooth polygon boundaries to create clean, single edges between farms.
    
    Strategy:
      1. Buffer-unbuffer each polygon to close micro-gaps between neighbors
      2. Apply Douglas-Peucker simplification to reduce vertex noise
      3. Apply Chaikin smoothing for natural-looking curved boundaries
      4. Snap shared boundaries so adjacent farms share the same edge
      5. Final overlap pass to ensure no new overlaps from smoothing
    """
    print("\n" + "=" * 60)
    print("STEP 3: Smoothing boundaries — single edge between farms")
    print("=" * 60)

    polygons = list(gdf_utm.geometry)
    n = len(polygons)
    print(f"  Input: {n} polygons")

    # --- Sub-step A: Buffer-unbuffer to merge micro-gaps ---
    print("\n  A) Closing micro-gaps between adjacent farms...")
    # Use a small positive buffer to expand polygons, filling gaps,
    # then negative buffer back to original size
    gap_close_dist = SHARED_EDGE_SNAP_M * 0.5  # ~0.75m
    closed = []
    for i, p in enumerate(polygons):
        if i % 10000 == 0 and i > 0:
            print(f"      Processing {i}/{n}...")
        try:
            b = p.buffer(gap_close_dist).buffer(-gap_close_dist)
            b = fix_geometry(b)
            if b is not None and not b.is_empty:
                closed.extend(extract_polygons(b))
            else:
                closed.append(p)
        except Exception:
            closed.append(p)
    print(f"    → {len(closed)} polygons after gap closing")

    # --- Sub-step B: Douglas-Peucker + Chaikin smoothing ---
    print(f"\n  B) Smoothing (DP tolerance={SMOOTH_TOLERANCE_M}m, Chaikin iterations={CHAIKIN_ITERATIONS})...")
    smoothed = []
    for i, p in enumerate(closed):
        if i % 10000 == 0 and i > 0:
            print(f"      Processing {i}/{len(closed)}...")
        sp = smooth_polygon(p, SMOOTH_TOLERANCE_M)
        if sp is not None and sp.area > 0:
            smoothed.append(sp)
    print(f"    → {len(smoothed)} polygons after smoothing")

    # --- Sub-step C: Vertex simplification pass ---
    # Remove redundant vertices that are too close together
    print("\n  C) Removing redundant vertices...")
    simplified = []
    for p in smoothed:
        try:
            sp = p.simplify(SMOOTH_TOLERANCE_M * 0.3, preserve_topology=True)
            if sp.is_valid and sp.area > 0:
                simplified.append(sp)
            else:
                simplified.append(p)
        except Exception:
            simplified.append(p)
    print(f"    → {len(simplified)} polygons")

    # --- Sub-step D: Final overlap cleanup (smoothing can create new overlaps) ---
    print("\n  D) Final overlap cleanup after smoothing...")
    tree = STRtree(simplified)
    areas = [p.area for p in simplified]
    sorted_indices = sorted(range(len(simplified)), key=lambda i: areas[i])
    priority = [0] * len(simplified)
    for rank, idx in enumerate(sorted_indices):
        priority[idx] = rank

    result = [None] * len(simplified)
    for progress, idx in enumerate(sorted_indices):
        if progress % 10000 == 0 and progress > 0:
            print(f"      Processing {progress}/{len(simplified)}...")

        poly = simplified[idx]
        neighbor_indices = tree.query(poly)

        for ni in neighbor_indices:
            if ni == idx:
                continue
            if priority[ni] < priority[idx] and result[ni] is not None:
                try:
                    poly = poly.difference(result[ni])
                    if poly.is_empty:
                        break
                except Exception:
                    continue

        poly = fix_geometry(poly)
        if poly is not None and not poly.is_empty and poly.area > 0:
            result[idx] = poly

    # Collect final polygons
    final = []
    for p in result:
        if p is not None:
            final.extend(extract_polygons(p))

    # Remove any remaining tiny fragments
    final = [p for p in final if p.area >= MIN_FARM_AREA_SQM]
    print(f"    → {len(final)} final smoothed polygons")

    return gpd.GeoDataFrame(geometry=final, crs=utm_crs)


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    print("🌾 Post-Processing Pipeline")
    print("=" * 60)
    print(f"  Input:  {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print()

    # Step 1: Area filter + border removal
    gdf_utm, original_crs, utm_crs = step1_filter(INPUT_FILE)

    # Step 2: Remove overlaps
    gdf_utm = step2_remove_overlaps(gdf_utm, utm_crs)

    # Step 3: Smooth boundaries
    gdf_utm = step3_smooth_boundaries(gdf_utm, utm_crs)

    # ============================================================
    # SAVE OUTPUT
    # ============================================================
    print("\n" + "=" * 60)
    print("SAVING OUTPUT")
    print("=" * 60)

    # Calculate metadata
    gdf_utm["area_sqm"] = gdf_utm.geometry.area
    gdf_utm["perimeter"] = gdf_utm.geometry.length
    gdf_utm["compact"] = gdf_utm.geometry.apply(polsby_popper)

    # Re-project to original CRS
    final_gdf = gdf_utm.to_crs(original_crs)

    print(f"\n  Saving {len(final_gdf)} polygons to: {OUTPUT_FILE}")
    final_gdf.to_file(OUTPUT_FILE, driver="GPKG")

    # Stats
    elapsed = time.time() - start_time
    areas = gdf_utm.geometry.area
    print(f"\n{'=' * 60}")
    print(f"✅ POST-PROCESSING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Input polygons:  {len(gpd.read_file(INPUT_FILE))}")
    print(f"  Output polygons: {len(final_gdf)}")
    print(f"  Area statistics (sqm):")
    print(f"    Min:    {areas.min():.1f}")
    print(f"    Max:    {areas.max():.1f}")
    print(f"    Mean:   {areas.mean():.1f}")
    print(f"    Median: {areas.median():.1f}")
    print(f"  CRS: {original_crs}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
