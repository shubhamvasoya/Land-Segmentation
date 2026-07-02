# -*- coding: utf-8 -*-
"""
Upgrade Farm Polygons v2 — More Farms, Same Quality
=====================================================
Takes the clean 15k farm polygons as the base and fills in missing farms 
from the raw SAM output (110k polygons). Applies quality filtering so 
only good polygons are added.

Strategy:
  1. Load clean 15k polygons as HIGH-PRIORITY base
  2. Load raw 110k SAM polygons, filter for quality
  3. Find raw polygons that cover NEW areas (not already in clean set)
  4. Add those new polygons to the base
  5. Remove overlaps, smooth, cleanup → output

Does NOT add grids, boxes, or chunking artifacts.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.strtree import STRtree
import time
import warnings
import sys
import os

# Force UTF-8 for Windows console
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
CLEAN_FILE = r"D:\BISAG\SAM\farm_polygons_clean_area.shp"
RAW_FILE = r"D:\BISAG\raw_sam_output.shp"
OUTPUT_FILE = r"D:\BISAG\SAM\farm_polygons_upgraded_v2.shp"

# Area thresholds in square meters
MIN_FARM_AREA_SQM = 50         # Min farm size — keep smaller farms!
MAX_FARM_AREA_SQM = 50000      # Max farm size
MIN_NEW_AREA_SQM = 30          # Min area for a "new" raw polygon to qualify

# Quality filters
MIN_COMPACTNESS = 0.06          # Polsby-Popper: removes thin slivers
MIN_OVERLAP_RATIO = 0.3        # If a raw polygon overlaps >30% with clean, skip it

# Smoothing
SMOOTH_TOLERANCE_M = 0.8
CHAIKIN_ITERATIONS = 2


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
    """Polsby-Popper compactness (0-1, circle=1)."""
    if poly.area == 0 or poly.length == 0:
        return 0
    return (4 * np.pi * poly.area) / (poly.length ** 2)


def chaikin_smooth(coords, iterations=2):
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
# PHASE 1: LOAD BOTH DATASETS
# ============================================================

def phase1_load(clean_file, raw_file):
    """Load clean and raw shapefiles, project to UTM."""
    print("=" * 60)
    print("PHASE 1: Loading datasets")
    print("=" * 60)

    # Load clean
    print(f"  Loading clean file: {clean_file}")
    gdf_clean = gpd.read_file(clean_file)
    original_crs = gdf_clean.crs
    print(f"    → {len(gdf_clean)} clean polygons (CRS: {original_crs})")

    # Load raw
    print(f"  Loading raw file: {raw_file}")
    gdf_raw = gpd.read_file(raw_file)
    print(f"    → {len(gdf_raw)} raw polygons")

    # Project both to UTM
    utm_crs = gdf_clean.estimate_utm_crs()
    print(f"  Projecting to UTM: {utm_crs}")
    gdf_clean_utm = gdf_clean.to_crs(utm_crs)
    gdf_raw_utm = gdf_raw.to_crs(utm_crs)

    return gdf_clean_utm, gdf_raw_utm, original_crs, utm_crs


# ============================================================
# PHASE 2: FILTER RAW POLYGONS FOR QUALITY
# ============================================================

def phase2_filter_raw(gdf_raw_utm):
    """Filter raw polygons: area, compactness, validity."""
    print("\n" + "=" * 60)
    print("PHASE 2: Quality-filtering raw polygons")
    print("=" * 60)

    polygons = list(gdf_raw_utm.geometry)
    print(f"  Input: {len(polygons)} raw polygons")

    filtered = []
    stats = {'invalid': 0, 'too_small': 0, 'too_big': 0, 'sliver': 0, 'kept': 0}

    for i, p in enumerate(polygons):
        if i % 20000 == 0 and i > 0:
            print(f"    Processing {i}/{len(polygons)}... (kept: {stats['kept']})")

        # Fix geometry
        p = fix_geometry(p)
        if p is None:
            stats['invalid'] += 1
            continue

        # Extract polygons
        sub_polys = extract_polygons(p)
        for sp in sub_polys:
            area = sp.area
            
            # Area filter
            if area < MIN_NEW_AREA_SQM:
                stats['too_small'] += 1
                continue
            if area > MAX_FARM_AREA_SQM:
                stats['too_big'] += 1
                continue

            # Compactness filter
            comp = polsby_popper(sp)
            if comp < MIN_COMPACTNESS:
                stats['sliver'] += 1
                continue

            stats['kept'] += 1
            filtered.append(sp)

    print(f"  Filtering results:")
    print(f"    Invalid:   {stats['invalid']}")
    print(f"    Too small: {stats['too_small']}")
    print(f"    Too big:   {stats['too_big']}")
    print(f"    Slivers:   {stats['sliver']}")
    print(f"    → Kept:    {stats['kept']} quality raw polygons")

    return filtered


# ============================================================
# PHASE 3: FIND NEW POLYGONS (NOT IN CLEAN SET)
# ============================================================

def phase3_find_new_polygons(clean_polys, raw_filtered, utm_crs):
    """
    Find raw polygons that cover areas NOT already covered by clean polygons.
    A raw polygon is "new" if it doesn't significantly overlap with any clean polygon.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Finding new farm polygons from raw output")
    print("=" * 60)

    print(f"  Clean base: {len(clean_polys)} polygons")
    print(f"  Raw candidates: {len(raw_filtered)} polygons")

    # Build spatial index on clean polygons for fast intersection queries
    clean_tree = STRtree(clean_polys)

    new_polygons = []
    skipped_overlap = 0
    skipped_empty = 0

    for i, raw_poly in enumerate(raw_filtered):
        if i % 10000 == 0 and i > 0:
            print(f"    Checked {i}/{len(raw_filtered)}... "
                  f"(new: {len(new_polygons)}, skipped: {skipped_overlap})")

        # Find which clean polygons this raw polygon might overlap with
        candidate_indices = clean_tree.query(raw_poly)

        if len(candidate_indices) == 0:
            # No overlap at all — this is a completely new farm!
            new_polygons.append(raw_poly)
            continue

        # Check actual overlap
        total_overlap_area = 0
        for ci in candidate_indices:
            clean_p = clean_polys[ci]
            try:
                if raw_poly.intersects(clean_p):
                    intersection = raw_poly.intersection(clean_p)
                    total_overlap_area += intersection.area
            except Exception:
                continue

        overlap_ratio = total_overlap_area / raw_poly.area if raw_poly.area > 0 else 1.0

        if overlap_ratio < MIN_OVERLAP_RATIO:
            # This raw polygon mostly covers NEW area — keep it!
            # Clip it to only the non-overlapping part
            try:
                overlapping_clean = [clean_polys[ci] for ci in candidate_indices 
                                     if raw_poly.intersects(clean_polys[ci])]
                if overlapping_clean:
                    clean_union = unary_union(overlapping_clean)
                    clipped = raw_poly.difference(clean_union)
                    clipped = fix_geometry(clipped)
                    if clipped is not None and not clipped.is_empty:
                        sub_polys = extract_polygons(clipped)
                        for sp in sub_polys:
                            if sp.area >= MIN_NEW_AREA_SQM:
                                new_polygons.append(sp)
                            else:
                                skipped_empty += 1
                    else:
                        skipped_empty += 1
                else:
                    new_polygons.append(raw_poly)
            except Exception:
                new_polygons.append(raw_poly)
        else:
            skipped_overlap += 1

    print(f"\n  Results:")
    print(f"    New polygons found:     {len(new_polygons)}")
    print(f"    Skipped (>30% overlap): {skipped_overlap}")
    print(f"    Skipped (too small after clip): {skipped_empty}")

    return new_polygons


# ============================================================
# PHASE 4: COMBINE AND REMOVE OVERLAPS
# ============================================================

def phase4_combine_and_deoverlap(clean_polys, new_polys, utm_crs):
    """
    Combine clean + new polygons, then remove overlaps.
    Clean polygons get highest priority (they keep their area).
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Combining and removing overlaps")
    print("=" * 60)

    # Tag polygons: clean ones have priority
    n_clean = len(clean_polys)
    all_polys = clean_polys + new_polys
    n_total = len(all_polys)
    print(f"  Clean: {n_clean}, New: {len(new_polys)}, Total: {n_total}")

    # Sort by priority: clean first (by their order), then new ones by area (smallest first)
    # This ensures clean polygons are placed first and new ones don't eat into them
    clean_indices = list(range(n_clean))
    new_indices = list(range(n_clean, n_total))
    # Sort new polygons by area (smallest first)
    new_indices.sort(key=lambda i: all_polys[i].area)
    
    processing_order = clean_indices + new_indices

    # Build spatial index
    tree = STRtree(all_polys)
    result = [None] * n_total

    # Process in priority order
    for progress, idx in enumerate(processing_order):
        if progress % 5000 == 0:
            placed = sum(1 for r in result if r is not None)
            print(f"    Processing {progress}/{n_total}... (placed: {placed})")

        poly = all_polys[idx]

        # Find neighbors
        neighbor_indices = tree.query(poly)

        # Subtract all already-placed neighbors
        for ni in neighbor_indices:
            if ni == idx:
                continue
            if result[ni] is not None:
                try:
                    poly = poly.difference(result[ni])
                    if poly.is_empty:
                        break
                except Exception:
                    continue

        # Fix and store
        poly = fix_geometry(poly)
        if poly is not None and not poly.is_empty and poly.area > 0:
            result[idx] = poly

    # Collect results
    final = []
    clean_kept = 0
    new_kept = 0
    for i, p in enumerate(result):
        if p is not None:
            polys = extract_polygons(p)
            final.extend(polys)
            if i < n_clean:
                clean_kept += 1
            else:
                new_kept += 1

    print(f"\n  Results:")
    print(f"    Clean polygons kept: {clean_kept}")
    print(f"    New polygons added:  {new_kept}")
    print(f"    Total polygons:      {len(final)}")

    return final


# ============================================================
# PHASE 5: SMOOTHING AND FINAL CLEANUP
# ============================================================

def phase5_smooth_and_cleanup(polygons, utm_crs, original_crs):
    """Smooth boundaries and final filtering."""
    print("\n" + "=" * 60)
    print("PHASE 5: Smoothing and final cleanup")
    print("=" * 60)

    print(f"  Input: {len(polygons)} polygons")

    # Step 1: Buffer-unbuffer to close micro-cracks
    print("  Step 1: Closing micro-gaps...")
    refined = []
    for p in polygons:
        try:
            b = p.buffer(0.2).buffer(-0.2)
            b = fix_geometry(b)
            if b is not None and not b.is_empty:
                refined.extend(extract_polygons(b))
            else:
                refined.append(p)
        except Exception:
            refined.append(p)
    print(f"    → {len(refined)} polygons")

    # Step 2: Smooth boundaries
    print("  Step 2: Smoothing boundaries (Douglas-Peucker + Chaikin)...")
    smoothed = []
    for p in refined:
        sp = smooth_polygon(p, SMOOTH_TOLERANCE_M)
        if sp is not None and sp.area > 0:
            smoothed.append(sp)
    print(f"    → {len(smoothed)} polygons")

    # Step 3: Area filtering
    print("  Step 3: Area filtering...")
    area_filtered = []
    removed_small = 0
    removed_big = 0
    for p in smoothed:
        area = p.area
        if area < MIN_FARM_AREA_SQM:
            removed_small += 1
            continue
        if area > MAX_FARM_AREA_SQM:
            removed_big += 1
            continue
        area_filtered.append(p)
    print(f"    Removed {removed_small} too-small, {removed_big} too-large")
    print(f"    → {len(area_filtered)} polygons")

    # Step 4: Compactness filter
    print("  Step 4: Removing slivers...")
    compact = []
    removed_slivers = 0
    for p in area_filtered:
        if polsby_popper(p) >= MIN_COMPACTNESS:
            compact.append(p)
        else:
            removed_slivers += 1
    print(f"    Removed {removed_slivers} slivers")
    print(f"    → {len(compact)} polygons")

    # Step 5: Final validity
    print("  Step 5: Final validity check...")
    final = []
    for p in compact:
        p = fix_geometry(p)
        if p is not None:
            final.extend(extract_polygons(p))
    print(f"    → {len(final)} final valid polygons")

    return final


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    print("🌾 Farm Polygon Upgrade v2 — More Farms, Same Quality")
    print("=" * 60)
    print(f"  Clean base:  {CLEAN_FILE}")
    print(f"  Raw source:  {RAW_FILE}")
    print(f"  Output:      {OUTPUT_FILE}")
    print()

    # Phase 1: Load
    gdf_clean_utm, gdf_raw_utm, original_crs, utm_crs = phase1_load(CLEAN_FILE, RAW_FILE)

    # Phase 2: Filter raw polygons
    raw_filtered = phase2_filter_raw(gdf_raw_utm)

    # Phase 3: Find new polygons not in clean set
    clean_polys = list(gdf_clean_utm.geometry)
    # Fix clean polygons
    clean_fixed = []
    for p in clean_polys:
        p = fix_geometry(p)
        if p is not None:
            clean_fixed.extend(extract_polygons(p))
    clean_polys = clean_fixed

    new_polys = phase3_find_new_polygons(clean_polys, raw_filtered, utm_crs)

    # Phase 4: Combine and remove overlaps
    combined = phase4_combine_and_deoverlap(clean_polys, new_polys, utm_crs)

    # Phase 5: Smooth and cleanup
    final_polys = phase5_smooth_and_cleanup(combined, utm_crs, original_crs)

    # ============================================================
    # SAVE OUTPUT
    # ============================================================
    print("\n" + "=" * 60)
    print("SAVING OUTPUT")
    print("=" * 60)

    final_gdf_utm = gpd.GeoDataFrame(geometry=final_polys, crs=utm_crs)
    final_gdf_utm["area_sqm"] = final_gdf_utm.geometry.area
    final_gdf_utm["perimeter"] = final_gdf_utm.geometry.length
    final_gdf_utm["compact"] = final_gdf_utm.apply(
        lambda row: polsby_popper(row.geometry), axis=1
    )

    # Re-project to original CRS
    final_gdf = final_gdf_utm.to_crs(original_crs)

    print(f"\n  Saving {len(final_gdf)} polygons to: {OUTPUT_FILE}")
    final_gdf.to_file(OUTPUT_FILE, engine="fiona")

    # Stats
    elapsed = time.time() - start_time
    areas = final_gdf_utm.geometry.area
    print(f"\n{'=' * 60}")
    print(f"✅ UPGRADE v2 COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Original clean polygons: {len(gpd.read_file(CLEAN_FILE))}")
    print(f"  Raw SAM polygons used:   {len(gpd.read_file(RAW_FILE))}")
    print(f"  OUTPUT polygons:         {len(final_gdf)}")
    print(f"  Area statistics (sqm):")
    print(f"    Min:    {areas.min():.1f}")
    print(f"    Max:    {areas.max():.1f}")
    print(f"    Mean:   {areas.mean():.1f}")
    print(f"    Median: {areas.median():.1f}")
    print(f"  CRS: {original_crs}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
