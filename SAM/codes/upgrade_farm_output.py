# -*- coding: utf-8 -*-
"""
Upgrade Farm Polygon Output
============================
Takes the existing SAM-generated farm_polygons_clean_area.shp and upgrades it:
  1. Gap filling — detects unassigned areas between farms and creates new polygons
  2. Boundary refinement — snaps shared edges, removes micro-gaps/overlaps
  3. Overlap removal — ensures zero overlap between all polygons
  4. Boundary smoothing — Chaikin corner-cutting for natural-looking edges
  5. Area filtering — removes slivers and unreasonably large/small artifacts

Works entirely from the existing shapefile. Does NOT use raw SAM output.
Does NOT add any grid, bounding box, or chunking artifacts.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import unary_union, voronoi_diagram
from shapely.validation import make_valid
from shapely import wkt
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
INPUT_FILE = r"D:\BISAG\SAM\farm_polygons_clean_area.shp"
OUTPUT_FILE = r"D:\BISAG\SAM\farm_polygons_upgraded.shp"

# Area thresholds in square meters (UTM-based)
MIN_FARM_AREA_SQM = 100       # Minimum farm size to keep (sqm)
MAX_FARM_AREA_SQM = 50000     # Maximum farm size (sqm) — anything bigger is not a single farm
MIN_GAP_AREA_SQM = 80         # Minimum gap area worth filling (sqm)
MAX_GAP_AREA_SQM = 15000      # Maximum gap area to fill — larger gaps are likely non-farm

# Boundary refinement
SNAP_TOLERANCE_M = 0.5        # Snap vertices within 0.5m of each other
SMOOTH_TOLERANCE_M = 0.8      # Douglas-Peucker simplification tolerance (meters)
CHAIKIN_ITERATIONS = 2        # Chaikin smoothing iterations

# Buffer for gap detection (meters)
GAP_BUFFER_M = 1.0            # Buffer used to find gaps

# Compactness filter — removes long thin slivers
MIN_COMPACTNESS = 0.08        # Polsby-Popper compactness threshold


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
    """Fix invalid geometry and ensure it's a valid Polygon."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def polsby_popper(poly):
    """Calculate Polsby-Popper compactness score (0 to 1, circle = 1)."""
    if poly.area == 0 or poly.length == 0:
        return 0
    return (4 * np.pi * poly.area) / (poly.length ** 2)


def chaikin_smooth(coords, iterations=2):
    """Apply Chaikin's corner-cutting algorithm for natural-looking boundaries."""
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
    """Smooth a polygon: Douglas-Peucker simplification + Chaikin smoothing."""
    if poly.is_empty or poly.area == 0:
        return None

    # Step 1: Simplify with Douglas-Peucker
    simplified = poly.simplify(tolerance, preserve_topology=True)
    if simplified.is_empty or simplified.area == 0:
        return poly

    try:
        # Step 2: Chaikin smoothing on exterior ring
        exterior_coords = list(simplified.exterior.coords)
        smoothed_exterior = chaikin_smooth(exterior_coords, iterations=CHAIKIN_ITERATIONS)

        # Smooth interior rings (holes)
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
# PHASE 1: LOAD AND VALIDATE
# ============================================================

def phase1_load_and_validate(input_file):
    """Load existing shapefile, fix geometries, project to UTM."""
    print("=" * 60)
    print("PHASE 1: Loading and validating existing polygons")
    print("=" * 60)

    gdf = gpd.read_file(input_file)
    original_crs = gdf.crs
    print(f"  Loaded: {len(gdf)} polygons")
    print(f"  CRS: {original_crs}")

    # Fix geometries
    fixed_geoms = []
    for geom in gdf.geometry:
        g = fix_geometry(geom)
        if g is not None:
            polys = extract_polygons(g)
            fixed_geoms.extend(polys)

    print(f"  After validation: {len(fixed_geoms)} clean polygons")

    # Project to UTM for metric calculations
    gdf_clean = gpd.GeoDataFrame(geometry=fixed_geoms, crs=original_crs)
    utm_crs = gdf_clean.estimate_utm_crs()
    gdf_utm = gdf_clean.to_crs(utm_crs)
    print(f"  Projected to UTM: {utm_crs}")

    return gdf_utm, original_crs, utm_crs


# ============================================================
# PHASE 2: GAP DETECTION AND FILLING
# ============================================================

def phase2_fill_gaps(gdf_utm, utm_crs):
    """
    Detect gaps between existing polygons and fill them with new farm polygons.
    Uses Voronoi-based tessellation to assign gap space to nearest farms.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Detecting and filling gaps between farms")
    print("=" * 60)

    polygons = list(gdf_utm.geometry)
    print(f"  Input polygons: {len(polygons)}")

    # Create the union of all existing polygons
    print("  Computing union of all existing polygons...")
    all_union = unary_union(polygons)

    # Get the overall bounding convex hull of the farm area
    # Use a concave-style hull via buffering to avoid adding farms outside the area
    overall_hull = all_union.convex_hull

    # Shrink the hull slightly inward to avoid creating edge artifacts
    # Use negative buffer then positive buffer to create a slightly smoothed boundary
    hull_shrink = overall_hull.buffer(-GAP_BUFFER_M).buffer(GAP_BUFFER_M * 0.5)
    if hull_shrink.is_empty or hull_shrink.area < 100:
        hull_shrink = overall_hull

    # Find the gap regions = hull minus all existing polygons
    print("  Computing gap regions...")
    gap_region = hull_shrink.difference(all_union)
    gap_region = fix_geometry(gap_region)

    if gap_region is None or gap_region.is_empty:
        print("  No gaps found — polygons fully tile the area!")
        return gdf_utm

    # Extract individual gap polygons
    gap_polygons = extract_polygons(gap_region)
    print(f"  Found {len(gap_polygons)} raw gap regions")

    # Filter gaps by area
    valid_gaps = []
    for gp in gap_polygons:
        area = gp.area
        comp = polsby_popper(gp)
        if MIN_GAP_AREA_SQM <= area <= MAX_GAP_AREA_SQM and comp >= MIN_COMPACTNESS * 0.5:
            valid_gaps.append(gp)

    print(f"  After area/compactness filtering: {len(valid_gaps)} gap polygons to add")

    if len(valid_gaps) == 0:
        return gdf_utm

    # Subdivide large gaps using Voronoi tessellation of their centroids
    # This breaks up big gaps into farm-sized pieces
    subdivided_gaps = []
    for gap in valid_gaps:
        if gap.area > MAX_GAP_AREA_SQM * 0.5:
            # Try to subdivide this large gap
            sub_polys = subdivide_gap(gap, polygons)
            subdivided_gaps.extend(sub_polys)
        else:
            subdivided_gaps.append(gap)

    print(f"  After subdivision: {len(subdivided_gaps)} gap polygons")

    # Combine original polygons with gap fills
    all_polys = polygons + subdivided_gaps
    result_gdf = gpd.GeoDataFrame(geometry=all_polys, crs=utm_crs)
    print(f"  Total polygons after gap filling: {len(result_gdf)}")

    return result_gdf


def subdivide_gap(gap, existing_polygons):
    """
    Subdivide a large gap polygon into smaller pieces based on
    nearest-neighbor assignment from surrounding farm centers.
    """
    try:
        # Find neighboring farms (those that touch or are near this gap)
        gap_buffered = gap.buffer(5)  # 5m buffer for neighbor search
        neighbors = []
        for p in existing_polygons:
            if gap_buffered.intersects(p):
                neighbors.append(p)

        if len(neighbors) < 2:
            return [gap]  # Can't subdivide with <2 neighbors

        # Use the centroids of neighbors as Voronoi seeds
        from shapely.geometry import MultiPoint
        centers = MultiPoint([p.centroid for p in neighbors])

        # Generate Voronoi diagram
        try:
            voronoi = voronoi_diagram(centers, envelope=gap.buffer(50))
        except Exception:
            return [gap]

        # Clip Voronoi cells to the gap
        result = []
        for cell in voronoi.geoms:
            clipped = cell.intersection(gap)
            clipped = fix_geometry(clipped)
            if clipped is not None:
                sub_polys = extract_polygons(clipped)
                for sp in sub_polys:
                    if sp.area >= MIN_GAP_AREA_SQM:
                        result.append(sp)

        return result if result else [gap]

    except Exception:
        return [gap]


# ============================================================
# PHASE 3: BOUNDARY REFINEMENT (Vectorized — Fast)
# ============================================================

def phase3_refine_boundaries(gdf_utm, utm_crs):
    """
    Refine polygon boundaries using fast vectorized operations:
      - Buffer-unbuffer to close micro-gaps in individual polygons
      - Simplify jagged edges with a very conservative tolerance
      - buffer(0) to fix any self-intersections
    
    This replaces the slow pairwise snap approach.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Refining boundaries (fast vectorized)")
    print("=" * 60)

    polygons = list(gdf_utm.geometry)
    print(f"  Input: {len(polygons)} polygons")

    # Step 1: Buffer-unbuffer to close micro-gaps and smooth jagged edges
    print("  Step 1: Closing micro-gaps (buffer-unbuffer)...")
    buf_dist = SNAP_TOLERANCE_M * 0.4  # ~0.2m
    refined = []
    for i, p in enumerate(polygons):
        if i % 5000 == 0 and i > 0:
            print(f"    Processing polygon {i}/{len(polygons)}...")
        try:
            # Expand then shrink to close tiny cracks and smooth jaggies
            buffered = p.buffer(buf_dist).buffer(-buf_dist)
            buffered = fix_geometry(buffered)
            if buffered is not None and not buffered.is_empty:
                polys = extract_polygons(buffered)
                refined.extend(polys)
            else:
                refined.append(p)
        except Exception:
            refined.append(p)

    print(f"    → {len(refined)} polygons after micro-gap closing")

    # Step 2: Simplify with a very small tolerance to remove micro-noise vertices
    print("  Step 2: Removing vertex noise...")
    simplified = []
    for p in refined:
        try:
            sp = p.simplify(SNAP_TOLERANCE_M * 0.5, preserve_topology=True)
            if sp.is_valid and sp.area > 0:
                simplified.append(sp)
            else:
                simplified.append(p)
        except Exception:
            simplified.append(p)
    print(f"    → {len(simplified)} polygons after simplification")

    # Step 3: Fix any remaining invalid geometries
    print("  Step 3: Fixing validity...")
    final = []
    for p in simplified:
        p = fix_geometry(p)
        if p is not None:
            polys = extract_polygons(p)
            final.extend(polys)
    print(f"    → {len(final)} valid polygons")

    result_gdf = gpd.GeoDataFrame(geometry=final, crs=utm_crs)
    return result_gdf


# ============================================================
# PHASE 4: OVERLAP REMOVAL (Spatial-Index Based — Fast)
# ============================================================

def phase4_remove_overlaps(gdf_utm, utm_crs):
    """
    Remove all overlapping areas using spatial index for speed.
    Overlap is assigned to the smaller polygon (farm-friendly heuristic).
    
    Instead of maintaining a cumulative union (O(n²) and exponentially slower),
    this uses STRtree to find only actual neighbors for each polygon and
    subtracts only those with higher priority (smaller area).
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Removing overlaps (spatial-index based)")
    print("=" * 60)

    polygons = list(gdf_utm.geometry)
    n = len(polygons)
    print(f"  Input: {n} polygons")

    # Sort by area — smallest first (they get priority)
    # The rank determines priority: lower rank = smaller = higher priority
    areas = [p.area for p in polygons]
    sorted_indices = sorted(range(n), key=lambda i: areas[i])
    priority = [0] * n  # priority[original_index] = rank
    for rank, orig_idx in enumerate(sorted_indices):
        priority[orig_idx] = rank

    # Build spatial index on the original polygons
    from shapely.strtree import STRtree
    tree = STRtree(polygons)

    result = [None] * n  # result[i] = modified polygon for original index i

    # Process in priority order (smallest first)
    for progress, orig_idx in enumerate(sorted_indices):
        if progress % 2000 == 0:
            print(f"    Processing polygon {progress}/{n}...")

        poly = polygons[orig_idx]

        # Find all polygons that intersect this one
        neighbor_indices = tree.query(poly)

        # Subtract all higher-priority (smaller) neighbors that were already placed
        for ni in neighbor_indices:
            if ni == orig_idx:
                continue
            # Only subtract neighbors with higher priority (lower rank = smaller area)
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

    # Collect non-None results, extract individual polygons
    final = []
    for p in result:
        if p is not None:
            final.extend(extract_polygons(p))

    print(f"  → {len(final)} non-overlapping polygons")
    result_gdf = gpd.GeoDataFrame(geometry=final, crs=utm_crs)
    return result_gdf


# ============================================================
# PHASE 5: SMOOTHING AND FINAL CLEANUP
# ============================================================

def phase5_smooth_and_cleanup(gdf_utm, utm_crs, original_crs):
    """
    Apply boundary smoothing and final area/compactness filtering.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: Smoothing boundaries and final cleanup")
    print("=" * 60)

    polygons = list(gdf_utm.geometry)
    print(f"  Input: {len(polygons)} polygons")

    # Step 1: Smooth boundaries
    print("  Step 1: Smoothing boundaries (Douglas-Peucker + Chaikin)...")
    smoothed = []
    for p in polygons:
        sp = smooth_polygon(p, SMOOTH_TOLERANCE_M)
        if sp is not None and sp.area > 0:
            smoothed.append(sp)
    print(f"    → {len(smoothed)} polygons after smoothing")

    # Step 2: Area filtering
    print("  Step 2: Area filtering...")
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
    print(f"    → {len(area_filtered)} polygons remain")

    # Step 3: Compactness filtering (remove slivers)
    print("  Step 3: Removing slivers (compactness filter)...")
    compact_filtered = []
    removed_slivers = 0
    for p in area_filtered:
        if polsby_popper(p) >= MIN_COMPACTNESS:
            compact_filtered.append(p)
        else:
            removed_slivers += 1
    print(f"    Removed {removed_slivers} sliver polygons")
    print(f"    → {len(compact_filtered)} polygons remain")

    # Step 4: Final validity check
    print("  Step 4: Final validity check...")
    final = []
    for p in compact_filtered:
        p = fix_geometry(p)
        if p is not None:
            polys = extract_polygons(p)
            final.extend(polys)
    print(f"    → {len(final)} final valid polygons")

    return final


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    print("🌾 Farm Polygon Upgrade Script")
    print("=" * 60)
    print(f"  Input:  {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print()

    # Phase 1: Load and validate
    gdf_utm, original_crs, utm_crs = phase1_load_and_validate(INPUT_FILE)

    # Phase 2: Fill gaps
    gdf_utm = phase2_fill_gaps(gdf_utm, utm_crs)

    # Phase 3: Refine boundaries
    gdf_utm = phase3_refine_boundaries(gdf_utm, utm_crs)

    # Phase 4: Remove overlaps
    gdf_utm = phase4_remove_overlaps(gdf_utm, utm_crs)

    # Phase 5: Smooth and cleanup
    final_polys = phase5_smooth_and_cleanup(gdf_utm, utm_crs, original_crs)

    # ============================================================
    # SAVE OUTPUT
    # ============================================================
    print("\n" + "=" * 60)
    print("SAVING OUTPUT")
    print("=" * 60)

    # Build final GeoDataFrame in UTM for area calculation
    final_gdf_utm = gpd.GeoDataFrame(geometry=final_polys, crs=utm_crs)
    
    # Calculate area in square meters
    final_gdf_utm["area_sqm"] = final_gdf_utm.geometry.area
    final_gdf_utm["perimeter"] = final_gdf_utm.geometry.length
    final_gdf_utm["compact"] = final_gdf_utm.apply(
        lambda row: polsby_popper(row.geometry), axis=1
    )

    # Re-project back to original CRS for output
    final_gdf = final_gdf_utm.to_crs(original_crs)

    print(f"\n  Saving {len(final_gdf)} polygons to: {OUTPUT_FILE}")
    final_gdf.to_file(OUTPUT_FILE, engine="fiona")

    # Print summary statistics
    elapsed = time.time() - start_time
    areas = final_gdf_utm.geometry.area
    print(f"\n{'=' * 60}")
    print(f"✅ UPGRADE COMPLETE!")
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
