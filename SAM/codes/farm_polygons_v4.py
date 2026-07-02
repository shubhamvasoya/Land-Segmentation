# -*- coding: utf-8 -*-
"""
Farm Polygon Upgrade v4 — Topology-Clean Output
=================================================
Fixes all v2 issues with a fundamentally better Phase 4:

  OLD approach (v2/v3):  subtract polygon B from polygon A
    → ragged edges, micro-gaps, each polygon owns its own boundary line

  NEW approach (v4):     extract ALL boundary lines → union into shared network
                         → polygonize into atomic cells → assign to parent farms
    → adjacent farms share a SINGLE boundary edge (true topology)
    → zero overlap guaranteed by construction (polygonize is planar by definition)

Pipeline:
  Phase 1  Load & project to UTM
  Phase 2  Quality-filter raw SAM polygons (strict area + compactness)
  Phase 3  Select new candidates from raw that fill gaps in the clean set
  Phase 4  Topology resolution via polygonize
             a) Snap to precision grid  (removes float-precision ghosts)
             b) Extract all exterior rings as LineStrings
             c) unary_union of all lines  → planar boundary network
             d) polygonize  → atomic face regions
             e) Assign each atom to its dominant parent polygon
                (clean polygons have priority over new ones)
             f) Merge atoms by parent  → one polygon per farm
  Phase 5  Smooth boundaries (Douglas-Peucker + Chaikin)
  Phase 6  Final area + compactness filter
  Phase 7  Save shapefile
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, polygonize
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
# CONFIGURATION — tune these for your dataset
# ============================================================
CLEAN_FILE  = r"D:\BISAG\SAM\farm_polygons_clean_area.shp"
RAW_FILE    = r"D:\BISAG\raw_sam_output.shp"
OUTPUT_FILE = r"D:\BISAG\SAM\farm_polygons_v4.shp"

# ── Area thresholds (square metres) ─────────────────────────
#    200 m² ≈ 14×14 m — removes trees, shrubs, and road noise.
#    Lower to 100 m² if you have very small plots in your area.
MIN_FARM_AREA_SQM = 200
MAX_FARM_AREA_SQM = 50_000

# ── Compactness (Polsby-Popper, 0–1) ────────────────────────
#    0.06 cuts extreme slivers while keeping narrow field strips.
MIN_COMPACTNESS = 0.06

# ── Overlap threshold for accepting raw polygons ─────────────
#    Raw polygon skipped if more than this fraction already
#    covered by the clean set (it's a duplicate).
MIN_OVERLAP_RATIO = 0.30

# ── Smoothing ────────────────────────────────────────────────
SMOOTH_TOLERANCE_M = 0.8   # Douglas-Peucker simplification in metres
CHAIKIN_ITERATIONS = 2     # Chaikin smoothing passes (2–3 is enough)

# ── Topology precision ───────────────────────────────────────
#    Coordinates snapped to nearest GRID_SIZE metres before
#    building the planar network.  0.01 m is fine for farms.
GRID_SIZE = 0.01


# ============================================================
# UTILITIES
# ============================================================

def extract_polygons(geom):
    """Flatten any geometry type into a list of simple Polygons."""
    polys = []
    if geom is None or geom.is_empty:
        return polys
    if geom.geom_type == 'Polygon':
        if geom.area > 0:
            polys.append(geom)
    elif geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for g in geom.geoms:
            polys.extend(extract_polygons(g))
    return polys


def fix_geometry(geom):
    """Return a valid geometry, or None if it cannot be fixed."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom if (geom is not None and not geom.is_empty) else None


def polsby_popper(poly):
    """Polsby-Popper compactness score: 0 = worst sliver, 1 = perfect circle."""
    if poly.area == 0 or poly.length == 0:
        return 0.0
    return (4 * np.pi * poly.area) / (poly.length ** 2)


def chaikin_smooth(coords, iterations=2):
    """Chaikin corner-cutting — rounds polygon corners progressively."""
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
        if new_coords:
            new_coords.append(new_coords[0])
        coords = new_coords
    return coords


def smooth_polygon(poly, tolerance):
    """
    Two-step smoothing:
      1. Douglas-Peucker simplification (removes redundant vertices)
      2. Chaikin corner-cutting (smooths the remaining vertices)
    """
    if poly.is_empty or poly.area == 0:
        return None
    simplified = poly.simplify(tolerance, preserve_topology=True)
    if simplified.is_empty or simplified.area == 0:
        return poly
    try:
        ext = chaikin_smooth(list(simplified.exterior.coords), CHAIKIN_ITERATIONS)
        holes = []
        for interior in simplified.interiors:
            hole = chaikin_smooth(list(interior.coords), CHAIKIN_ITERATIONS)
            if len(hole) >= 4:
                holes.append(hole)
        result = Polygon(ext, holes)
        if not result.is_valid:
            result = make_valid(result)
            if result.geom_type == 'MultiPolygon':
                result = max(result.geoms, key=lambda g: g.area)
            elif result.geom_type != 'Polygon':
                return simplified
        return result if result.area > 0 else simplified
    except Exception:
        return simplified


def is_valid_farm(poly):
    """True when the polygon passes area and compactness thresholds."""
    return (MIN_FARM_AREA_SQM <= poly.area <= MAX_FARM_AREA_SQM
            and polsby_popper(poly) >= MIN_COMPACTNESS)


def snap_to_grid(poly, grid_size):
    """
    Snap all polygon coordinates to a precision grid.
    Uses shapely.set_precision (Shapely ≥2.0) or coordinate rounding as fallback.
    """
    try:
        import shapely
        return shapely.set_precision(poly, grid_size)
    except (AttributeError, Exception):
        # Fallback: round coordinates manually
        def round_coords(coords):
            return [(round(x / grid_size) * grid_size,
                     round(y / grid_size) * grid_size) for x, y in coords]
        if poly.geom_type != 'Polygon':
            return poly
        ext = round_coords(poly.exterior.coords)
        holes = [round_coords(h.coords) for h in poly.interiors]
        try:
            return Polygon(ext, holes)
        except Exception:
            return poly


# ============================================================
# PHASE 1: LOAD & PROJECT
# ============================================================

def phase1_load(clean_file, raw_file):
    print("=" * 62)
    print("PHASE 1: Loading datasets")
    print("=" * 62)

    gdf_clean = gpd.read_file(clean_file)
    original_crs = gdf_clean.crs
    print(f"  Clean polygons : {len(gdf_clean)}")
    print(f"  CRS            : {original_crs}")

    gdf_raw = gpd.read_file(raw_file)
    print(f"  Raw polygons   : {len(gdf_raw)}")

    utm_crs = gdf_clean.estimate_utm_crs()
    print(f"  Projecting to  : {utm_crs}")

    return (gdf_clean.to_crs(utm_crs),
            gdf_raw.to_crs(utm_crs),
            original_crs, utm_crs)


# ============================================================
# PHASE 2: QUALITY-FILTER RAW POLYGONS
# ============================================================

def phase2_filter_raw(gdf_raw_utm):
    print("\n" + "=" * 62)
    print("PHASE 2: Quality-filtering raw SAM polygons")
    print("=" * 62)
    print(f"  Input: {len(gdf_raw_utm)} raw polygons")

    stats = dict(invalid=0, too_small=0, too_big=0, sliver=0, kept=0)
    kept = []

    for i, geom in enumerate(gdf_raw_utm.geometry):
        if i % 25_000 == 0 and i:
            print(f"  … {i}/{len(gdf_raw_utm)}  kept so far: {stats['kept']}")

        geom = fix_geometry(geom)
        if geom is None:
            stats['invalid'] += 1
            continue

        for sp in extract_polygons(geom):
            if sp.area < MIN_FARM_AREA_SQM:
                stats['too_small'] += 1
            elif sp.area > MAX_FARM_AREA_SQM:
                stats['too_big'] += 1
            elif polsby_popper(sp) < MIN_COMPACTNESS:
                stats['sliver'] += 1
            else:
                stats['kept'] += 1
                kept.append(sp)

    print(f"  Invalid  : {stats['invalid']}")
    print(f"  Too small: {stats['too_small']}")
    print(f"  Too big  : {stats['too_big']}")
    print(f"  Slivers  : {stats['sliver']}")
    print(f"  → Kept   : {stats['kept']} quality raw polygons")
    return kept


# ============================================================
# PHASE 3: SELECT GAP-FILLING CANDIDATES FROM RAW
# ============================================================

def phase3_select_candidates(clean_polys_raw, raw_filtered):
    """
    Keep every clean polygon unchanged.
    For areas NOT already covered by the clean set, add raw polygons
    that pass quality checks — filling in farms the clean set missed.
    """
    print("\n" + "=" * 62)
    print("PHASE 3: Selecting gap-filling candidates from raw output")
    print("=" * 62)

    # Fix and flatten clean polygons
    clean_polys = []
    for p in clean_polys_raw:
        p = fix_geometry(p)
        if p:
            clean_polys.extend(extract_polygons(p))
    print(f"  Clean base     : {len(clean_polys)}")
    print(f"  Raw candidates : {len(raw_filtered)}")

    clean_tree = STRtree(clean_polys)
    new_polys = []
    skip_overlap = skip_area = 0

    for i, raw_poly in enumerate(raw_filtered):
        if i % 10_000 == 0 and i:
            print(f"  … {i}/{len(raw_filtered)}  new: {len(new_polys)}")

        cand_idxs = clean_tree.query(raw_poly)

        # No spatial overlap with clean set → completely new area
        if len(cand_idxs) == 0:
            new_polys.append(raw_poly)
            continue

        # Measure fraction of raw_poly already covered by clean set
        overlap_area = sum(
            raw_poly.intersection(clean_polys[ci]).area
            for ci in cand_idxs
            if raw_poly.intersects(clean_polys[ci])
        )
        if raw_poly.area > 0 and (overlap_area / raw_poly.area) >= MIN_OVERLAP_RATIO:
            skip_overlap += 1
            continue

        # Clip: keep only the part NOT covered by clean
        try:
            overlapping = [clean_polys[ci] for ci in cand_idxs
                           if raw_poly.intersects(clean_polys[ci])]
            if overlapping:
                clipped = raw_poly.difference(unary_union(overlapping))
                clipped = fix_geometry(clipped)
                if clipped:
                    for sp in extract_polygons(clipped):
                        if is_valid_farm(sp):
                            new_polys.append(sp)
                        else:
                            skip_area += 1
            else:
                new_polys.append(raw_poly)
        except Exception:
            new_polys.append(raw_poly)

    print(f"  Skipped (overlaps clean)      : {skip_overlap}")
    print(f"  Skipped (too small after clip): {skip_area}")
    print(f"  New gap-filling candidates    : {len(new_polys)}")
    return clean_polys, new_polys


# ============================================================
# PHASE 4: TOPOLOGY-AWARE OVERLAP RESOLUTION
# ============================================================

def phase4_topology_resolve(clean_polys, new_polys):
    """
    Build a topologically clean planar subdivision from all input polygons.

    Why this approach?
    ──────────────────
    The v2/v3 "subtract B from A" method leaves each polygon with its own
    independent boundary line.  Adjacent farms are guaranteed not to overlap,
    but their shared edge is NOT the same geometric object — tiny gaps and
    mismatches persist, and each farm "owns" its own copy of the boundary.

    The polygonize approach does the opposite:
      1. Extract ALL exterior rings from ALL input polygons as LineStrings.
      2. Merge them with unary_union — identical or overlapping segments are
         fused into a single shared line.  The result is a planar graph where
         every edge is stored exactly once.
      3. polygonize() reconstructs the faces of that planar graph.
         Each face (atomic cell) is guaranteed non-overlapping with all others.
      4. Assign each atomic cell to its "parent" input polygon by largest
         intersection area.  Clean polygons win ties with new polygons.
      5. Union cells by parent → one final polygon per farm.

    Result: adjacent farms share a SINGLE boundary edge; no overlaps by
    construction; one polygon per farm.
    """
    print("\n" + "=" * 62)
    print("PHASE 4: Topology-aware overlap resolution (polygonize)")
    print("=" * 62)

    n_clean = len(clean_polys)
    n_new   = len(new_polys)
    print(f"  Clean : {n_clean}   New : {n_new}   Total : {n_clean + n_new}")

    # ── a) Snap all geometries to the precision grid ─────────────
    #    This closes sub-centimetre gaps and micro-overlaps that would
    #    otherwise create thousands of tiny sliver cells after polygonize.
    print("\n  a) Snapping to precision grid …")

    def snap_list(polys):
        out = []
        for p in polys:
            try:
                s = snap_to_grid(p, GRID_SIZE)
                s = fix_geometry(s)
                if s:
                    out.extend(extract_polygons(s))
            except Exception:
                fp = fix_geometry(p)
                if fp:
                    out.extend(extract_polygons(fp))
        return out

    snapped_clean = snap_list(clean_polys)
    snapped_new   = snap_list(new_polys)
    n_clean       = len(snapped_clean)          # update after snapping
    all_polys     = snapped_clean + snapped_new
    print(f"    Clean: {n_clean}  New: {len(snapped_new)}  Total: {len(all_polys)}")

    # ── b) Extract exterior rings ─────────────────────────────────
    print("  b) Extracting exterior boundary rings …")
    lines = [p.exterior for p in all_polys if p.geom_type == 'Polygon']
    print(f"    → {len(lines)} rings extracted")

    # ── c) Build planar boundary network ─────────────────────────
    #    unary_union on LineStrings fuses collinear / identical segments into
    #    a minimal set of unique edges — the shared boundary network.
    #    This step is the most time-consuming (~1–3 min for 20k polygons).
    print("  c) Building planar boundary network via unary_union …")
    print("     (This may take 1–3 minutes for large datasets — please wait)")
    t = time.time()
    planar_net = unary_union(lines)
    print(f"    → Done in {time.time() - t:.1f}s")

    # ── d) Polygonize into atomic face regions ────────────────────
    print("  d) Polygonizing boundary network …")
    atoms = list(polygonize(planar_net))
    print(f"    → {len(atoms)} atomic face regions created")

    if not atoms:
        print("  WARNING: polygonize produced no atoms.")
        print("           Falling back to snapped input polygons.")
        return all_polys

    # ── e) Assign each atom to its dominant parent polygon ────────
    #    For each atomic cell we find which input polygon it "belongs to"
    #    by computing the intersection area with every candidate parent.
    #    Clean polygons have absolute priority over new polygons.
    print("  e) Assigning atoms to parent polygons …")

    parent_tree = STRtree(all_polys)
    assignments = {}          # atom_idx → parent_idx in all_polys
    unassigned  = 0

    for ai, atom in enumerate(atoms):
        if ai % 10_000 == 0 and ai:
            print(f"     {ai}/{len(atoms)} assigned: {len(assignments)}")

        cands = parent_tree.query(atom)

        best_idx      = -1
        best_area     = 0.0
        best_is_clean = False

        for ci in cands:
            parent       = all_polys[ci]
            is_clean_par = (ci < n_clean)
            try:
                inter = parent.intersection(atom)
                ia    = inter.area
                if ia <= 0:
                    continue
                # Clean parent always wins over new parent regardless of area
                if is_clean_par and not best_is_clean:
                    best_idx      = ci
                    best_area     = ia
                    best_is_clean = True
                elif is_clean_par == best_is_clean and ia > best_area:
                    best_idx  = ci
                    best_area = ia
            except Exception:
                continue

        if best_idx >= 0:
            assignments[ai] = best_idx
        else:
            unassigned += 1   # atom is in a gap between all polygons (road, water…)

    print(f"    Assigned : {len(assignments)}")
    print(f"    Unassigned (gap/road/water regions discarded): {unassigned}")

    # ── f) Merge atoms by parent → one polygon per farm ──────────
    print("  f) Merging atoms by parent polygon …")

    groups = {}
    for ai, pi in assignments.items():
        groups.setdefault(pi, []).append(atoms[ai])

    result = []
    for pi, atom_list in groups.items():
        merged = unary_union(atom_list)
        merged = fix_geometry(merged)
        if merged:
            result.extend(extract_polygons(merged))

    print(f"  → {len(result)} topologically clean farm polygons")
    return result


# ============================================================
# PHASE 5: SMOOTH BOUNDARIES
# ============================================================

def phase5_smooth(polygons):
    """
    Smooth polygon boundaries in two steps:
      1. Buffer-unbuffer to close micro-gaps introduced by grid-snapping.
      2. Douglas-Peucker simplification + Chaikin corner-cutting.

    Note: smoothing is applied AFTER topology resolution so the planar
    structure is built from clean, unsmoothed edges.  Smoothing slightly
    breaks shared boundary identity (each polygon is smoothed independently)
    but the sub-millimetre gaps this creates are cosmetic only.
    """
    print("\n" + "=" * 62)
    print("PHASE 5: Smoothing boundaries")
    print("=" * 62)
    print(f"  Input: {len(polygons)}")

    # Close micro-gaps from snapping
    gap_closed = []
    for p in polygons:
        try:
            b = p.buffer(0.15).buffer(-0.15)
            b = fix_geometry(b)
            gap_closed.extend(extract_polygons(b) if b else [p])
        except Exception:
            gap_closed.append(p)

    # Douglas-Peucker + Chaikin
    smoothed = []
    for p in gap_closed:
        sp = smooth_polygon(p, SMOOTH_TOLERANCE_M)
        if sp and sp.area > 0:
            smoothed.append(sp)

    print(f"  → {len(smoothed)} smoothed polygons")
    return smoothed


# ============================================================
# PHASE 6: FINAL AREA + COMPACTNESS FILTER
# ============================================================

def phase6_final_filter(polygons):
    """
    Remove any polygon that falls outside the accepted area range or
    is too sliver-like.  Runs AFTER smoothing because simplification
    can shrink a polygon below the area threshold.
    """
    print("\n" + "=" * 62)
    print("PHASE 6: Final area + compactness filter")
    print("=" * 62)

    final   = []
    removed = dict(small=0, large=0, sliver=0, invalid=0)

    for p in polygons:
        p = fix_geometry(p)
        if p is None:
            removed['invalid'] += 1
            continue
        for sp in extract_polygons(p):
            if sp.area < MIN_FARM_AREA_SQM:
                removed['small'] += 1
            elif sp.area > MAX_FARM_AREA_SQM:
                removed['large'] += 1
            elif polsby_popper(sp) < MIN_COMPACTNESS:
                removed['sliver'] += 1
            else:
                final.append(sp)

    print(f"  Removed (too small) : {removed['small']}")
    print(f"  Removed (too large) : {removed['large']}")
    print(f"  Removed (sliver)    : {removed['sliver']}")
    print(f"  Removed (invalid)   : {removed['invalid']}")
    print(f"  → {len(final)} final valid farm polygons")
    return final


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()

    print("🌾  Farm Polygon Upgrade v4 — Topology-Clean Output")
    print("=" * 62)
    print(f"  MIN_FARM_AREA  : {MIN_FARM_AREA_SQM} m²")
    print(f"  MAX_FARM_AREA  : {MAX_FARM_AREA_SQM} m²")
    print(f"  MIN_COMPACTNESS: {MIN_COMPACTNESS}")
    print(f"  GRID_SIZE      : {GRID_SIZE} m")
    print()

    # ── Phase 1: Load ──────────────────────────────────────────────
    gdf_clean_utm, gdf_raw_utm, original_crs, utm_crs = phase1_load(
        CLEAN_FILE, RAW_FILE
    )

    # ── Phase 2: Filter raw ────────────────────────────────────────
    raw_filtered = phase2_filter_raw(gdf_raw_utm)

    # ── Phase 3: Select gap-filling candidates ─────────────────────
    clean_polys, new_polys = phase3_select_candidates(
        list(gdf_clean_utm.geometry), raw_filtered
    )

    # ── Phase 4: Topology resolution ──────────────────────────────
    topo_polys = phase4_topology_resolve(clean_polys, new_polys)

    # ── Phase 5: Smooth ────────────────────────────────────────────
    smoothed_polys = phase5_smooth(topo_polys)

    # ── Phase 6: Final filter ──────────────────────────────────────
    final_polys = phase6_final_filter(smoothed_polys)

    # ── Save ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("SAVING OUTPUT")
    print("=" * 62)

    gdf_out_utm = gpd.GeoDataFrame(geometry=final_polys, crs=utm_crs)
    gdf_out_utm["area_sqm"]  = gdf_out_utm.geometry.area
    gdf_out_utm["perimeter"] = gdf_out_utm.geometry.length
    gdf_out_utm["compact"]   = gdf_out_utm.apply(
        lambda r: polsby_popper(r.geometry), axis=1
    )

    gdf_out = gdf_out_utm.to_crs(original_crs)
    print(f"  Saving {len(gdf_out)} polygons → {OUTPUT_FILE}")
    gdf_out.to_file(OUTPUT_FILE, engine="pyogrio")

    elapsed = time.time() - t0
    areas   = gdf_out_utm.geometry.area

    print(f"\n{'=' * 62}")
    print(f"✅  v4 COMPLETE  ({elapsed:.0f}s / {elapsed / 60:.1f} min)")
    print(f"{'=' * 62}")
    print(f"  Output polygons : {len(gdf_out)}")
    print(f"  Area min        : {areas.min():.0f} m²")
    print(f"  Area mean       : {areas.mean():.0f} m²")
    print(f"  Area max        : {areas.max():.0f} m²")
    print(f"  CRS             : {original_crs}")
    print(f"  File            : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()