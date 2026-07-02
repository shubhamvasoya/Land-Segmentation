import os
os.environ["GDAL_DATA"] = r"C:\Users\Dishan\anaconda3\envs\sam_env\Library\share\gdal"
os.environ["PROJ_LIB"]  = r"C:\Users\Dishan\anaconda3\envs\sam_env\Library\share\proj"


import geopandas as gpd
import numpy as np

INPUT = r"D:\BISAG\farm_dual_pass.gpkg"
OUTPUT = r"D:\BISAG\farm_dual_pass_1.gpkg"

print("Loading polygons...")

gdf = gpd.read_file(INPUT)

print("Initial polygon count:", len(gdf))

# ---------------------------------------
# FIX CRS (for correct area)
# ---------------------------------------

gdf = gdf.to_crs(3857)

# ---------------------------------------
# FIX GEOMETRIES
# ---------------------------------------

gdf["geometry"] = gdf.buffer(0)

# ---------------------------------------
# AREA CALCULATION
# ---------------------------------------

gdf["area"] = gdf.area

# ---------------------------------------
# QUANTILE THRESHOLDS
# ---------------------------------------

small_thresh = gdf["area"].quantile(0.005)
large_thresh = gdf["area"].quantile(0.998)

print("\n=== AREA THRESHOLDS ===")
print("Small threshold (m²):", small_thresh)
print("Small threshold (km²):", small_thresh / 1e6)

print("Large threshold (m²):", large_thresh)
print("Large threshold (km²):", large_thresh / 1e6)

# ---------------------------------------
# REPORT REMOVED SMALL POLYGONS
# ---------------------------------------

removed_small = gdf[gdf["area"] <= small_thresh]

print("\n=== REMOVED SMALL POLYGONS ===")
print("Count:", len(removed_small))

if len(removed_small) > 0:
    print("Min area:", removed_small["area"].min())
    print("Max area:", removed_small["area"].max())

# ---------------------------------------
# REPORT REMOVED LARGE POLYGONS
# ---------------------------------------

removed_large = gdf[gdf["area"] >= large_thresh]

print("\n=== REMOVED LARGE POLYGONS ===")
print("Count:", len(removed_large))

if len(removed_large) > 0:
    print("Min area:", removed_large["area"].min())
    print("Max area:", removed_large["area"].max())

# ---------------------------------------
# APPLY FILTER
# ---------------------------------------

gdf = gdf[
    (gdf["area"] > small_thresh) &
    (gdf["area"] < large_thresh)
]

print("\nAfter area filtering:", len(gdf))

# ---------------------------------------
# REMOVE RECTANGULAR TILE BOXES
# ---------------------------------------

def aspect_ratio(poly):
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    if h == 0:
        return 0
    return max(w/h, h/w)

gdf["ratio"] = gdf.geometry.apply(aspect_ratio)

gdf = gdf[gdf["ratio"] < 6]

print("After removing tile rectangles:", len(gdf))

# ---------------------------------------
# REMOVE OVERLAPPING POLYGONS
# ---------------------------------------

print("\nResolving overlaps...")

gdf = gdf.sort_values("area", ascending=False).reset_index(drop=True)

sindex = gdf.sindex

keep = []
removed = set()

for i, geom in enumerate(gdf.geometry):

    if i in removed:
        continue

    keep.append(i)

    possible = list(sindex.intersection(geom.bounds))

    for j in possible:

        if j <= i or j in removed:
            continue

        other = gdf.geometry[j]

        if geom.intersects(other):

            inter = geom.intersection(other).area

            if inter / other.area > 0.6:
                removed.add(j)

gdf = gdf.loc[keep]

print("After overlap removal:", len(gdf))

# ---------------------------------------
# SMOOTH BOUNDARIES
# ---------------------------------------

print("\nSmoothing farm boundaries...")

gdf["geometry"] = gdf.buffer(0.5).buffer(-0.5)

# ---------------------------------------
# CLEANUP
# ---------------------------------------

gdf = gdf.drop(columns=["area", "ratio"], errors="ignore")

gdf = gdf.reset_index(drop=True)

gdf = gdf.to_crs(4326)

print("\nFinal polygon count:", len(gdf))

# ---------------------------------------
# SAVE OUTPUT
# ---------------------------------------

gdf.to_file(OUTPUT, driver="GPKG")

print("\nSaved clean farm polygons:", OUTPUT)