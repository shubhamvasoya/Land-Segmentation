import geopandas as gpd
import numpy as np

INPUT = r"D:\BISAG\farm_polygons_clean_cpu.shp"
OUTPUT = r"D:\BISAG\farm_polygons_123.shp"

print("Loading polygons...")

gdf = gpd.read_file(INPUT)

print("Initial polygon count:", len(gdf))

# ---------------------------------------
# FIX CRS (VERY IMPORTANT)
# ---------------------------------------
gdf = gdf.to_crs(3857)

# ---------------------------------------
# FIX INVALID GEOMETRIES
# ---------------------------------------
gdf["geometry"] = gdf.buffer(0)

# ---------------------------------------
# AREA CALCULATION
# ---------------------------------------
gdf["area"] = gdf.area

# ---------------------------------------
# REMOVE VERY SMALL POLYGONS (noise)
# ---------------------------------------
small_thresh = gdf["area"].quantile(0.005)

gdf = gdf[gdf["area"] > small_thresh]

print("After removing small noise:", len(gdf))

# ---------------------------------------
# REMOVE VERY LARGE POLYGONS (tile artifacts)
# ---------------------------------------
large_thresh = gdf["area"].quantile(0.998)

gdf = gdf[gdf["area"] < large_thresh]

print("After removing large polygons:", len(gdf))

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
print("Resolving overlaps...")

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
print("Smoothing farm boundaries...")

gdf["geometry"] = gdf.buffer(0.5).buffer(-0.5)

# ---------------------------------------
# CLEANUP
# ---------------------------------------
gdf = gdf.drop(columns=["area", "ratio"], errors="ignore")

gdf = gdf.reset_index(drop=True)

gdf = gdf.to_crs(4326)

print("Final polygon count:", len(gdf))

# ---------------------------------------
# SAVE OUTPUT
# ---------------------------------------
gdf.to_file(OUTPUT)

print("Saved clean farm polygons:", OUTPUT)