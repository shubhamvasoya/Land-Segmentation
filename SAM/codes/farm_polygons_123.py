import numpy as np
import torch
import rasterio
import cv2
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.windows import Window

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ==============================
# PATHS
# ==============================
IMAGE_PATH = r"D:\BISAG\GUJ_235182318471_3_JAN_2021_C2EM.tif"
SAM_CHECKPOINT = r"D:\BISAG\sam_vit_b_01ec64.pth"
OUTPUT_FILE = r"D:\BISAG\farm_polygons_123.shp"

# ==============================
# TILE SETTINGS
# ==============================
TILE_SIZE = 1024
STRIDE = 900

device = "cpu"
print("Using CPU")

# ==============================
# LOAD SAM MODEL
# ==============================
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=8,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.93,
    min_mask_region_area=500,
    crop_n_layers=0
)

first_write = True

# ==============================
# PROCESS IMAGE
# ==============================
with rasterio.open(IMAGE_PATH) as src:

    width = src.width
    height = src.height
    transform = src.transform
    crs = src.crs

    print("Image size:", width, height)

    for y in range(0, height, STRIDE):
        for x in range(0, width, STRIDE):

            print(f"Processing tile x:{x}, y:{y}")

            win_w = min(TILE_SIZE, width - x)
            win_h = min(TILE_SIZE, height - y)

            window = Window(x, y, win_w, win_h)
            image = src.read(window=window)

            # -----------------------------
            # HANDLE BANDS
            # -----------------------------
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            else:
                image = image[:3]

            image = image.astype(np.float32)

            # NORMALIZATION
            for i in range(3):
                band = image[i]
                if band.max() > band.min():
                    band = (band - band.min()) / (band.max() - band.min())
                image[i] = band * 255

            image = image.astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))

            # ==============================
            # SAM INFERENCE (CPU SAFE)
            # ==============================
            with torch.no_grad():
                masks = mask_generator.generate(image)

            tile_polygons = []

            # ==============================
            # MASK → POLYGON
            # ==============================
            for m in masks:

                mask = m["segmentation"].astype(np.uint8)

                # Morphological cleanup
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for cnt in contours:

                    if len(cnt) < 4:
                        continue

                    coords = []

                    for pt in cnt:
                        px = pt[0][0] + x
                        py = pt[0][1] + y

                        gx, gy = rasterio.transform.xy(transform, py, px)
                        coords.append((gx, gy))

                    poly = Polygon(coords)

                    if poly.is_valid and poly.area > 0:
                        tile_polygons.append(poly)

            # ==============================
            # SAVE TILE OUTPUT
            # ==============================
            if len(tile_polygons) > 0:

                gdf = gpd.GeoDataFrame(geometry=tile_polygons, crs=crs)

                if first_write:
                    gdf.to_file(OUTPUT_FILE)
                    first_write = False
                else:
                    gdf.to_file(OUTPUT_FILE, mode="a")

print("✅ CPU polygon extraction completed")