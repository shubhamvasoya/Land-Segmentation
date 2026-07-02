import os

# ── Fix GDAL/PROJ paths for Conda on Windows ──────────────────────────────────
conda_env = os.path.dirname(os.path.dirname(os.__file__))   # e.g. C:\Users\...\anaconda3\envs\sam_env
os.environ.setdefault("GDAL_DATA", os.path.join(conda_env, "Library", "share", "gdal"))
os.environ.setdefault("PROJ_LIB",  os.path.join(conda_env, "Library", "share", "proj"))



import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("fiona").setLevel(logging.ERROR)

import numpy as np
import torch
import rasterio
import cv2
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.validation import make_valid
from rasterio.windows import Window
import gc
import time
import pickle
import os
from datetime import timedelta
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE_PATH      = r"D:\BISAG\GUJ_235182318471_3_JAN_2021_C2EM.tif"
SAM_CHECKPOINT  = r"D:\BISAG\sam_vit_b_01ec64.pth"
OUTPUT_FILE = r"D:\BISAG\farm_polygons_gpu.gpkg"   # change extension
PROGRESS_FILE   = r"D:\BISAG\farm_polygons_checkpoint.pkl"   # ← checkpoint path

TILE_SIZE = 1024
STRIDE    = 768     # 256px overlap — catches farms that fall on tile edges

# ── GPU Setup ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU detected: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")

    if vram_gb <= 4:
        POINTS_PER_BATCH = 32
    elif vram_gb <= 8:
        POINTS_PER_BATCH = 64
    else:
        POINTS_PER_BATCH = 128

    torch.backends.cudnn.benchmark = True
else:
    device           = "cpu"
    POINTS_PER_BATCH = 32
    print("No GPU found — falling back to CPU")

print(f"Running on: {device.upper()}  |  points_per_batch={POINTS_PER_BATCH}")

# ── Load SAM ──────────────────────────────────────────────────────────────────
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device)

if device == "cuda":
    sam = sam.half()
    print("Mixed precision (fp16) enabled")

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    points_per_batch=POINTS_PER_BATCH,
    pred_iou_thresh=0.80,
    stability_score_thresh=0.85,
    box_nms_thresh=0.7,
    min_mask_region_area=50,
    crop_n_layers=1,
    crop_overlap_ratio=0.5,
    crop_n_points_downscale_factor=2,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_tile(image: np.ndarray) -> np.ndarray:
    """Percentile clip (2–98 %) + CLAHE per channel."""
    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(3):
        band = image[i].astype(np.float32)
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi > lo:
            band = np.clip(band, lo, hi)
            band = (band - lo) / (hi - lo) * 255.0
        result[i] = band.astype(np.uint8)

    img_hwc   = np.transpose(result, (1, 2, 0))
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = np.zeros_like(img_hwc)
    for c in range(3):
        enhanced[:, :, c] = clahe.apply(img_hwc[:, :, c])

    return enhanced


def mask_to_polygons(mask: np.ndarray, tile_x: int, tile_y: int, transform) -> list:
    """Convert one SAM mask to a list of georeferenced Shapely polygons."""
    mask_u8 = mask.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )

    polys = []
    for cnt in contours:
        if len(cnt) < 4:
            continue

        coords = []
        for pt in cnt:
            px = pt[0][0] + tile_x
            py = pt[0][1] + tile_y
            gx, gy = rasterio.transform.xy(transform, py, px)
            coords.append((gx, gy))

        if len(coords) < 4:
            continue

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_valid and poly.area > 0:
            polys.append(poly)

    return polys


def format_eta(seconds: float) -> str:
    """Convert seconds into a human-readable HH:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))


def save_checkpoint(path: str, tile_num: int, polygons: list) -> None:
    """Persist current progress to disk."""
    with open(path, "wb") as f:
        pickle.dump({"tile_num": tile_num, "polygons": polygons}, f)


def load_checkpoint(path: str) -> tuple[int, list]:
    """Load previously saved progress; returns (last_tile, polygons)."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Checkpoint found — resuming from tile {data['tile_num']}  "
              f"({len(data['polygons'])} polygons already saved)")
        return data["tile_num"], data["polygons"]
    return 0, []


CHECKPOINT_EVERY = 25   # save to disk every N tiles


# ── Load Checkpoint (if any) ──────────────────────────────────────────────────
start_tile, all_polygons = load_checkpoint(PROGRESS_FILE)

# ── Main Loop ─────────────────────────────────────────────────────────────────
tile_times: list[float] = []   # rolling window for ETA

with rasterio.open(IMAGE_PATH) as src:
    width     = src.width
    height    = src.height
    transform = src.transform
    crs       = src.crs

    tiles_x = (width  + STRIDE - 1) // STRIDE
    tiles_y = (height + STRIDE - 1) // STRIDE
    total   = tiles_x * tiles_y
    print(f"\nImage: {width} × {height} px  |  Tiles: {total}  ({tiles_x} × {tiles_y})\n")

    tile_num = 0

    for y in range(0, height, STRIDE):
        for x in range(0, width, STRIDE):
            tile_num += 1

            # ── Resume support — skip already-processed tiles ──────────────
            if tile_num <= start_tile:
                continue

            t0 = time.perf_counter()

            win_w  = min(TILE_SIZE, width  - x)
            win_h  = min(TILE_SIZE, height - y)
            window = Window(x, y, win_w, win_h)
            image  = src.read(window=window)

            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            else:
                image = image[:3]

            if image.max() == image.min():
                elapsed = time.perf_counter() - t0
                tile_times.append(elapsed)
                remaining = (total - tile_num) * (sum(tile_times[-20:]) / len(tile_times[-20:]))
                print(f"  [{tile_num:>4}/{total}] ({x:>5},{y:>5}) — blank, skipped"
                      f"  |  ETA {format_eta(remaining)}")
                continue

            image = normalize_tile(image)

            with torch.no_grad():
                if device == "cuda":
                    with torch.amp.autocast("cuda"):    # replaces deprecated cuda.amp.autocast
                        masks = mask_generator.generate(image)
                else:
                    masks = mask_generator.generate(image)

            tile_polygons = []
            for m in masks:
                tile_polygons.extend(mask_to_polygons(m["segmentation"], x, y, transform))

            all_polygons.extend(tile_polygons)

            elapsed = time.perf_counter() - t0
            tile_times.append(elapsed)

            # ETA based on rolling average of last 20 tiles
            window_size  = min(20, len(tile_times))
            avg_time     = sum(tile_times[-window_size:]) / window_size
            remaining    = (total - tile_num) * avg_time
            elapsed_fmt  = f"{elapsed:.1f}s"

            print(
                f"  [{tile_num:>4}/{total}] ({x:>5},{y:>5})"
                f"  masks={len(masks):>4}  polys={len(tile_polygons):>4}"
                f"  total={len(all_polygons):>6}"
                f"  |  tile={elapsed_fmt:<6}  ETA {format_eta(remaining)}"
            )

            del masks, tile_polygons, image
            gc.collect()

            if device == "cuda":
                torch.cuda.empty_cache()

            # ── Checkpoint every N tiles ───────────────────────────────────
            if tile_num % CHECKPOINT_EVERY == 0:
                save_checkpoint(PROGRESS_FILE, tile_num, all_polygons)
                print(f"  ── checkpoint saved at tile {tile_num} ──")


# ── Post-processing ───────────────────────────────────────────────────────────
print(f"\nRaw polygons (pre-dedup): {len(all_polygons)}")

if all_polygons:
    gdf = gpd.GeoDataFrame(geometry=all_polygons)
    before = len(gdf)
    gdf    = gdf.drop_duplicates(subset="geometry")
    print(f"Dropped {before - len(gdf)} exact duplicates  →  {len(gdf)} polygons remaining")

    gdf.to_file(OUTPUT_FILE, driver="GPKG")
    print(f"\nSaved → {OUTPUT_FILE}")

    # Clean up checkpoint once fully done
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"Checkpoint file removed.")
else:
    print("No polygons detected.")

print("Processing finished")