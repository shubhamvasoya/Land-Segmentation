<div align="center">

# 🌾 Delineate Anything

### AI-Based Agricultural Land Segmentation System

**Automated farm boundary extraction from satellite imagery using Foundation Models, YOLO-based tiling, and Spectral Homogeneity Merging.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-0099E5?style=for-the-badge)](https://ultralytics.com)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](LICENSE)
[![BISAG-N](https://img.shields.io/badge/Organization-BISAG--N-green?style=for-the-badge)](https://bisag-n.gov.in)

*Developed at BISAG-N · Gujarat, India*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Pipeline Stages](#-pipeline-stages)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration Reference](#-configuration-reference)
- [Utility Scripts](#-utility-scripts)
- [Troubleshooting](#-troubleshooting)
- [License & Credits](#-license--credits)

---

## 🔭 Overview

**Delineate Anything** is a fully automated, end-to-end geospatial pipeline for extracting agricultural field boundaries from large-scale satellite GeoTIFF imagery. Developed at **BISAG-N (Bhaskaracharya National Institute for Space Applications and Geo-informatics)**, the system combines two foundation AI models into a unified, quality-aware output:

1. **Delineate Anything (YOLO-based)** — A pretrained instance segmentation model optimized for agricultural field delineation, handling massive images through a two-level tiling hierarchy with intelligent cross-tile merging.

2. **SAM Pipeline** — Meta's Segment Anything Model (SAM) in Automatic Mask Generation mode, with dual-pass fusion (quality + quantity) and CLAHE-enhanced preprocessing to detect thin bunds.

3. **Smart Merge** — A post-processing layer that fuses both model outputs using **Spectral Coefficient of Variation (CV)** to pick the boundary with greater spectral homogeneity, followed by **Watershed Gap Closing** to fill inter-farm slivers.

**Tested on:** 8060 × 8382 px, 4-band (NIR/R/G/B), ~0.59 m/px GeoTIFF of Gujarat, India · EPSG:32643 (UTM Zone 43N) · NVIDIA GTX 1650 (4 GB VRAM)

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                 │
│     GeoTIFF Satellite Images  +  YAML Configuration Files          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
             ┌─────────────────▼──────────────────┐
             │           INFERENCE LAYER           │
             │                                     │
             │  ┌─────────────────────────────┐   │
             │  │  Delineate Anything (YOLO)  │   │
             │  │  Tiling → Inference →       │   │
             │  │  Polygonization             │   │
             │  └──────────────┬──────────────┘   │
             │                 │                   │
             │  ┌──────────────▼──────────────┐   │
             │  │     SAM Pipeline            │   │
             │  │  Grid Prompting → Mask →    │   │
             │  │  Vector Conversion          │   │
             │  └──────────────┬──────────────┘   │
             └─────────────────┼──────────────────┘
                               │
             ┌─────────────────▼──────────────────┐
             │         MERGING & REFINING          │
             │                                     │
             │  CV-Based Overlap Comparison        │
             │  (Union-Find Overlap Groups)        │
             │                                     │
             │  Watershed Gap Closing              │
             │  Chaikin Smoothing                  │
             └─────────────────┬──────────────────┘
                               │
             ┌─────────────────▼──────────────────┐
             │           OUTPUT LAYER             │
             │  Merged GeoPackage (.gpkg) → QGIS  │
             └────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Dual Foundation Model** | Combines YOLO (Delineate Anything) + SAM for complementary detection |
| 🗺️ **Large Image Handling** | Two-level tiling (4096 px regions → 512 px tiles) with configurable overlap |
| ⚡ **Parallel Processing** | Multiprocessing workers for post-processing, polygonization & simplification |
| 🧩 **Cross-Tile Merging** | Union-Find algorithm resolves field IDs across tile and region boundaries |
| 📊 **Spectral Quality Merge** | CV-based (σ/μ) model selection — picks the most spectrally homogeneous boundary |
| 🌊 **Watershed Gap Closing** | Expands polygons along image gradients to fill inter-farm slivers |
| 🔍 **LCLU Mask Integration** | Optional land cover mask to filter non-agricultural areas |
| 📐 **Topology-Aware Simplification** | Douglas-Peucker preserves shared edges between adjacent fields |
| 🔧 **Super-Resolution** | Auto-upscaling for coarser imagery (≥ 5 m/px) |
| 💾 **Memory-Safe** | Lightweight numpy queuing prevents Windows shared memory crashes |

---

## 🔄 Pipeline Stages

The Delineate Anything pipeline runs in 10 sequential stages:

```
Stage 0  →  Setup & Input Preparation
Stage 1  →  Initialization & Data Analysis      (DataAnalyser)
Stage 2  →  Execution Planning & Tiling         (ExecutionPlanner)
Stage 3  →  Model Inference (YOLO)              (DataLoaderCached + YOLO)
Stage 4  →  Per-Tile Post-Processing            (PostprocWorker)
Stage 5  →  Region Assembly & ID Mapping        (IDMapper / Union-Find)
Stage 6  →  Background Field Injection          (BackgroundLoader)
Stage 7  →  Polygonization (Raster → Vector)    (PolygonizationWorker)
Stage 8  →  Post-Delineation Merge              (postdelineation_merge)
Stage 9  →  Polygon Buffering                   (postdelineation_buffer)
Stage 10 →  Topology-Aware Simplification       (simplification/)
```

After delineation, two additional scripts complete the pipeline:

```
merge_outputs.py   →  CV-based quality merge of DA + SAM outputs
close_gaps.py      →  Watershed gap closing for seamless boundaries
```

> See [`Docs/PIPELINE.md`](Docs/PIPELINE.md) for the full stage-by-stage technical breakdown.

---

## 📁 Project Structure

```
Delineate-Anything/
│
├── delineate.py                  ← Main entry point (batch orchestration)
├── merge_outputs.py              ← Quality-aware polygon merge (CV-based)
├── close_gaps.py                 ← Watershed gap closing pipeline
├── shift.py                      ← Polygon coordinate shifter
├── simplify.py                   ← Standalone simplification entry point
│
├── conf_gujarat.yaml             ← Delineation config for Gujarat region
├── batch_gujarat.yaml            ← Batch processing configuration
├── simp_sample.yaml              ← Simplification configuration sample
├── requirements.txt              ← Python dependencies
│
├── methods/
│   └── main/
│       ├── inference.py          ← Core 10-stage pipeline orchestrator
│       ├── DataAnalyser.py       ← Input validation & normalization bounds
│       ├── ExecutionPlanner.py   ← Region/tile grid planner
│       ├── DataLoaderCached.py   ← Cached image loader with LCLU integration
│       ├── PostprocHandler.py    ← Shared memory & worker coordination
│       ├── PostprocWorker.py     ← Per-tile mask cleaning & composition
│       ├── UnitedWorker.py       ← Dual-mode multiprocessing worker
│       ├── PolygonizationWorker.py ← Raster-to-vector converter
│       ├── IDMapper.py           ← Union-Find for cross-tile field IDs
│       ├── BackgroundLoader.py   ← Background field injection
│       └── utils.py              ← GeoPackage creation utility
│
├── simplification/
│   ├── simplify.py               ← Topology-aware simplification engine
│   ├── ReadWorker.py             ← Polygon reader process
│   ├── SimplificationWorker.py   ← Vertex counting & Douglas-Peucker worker
│   └── WriteWorker.py            ← Output writer process
│
├── data/
│   ├── images/<Region>/          ← Input GeoTIFF satellite imagery
│   ├── masks/                    ← Optional LCLU land-use masks
│   └── delineated/
│       ├── <Region>.gpkg         ← Delineate Anything raw output
│       ├── SAM/                  ← SAM pipeline output files
│       └── merged/
│           └── merged_output.gpkg ← Final quality-merged output
│
├── Diagrams/                     ← System diagrams (Draw.io + images)
└── Docs/                         ← Full technical documentation
    ├── PIPELINE.md               ← End-to-end pipeline walkthrough
    ├── CODE_DOCUMENTATION.md     ← Function-by-function code reference
    ├── JOURNEY.md                ← Problem & solution log
    ├── Chapter10_Appendix.md     ← Appendix with code listings & glossary
    └── presentation_slides.md    ← Presentation slide outline
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11
- CUDA-capable NVIDIA GPU (recommended; 4 GB+ VRAM)
- GDAL (system-level installation)
- Conda (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Delineate-Anything.git
cd Delineate-Anything
```

### 2. Create & Activate Environment

```bash
conda create -n delineate python=3.11
conda activate delineate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA must be installed separately if not already present. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct install command for your CUDA version.

**Core Dependencies:**

| Package | Version | Role |
|---------|---------|------|
| `ultralytics` | 8.3.148 | YOLO model inference backbone |
| `rasterio` | 1.4.3 | GeoTIFF I/O & rasterization |
| `shapely` | 2.1.1 | Polygon geometry operations |
| `opencv-python` | 4.11.0.86 | Morphological ops & image processing |
| `huggingface-hub` | 0.32.4 | Auto-download of pretrained model weights |
| `gdal` | ≥ 3.6 | Low-level raster/vector access & reprojection |
| `geopandas` | ≥ 0.14 | Vector geospatial operations & GeoPackage I/O |
| `scipy` | ≥ 1.11 | Binary closing for gap mask generation |
| `scikit-image` | ≥ 0.21 | Watershed segmentation algorithm |
| `tqdm` | 4.67.1 | Progress bars |
| `numba` | 0.62.1 | JIT-compiled numerical kernels |

---

## 🚀 Quick Start

### Step 1 — Prepare Input Data

```bash
mkdir -p data/images/MyRegion
cp my_satellite_image.tif data/images/MyRegion/
```

> Optional: Place an LCLU mask at `data/masks/MyRegion.tif` to filter non-agricultural areas.

### Step 2 — Configure

Copy and edit the sample config files:

```bash
cp conf_gujarat.yaml conf_myregion.yaml
cp batch_gujarat.yaml batch_myregion.yaml
```

**Key settings to verify in `conf_myregion.yaml`:**

```yaml
data_loader:
  bands: [2, 3, 4]    # ← Must match your image's actual R,G,B band indices!
                       #   Use `gdalinfo your_image.tif` to check band order.

passes:
  - model_args:
      - name: large
        use_half: false  # ← Set to false if you get 0 detections (FP16 bug on some GPUs)
```

**Update `batch_myregion.yaml` to point to your region:**

```yaml
base_config: conf_myregion.yaml
data_root:   data/images
output_root: data/delineated
include:
  - MyRegion
```

### Step 3 — Run Delineation

```bash
python delineate.py -b batch_myregion.yaml
```

Model weights (~17.6 MB or ~119 MB) are downloaded from HuggingFace automatically on first run.

### Step 4 — (Optional) Run SAM & Merge

```bash
# Run the SAM pipeline (if configured)
python sam_pipeline.py

# Merge DA and SAM outputs using spectral quality comparison
python merge_outputs.py

# Fill inter-farm gaps using watershed algorithm
python close_gaps.py
```

### Step 5 — Outputs

```
data/delineated/
├── MyRegion.gpkg          ← Full-resolution delineated field polygons
└── MyRegion.simp.gpkg     ← Topology-simplified polygons (if enabled)

data/delineated/merged/
└── merged_output.gpkg     ← Final quality-merged output (DA + SAM)
```

Open any `.gpkg` file in **QGIS**, **ArcGIS**, or any GIS software to visualize and validate the results.

---

## 📝 Configuration Reference

### Delineation Config (`conf_*.yaml`)

```yaml
# ─── MODEL ───────────────────────────────────────────────────────────
model: ["large"]              # "small" (17.6 MB) or "large" (119 MB), or list for ensemble
method: main

# ─── SUPER-RESOLUTION ────────────────────────────────────────────────
super_resolution: null        # null = auto (1 for <5 m/px, 2 for ≥5 m/px)
treat_as_vrt: false           # true = merge multiple TIFFs into virtual raster

# ─── LCLU MASK ───────────────────────────────────────────────────────
mask_info:
  range: 24
  filter_classes: [1,10,11,12,23]  # Eliminate whole polygons on overlap
  clip_classes: [0,13,14]          # Clip polygon area on overlap

# ─── DATA LOADING ────────────────────────────────────────────────────
data_loader:
  bands: [3, 2, 1]            # Band indices (1-based GDAL): R, G, B
  nodata_value: [0, 0, 0]

# ─── TILING ──────────────────────────────────────────────────────────
execution_planner:
  region_width: 4096          # Pixels per region (reduce if out of RAM)
  region_height: 4096
  pixel_offset: [-1, -1]

# ─── PARALLELISM ─────────────────────────────────────────────────────
postprocess_limits:
  num_workers: [2, 2]         # Worker grid [rows, cols]
  queue_tiles_capacity: 32
  max_tiles_inflight: 64

# ─── INFERENCE PASSES ────────────────────────────────────────────────
passes:
  - batch_size: 16            # Tiles per GPU batch (reduce if out of VRAM)
    tile_size: null           # null = auto (512 for <5 m, 256 for ≥5 m)
    tile_step: 0.5            # Overlap ratio (0.5 = 50% overlap)
    model_args:
      - name: large
        minimal_confidence: 0.005
        use_half: true        # Set false on older GPUs if 0 detections

# ─── FILTERING & OUTPUT ──────────────────────────────────────────────
filtering_args:
  minimum_area_m2: 2500       # Discard polygons smaller than this (m²)
  minimum_hole_area_m2: 2500  # Remove interior holes smaller than this
  buffer_distance_m: null     # Negative = shrink (e.g. -1.5 for gaps)

# ─── SIMPLIFICATION ──────────────────────────────────────────────────
simplification_args:
  simplify: true
  epsilon_scale: 2            # Higher = more aggressive simplification
  num_workers: -1             # -1 = all CPUs
```

### Batch Config (`batch_*.yaml`)

```yaml
base_config: conf_sample.yaml   # Path to delineation config

data_root:   data/images        # Root folder with one subfolder per region
output_root: data/delineated    # Output folder for GeoPackages
temp_root:   data/temp          # Temporary working directory
mask_root:   data/masks         # LCLU mask folder
keep_temp:   false

include:                        # Only process these regions (null = all)
  - MyRegion
exclude: null
```

---

## 🛠 Utility Scripts

### `shift.py` — Geometry Coordinate Shift

Translates all polygon geometries by a fixed X/Y offset. Useful for correcting misregistered output:

```bash
python shift.py -i input.gpkg -o output.gpkg -s raster.tif -x 2 -y -1
```

| Flag | Description |
|------|-------------|
| `-i` | Input GeoPackage |
| `-o` | Output GeoPackage |
| `-s` | Reference raster (pixel size used as shift unit) |
| `-x` | Shift in X direction (pixels if `-s` given, else CRS units) |
| `-y` | Shift in Y direction |

### `simplify.py` — Standalone Simplification

Run topology-aware polygon simplification independently:

```bash
python simplify.py -c simp_sample.yaml
```

---

## 🔧 Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Wrong band order** | 0 detections from the model | Run `gdalinfo your_image.tif` to verify actual band content. NIR is often Band 1 in 4-band imagery. Set `bands: [2,3,4]` for RGBNIR. |
| **FP16 silent failure** | 0 detections even with `conf=0.0` | Set `use_half: false` in `conf_*.yaml`. Some GPU + model combinations silently produce empty outputs in FP16. |
| **Windows memory crash** | `RuntimeError: error 1455 (paging file too small)` | Already fixed in this fork — PyTorch results are converted to lightweight numpy dicts before multiprocessing queues. |
| **OpenCV int32 error** | `cv2.resize` assertion failure | Fixed: arrays are cast through `float32` with `INTER_NEAREST` before resize. |
| **Huge processing time** | Hours per small region | Increase `batch_size`, enable `use_half: true` (if supported), reduce `tile_step` overlap, use larger `region_width`. |
| **Jagged boundaries** | Staircase-like polygon edges | Enable simplification with `simplify: true` and adjust `epsilon_scale`. |
| **Missing fields in gaps** | Empty areas between regions | Increase `region_width`/`region_height`, or provide a `background_info` vector source. |
| **Out of VRAM** | CUDA out of memory error | Reduce `batch_size` to 1, or switch from `large` to `small` model. |

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [`Docs/PIPELINE.md`](Docs/PIPELINE.md) | Detailed end-to-end pipeline walkthrough (all 10 stages) |
| [`Docs/CODE_DOCUMENTATION.md`](Docs/CODE_DOCUMENTATION.md) | Function-by-function code reference for every module |
| [`Docs/JOURNEY.md`](Docs/JOURNEY.md) | Problem & solution log — every bug encountered and fixed |
| [`Docs/Chapter10_Appendix.md`](Docs/Chapter10_Appendix.md) | Code listings, glossary & algorithm parameter reference |
| [`Docs/delineation_config_guide.md`](Docs/delineation_config_guide.md) | Quick YAML configuration guide |

---

## ⚖️ License & Credits

This project is licensed under the [GNU General Public License v3.0](LICENSE).

### Original Model

- **Delineate Anything** model weights by [Mykola Lavreniuk](https://huggingface.co/MykolaL/DelineateAnything) — pretrained YOLO instance segmentation model for agricultural field boundaries.

### Frameworks & Libraries

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — inference backbone
- [Meta SAM](https://github.com/facebookresearch/segment-anything) — Segment Anything Model
- [GDAL / OGR](https://gdal.org) — geospatial raster & vector I/O
- [Rasterio](https://rasterio.readthedocs.io) — GeoTIFF access
- [Shapely](https://shapely.readthedocs.io) — polygon geometry operations
- [GeoPandas](https://geopandas.org) — vector data processing

### Organization

Developed as part of a geospatial AI research project at **BISAG-N** (Bhaskaracharya National Institute for Space Applications and Geo-informatics), Gandhinagar, Gujarat, India.

---

<div align="center">

*Made with ❤️ for agricultural boundary delineation at scale.*

</div>
