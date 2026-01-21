# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Don't generate a 'reference guide','QUICK REFERENCE', 'OVERVIEW', 'comprehensive summary document,' or any kind of docs md files until the user explicitly mentions them.

## Project Overview

Research project for Fresh Fruit Bunch (FFB) oil palm detection and ripeness classification using YOLO models. The project involves experiments with RGB, depth, and RGBD inputs for both localization (object detection) and ripeness classification tasks.

## Environment Setup

### Virtual Environment
```bash
# Activate virtual environment (REQUIRED for all operations)
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies
- Python 3.10+ (3.12 recommended)
- ultralytics (YOLO)
- opencv-python
- numpy, pandas, matplotlib, seaborn
- anylabeling (for manual annotation)
- **NEW**: torch, transformers (for Depth-Anything-V2)
- **NEW**: albumentations (for RGBD synced augmentation)

## Project Structure

```
Anylabel/
├── Dataset/                          # Raw dataset storage
│   ├── 28574489.zip                 # Original dataset (4.3GB)
│   ├── gohjinyu-oilpalm-ffb-dataset-d66eb99/  # Extracted dataset
│   │   ├── rgb_images/
│   │   ├── depth_maps/
│   │   └── point_clouds/
│   └── scripts/                     # Data collection utilities
│       ├── ffb_data_collection.py
│       ├── opencv_pc.py
│       └── stitching.py
├── Experiments/                      # Training experiments & scripts
│   ├── scripts/                     # All Python scripts for experiments
│   │   ├── Data Preparation:
│   │   │   ├── simple_eda.py
│   │   │   ├── split_localization_data.py
│   │   │   ├── prepare_depth_data.py
│   │   │   ├── convert_json_to_yolo.py
│   │   │   └── cleanup_ffb_localization_structure.py
│   │   ├── Training:
│   │   │   ├── train_a1_rgb.py
│   │   │   ├── train_a2_depth.py
│   │   │   ├── train_a3_rgbd.py              # NEW: RGB+Depth 4-channel
│   │   │   ├── train_a4a_synthetic_depth.py  # NEW: Synthetic depth only
│   │   │   ├── train_a4b_rgbd_synthetic.py   # NEW: RGB+Synthetic depth
│   │   │   ├── train_b1_classification.py
│   │   │   ├── train_b2_stage1_detector.py   # NEW: Two-stage Stage 1
│   │   │   ├── train_b2_stage2_classifier.py # NEW: Two-stage Stage 2
│   │   │   ├── train_ablation.py
│   │   │   └── train_scaling_adamw.py
│   │   ├── Data Generation:
│   │   │   ├── generate_synthetic_depth.py   # NEW: Depth-Anything-V2
│   │   │   ├── prepare_synthetic_depth_data.py
│   │   │   ├── extract_crops_b2.py           # NEW: Extract crops for B.2
│   │   │   └── custom_rgbd_dataset.py        # NEW: RGBD dataloader
│   │   ├── Evaluation:
│   │   │   ├── evaluate_all.py
│   │   │   ├── failure_analysis.py
│   │   │   ├── find_best_map.py
│   │   │   ├── inference_b2_twostage.py      # NEW: Two-stage inference
│   │   │   └── compare_real_vs_synthetic.py  # NEW: A.2 vs A.4a comparison
│   │   └── Kaggle Upload:
│   │       ├── build_uploadkaggle_depth_only.py
│   │       ├── build_uploadkaggle_rgbd_pairs.py
│   │       ├── build_uploadkaggle_synthetic_depth.py  # NEW: For A.4a
│   │       └── build_uploadkaggle_ripeness_*.py
│   ├── configs/                     # YOLO dataset configuration files
│   │   ├── ffb_localization.yaml
│   │   ├── ffb_localization_rgbd.yaml                # NEW: A.3 config
│   │   ├── ffb_localization_rgbd_synthetic.yaml      # NEW: A.4b config
│   │   ├── ffb_localization_depth_synthetic.yaml     # NEW: A.4a config
│   │   ├── ffb_ripeness_detect.yaml                  # NEW: B.2 Stage 1 config
│   │   └── ffb_localization_uploadkaggle.yaml
│   ├── datasets/                    # Processed datasets (train/val/test splits)
│   │   ├── ffb_localization/       # RGB dataset (YOLO format)
│   │   ├── ffb_localization_depth/ # Real depth dataset
│   │   ├── ffb_localization_depth_synthetic/     # NEW: Synthetic depth dataset (A.4a)
│   │   ├── depth_processed_rgb/    # 3-channel real depth (0-255)
│   │   ├── depth_synthetic_da2/    # NEW: 3-channel synthetic depth
│   │   ├── ffb_ripeness/           # Ripeness classification dataset
│   │   └── ffb_ripeness_twostage_crops/          # NEW: Crops for B.2 Stage 2
│   ├── UploadKaggle/               # Kaggle dataset packages (ZIP files)
│   ├── kaggleoutput/               # Training results from Kaggle
│   ├── eda_output/                 # EDA reports & visualizations
│   ├── labeling/                   # Manual annotation workspace
│   ├── notebooks/                   # NEW: Jupyter Notebooks for V3 experiments
│   │   ├── train_a3_rgbd_fix.ipynb
│   │   ├── generate_synthetic_depth.ipynb
│   │   ├── train_a4a_synthetic_depth.ipynb
│   │   ├── train_a4b_rgbd_synthetic.ipynb
│   │   └── train_b2_twostage.ipynb
│   ├── EXPERIMENT_GUIDE_V2.md      # Experiment requirements (Indonesian)
│   ├── EXPERIMENT_GUIDE_V3.md      # NEW: V3 experiments + notebooks guide
│   ├── README.md                   # Technical setup guide (English)
│   └── ablation_study_plan.md      # Ablation study results & analysis
└── Reports/
    └── FFB_Ultimate_Report/        # Final reports & artifacts
        ├── artifacts/              # Snapshot of experiment outputs
        ├── assets/                 # Images, plots, visualizations
        ├── result.md               # Main consolidated report
        └── README.md
```

## Jupyter Notebooks (V3 Experiments)

**NEW**: All V3 experiments now have Jupyter Notebooks for easier execution:

| Notebook | Experiment | Description |
|:---------|:-----------|:------------|
| `train_a3_rgbd_fix.ipynb` | A.3 Fix | RGB+Depth dengan augmentasi tersingkron |
| `generate_synthetic_depth.ipynb` | Data Prep | Generate synthetic depth (Depth-Anything-V2) |
| `train_a4a_synthetic_depth.ipynb` | A.4a | Synthetic depth only (3-channel) |
| `train_a4b_rgbd_synthetic.ipynb` | A.4b | RGB + Synthetic depth (4-channel) |
| `train_b2_twostage.ipynb` | B.2 | Two-stage classification (full pipeline) |

**Key Features**:
- Auto-detect environment (Kaggle vs Local)
- Auto-save results to `kaggleoutput/*.txt`
- Training 2 seeds (42, 123) for reproducibility
- Comprehensive evaluation with comparison tables

**Usage - Local**:
```bash
cd Experiments
jupyter lab
# or
jupyter notebook
# Navigate to notebooks/ and run
```

**Usage - Kaggle**:
1. Copy-paste notebook content to new Kaggle notebook
2. Add required datasets
3. Enable GPU accelerator
4. Run all cells

See `EXPERIMENT_GUIDE_V3.md` → "Jupyter Notebooks - Panduan Penggunaan" for complete guide.

---

## Common Commands

### Data Preparation
```bash
cd Experiments\scripts

# Step 1: Exploratory Data Analysis
python simple_eda.py

# Step 2: Split dataset (70:20:10 train/val/test)
python split_localization_data.py

# Step 3: Process depth maps (0.6m-6m → 0-255, 1→3 channel)
python prepare_depth_data.py

# NEW: Generate synthetic depth using Depth-Anything-V2
python generate_synthetic_depth.py         # Takes 20-30 min on GPU
python prepare_synthetic_depth_data.py     # Organize synthetic depth
python prepare_depth_data.py

# Convert JSON annotations to YOLO format
python convert_json_to_yolo.py

# Cleanup dataset structure
python cleanup_ffb_localization_structure.py
```

### Training Experiments

**Important**: All training scripts run from `Experiments/` directory.

```bash
cd Experiments\scripts

# A.1: RGB Baseline (2 runs with seeds 42, 123)
python train_a1_rgb.py

# A.2: Depth Only (3-channel depth)
python train_a2_depth.py

# NEW A.3: RGB+Depth (4-channel, with augmentation fix)
python train_a3_rgbd.py

# NEW A.4a: Synthetic Depth Only
python train_a4a_synthetic_depth.py

# NEW A.4b: RGB+Synthetic Depth (4-channel)
python train_a4b_rgbd_synthetic.py

# B.1: Ripeness Detection (2-class end-to-end)
python train_b1_classification.py

# NEW B.2: Two-Stage Ripeness Classification
python train_b2_stage1_detector.py    # Stage 1: Detect FFBs
python extract_crops_b2.py             # Extract crops from detections
python train_b2_stage2_classifier.py   # Stage 2: Classify crops
python inference_b2_twostage.py        # Run end-to-end pipeline

# Ablation Studies (modify config inside the script)
python train_ablation.py
python train_scaling_adamw.py
```

### Evaluation
```bash
cd Experiments\scripts

# Evaluate all experiments
python evaluate_all.py

# Failure analysis (FP/FN visualization)
python failure_analysis.py

# Find best model by mAP
python find_best_map.py
```

### Manual YOLO Validation
```bash
cd Experiments

# Validate on test set (use absolute or relative path from Experiments/)
yolo detect val model=runs/detect/exp_name/weights/best.pt data=configs/ffb_localization.yaml split=test
```

### Dataset Packaging (for Kaggle upload)
```bash
cd Experiments\scripts

# Build localization datasets
python build_uploadkaggle_depth_only.py
python build_uploadkaggle_rgbd_pairs.py

# Build ripeness datasets
python build_uploadkaggle_ripeness_detection.py
python build_uploadkaggle_ripeness_classification.py
python build_uploadkaggle_ripeness_crops.py
```

## Experiment Configuration

### Experiment Types
- **A.1**: RGB-only localization (1 class: fresh_fruit_bunch)
- **A.2**: Depth-only localization (real depth, 3-channel)
- **A.3**: RGB+Depth fusion (4-channel, **FIXED** with proper augmentation)
- **A.4a**: Synthetic depth-only (Depth-Anything-V2, 3-channel)
- **A.4b**: RGB+Synthetic depth fusion (4-channel)
- **B.1**: Ripeness detection (2-class end-to-end: ripe, unripe)
- **B.2**: Two-stage ripeness (Detect → Crop → Classify)

### Training Standards
- **Model**: YOLOv11n (Nano) baseline, YOLOv11s (Small) for ablation
- **Seeds**: Always use 42 and 123 for reproducibility (2 runs per experiment)
- **Epochs**: 50 (baseline), 300 (ablation studies with early stopping)
- **Batch Size**: 16 (detection), 32 (classification)
- **Image Size**: 640x640 (detection), 224x224 (classification)
- **Device**: CUDA (GPU) when available, CPU fallback
- **Depth Range**: 0.6m - 6.0m normalized to 0-255

### Evaluation Metrics
- **Detection**: mAP50, mAP50-95
- **Classification**: Top-1 Accuracy
- **Always report**: mean and std dev from 2 runs (seeds 42 & 123)
- **Always evaluate on**: test split (not validation)

## Architecture Notes

### Data Flow
1. Raw dataset (ZIP) → Extracted to `Dataset/gohjinyu-oilpalm-ffb-dataset-d66eb99/`
2. Manual annotation using AnyLabeling → YOLO format labels
3. Scripts in `Experiments/scripts/` → Process & split into `Experiments/datasets/`
4. Training scripts → Results to `runs/detect/` or `runs/classify/`
5. Evaluation scripts → Generate reports & analysis
6. Final artifacts → Archived in `Reports/FFB_Ultimate_Report/`

### YOLO Dataset Config Pattern
Dataset YAML files are stored in `Experiments/configs/` (e.g., `ffb_localization.yaml`):
```yaml
path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_localization
train: images/train
val: images/val
test: images/test
nc: 1  # number of classes
names: ['fresh_fruit_bunch']
```

**Critical**:
- Always use absolute paths in YAML configs for YOLO compatibility
- Reference configs from Experiments/ as: `data=configs/ffb_localization.yaml`

### Depth Processing Pipeline
1. Raw depth maps (`.png` in mm, 0-65535 range)
2. Filter to valid range: 600-6000mm (0.6m-6m)
3. Normalize to 0-255 using min-max scaling
4. Replicate single channel to 3 channels (R=G=B) for YOLO compatibility
5. Save as 3-channel PNG in `depth_processed_rgb/`

### Training Output Structure
```
runs/detect/exp_name/
├── weights/
│   ├── best.pt          # Best model checkpoint
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Training metrics per epoch
├── args.yaml            # Training arguments
├── F1_curve.png         # F1 score curve
├── PR_curve.png         # Precision-Recall curve
├── confusion_matrix.png
└── val_batch*_*.jpg     # Validation predictions
```

## Key Guidelines

### File Paths
- **Always use absolute paths** in YOLO dataset YAML configs
- Scripts expect to be run from `Experiments/` directory
- Dataset paths use Windows-style backslashes: `D:\Work\Assisten Dosen\Anylabel\...`

### Experiment Protocol
1. Always verify dataset split exists before training
2. Run 2 training runs per experiment (seeds 42, 123)
3. Evaluate on **test set** (not validation set)
4. Archive results to Kaggle/Reports for reproducibility
5. Document failures (FP/FN) with visual examples

### Code Quality
- Direct, technical communication (no emojis, no fluff)
- Clean up temporary files immediately after failures
- Use todo list for multi-step tasks
- Match existing code patterns and style
- No debug artifacts in final code

### Kaggle Integration
- Experiments often trained on Kaggle for GPU access
- Results downloaded as ZIP archives to `Experiments/kaggleoutput/`
- Use `build_uploadkaggle_*.py` scripts to package datasets for Kaggle upload
- Training scripts designed to work both locally and on Kaggle (adjust paths accordingly)

## Reference Documents

- `EXPERIMENT_GUIDE_V2.md`: Concise experiment requirements (Indonesian)
- `Experiments/README.md`: Detailed technical setup guide (English)
- `ablation_study_plan.md`: Ablation study results & analysis
- `.factory/instructions.md`: Project context & troubleshooting protocols
- `Reports/FFB_Ultimate_Report/result.md`: Final consolidated report

## Troubleshooting

### Common Issues
- **PowerShell script execution**: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **CUDA OOM**: Reduce batch size in training script (16 → 8 → 4)
- **No labels found**: Verify annotations exist in `Experiments/datasets/*/labels/train/`
- **Python not found**: Use full path to venv python: `.\venv\Scripts\python.exe`

### Dataset Validation
Before training, verify:
```bash
# Check dataset structure
ls Experiments\datasets\ffb_localization\images\train
ls Experiments\datasets\ffb_localization\labels\train

# Count images vs labels (should match)
(ls Experiments\datasets\ffb_localization\images\train\*.png).Count
(ls Experiments\datasets\ffb_localization\labels\train\*.txt).Count
```

## Current Status

- Dataset: Extracted and split (train/val/test)
- Annotations: Manual labeling completed (300+ images)
- Baseline experiments: A.1 (RGB), A.2 (Depth), B.1 (Detection) completed
- **NEW Experiments**: A.3 (RGBD fixed), A.4a/A.4b (Synthetic depth), B.2 (Two-stage)
- Ablation studies: YOLOv11s vs YOLOv11n, SGD vs AdamW, 50e vs 300e completed
- Best model: YOLOv11s + SGD + 300 epochs (mAP50-95: 0.433 on test set)
- Reports: Consolidated in `Reports/FFB_Ultimate_Report/`

## New Additions (2026-01-21)

### A.3 RGBD Fix
- **Issue**: Previous A.3 had inconsistent augmentation (HSV disabled)
- **Fix**: Created `custom_rgbd_dataset.py` with synced augmentation
- **Options**: Option A (HSV on RGB), Option B (no HSV for fair comparison)
- **Files**: `train_a3_rgbd.py`, `custom_rgbd_dataset.py`, `ffb_localization_rgbd.yaml`

### A.4 Synthetic Depth
- **Model**: Depth-Anything-V2-Large from HuggingFace
- **Purpose**: Test if synthetic depth can replace real depth sensor
- **Variants**:
  - A.4a: Synthetic depth only (compare with A.2)
  - A.4b: RGB + Synthetic depth (compare with A.3)
- **Files**: `generate_synthetic_depth.py`, `prepare_synthetic_depth_data.py`, 
  `train_a4a_synthetic_depth.py`, `train_a4b_rgbd_synthetic.py`
- **Comparison**: `compare_real_vs_synthetic.py` generates analysis

### B.2 Two-Stage Classification
- **Approach**: Detect → Crop → Classify (vs B.1 end-to-end)
- **Hypothesis**: Cropping improves classification accuracy
- **Pipeline**:
  1. Stage 1: Train detector for ripe/unripe FFBs
  2. Extract crops with 10% margin
  3. Stage 2: Train specialized classifier on crops
  4. Inference: Run full pipeline end-to-end
- **Files**: `train_b2_stage1_detector.py`, `extract_crops_b2.py`, 
  `train_b2_stage2_classifier.py`, `inference_b2_twostage.py`
