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
│   │   │   ├── train_b1_classification.py
│   │   │   ├── train_ablation.py
│   │   │   └── train_scaling_adamw.py
│   │   ├── Evaluation:
│   │   │   ├── evaluate_all.py
│   │   │   ├── failure_analysis.py
│   │   │   └── find_best_map.py
│   │   └── Kaggle Upload:
│   │       ├── build_uploadkaggle_depth_only.py
│   │       ├── build_uploadkaggle_rgbd_pairs.py
│   │       ├── build_uploadkaggle_ripeness_*.py
│   ├── configs/                     # YOLO dataset configuration files
│   │   ├── ffb_localization.yaml
│   │   └── ffb_localization_uploadkaggle.yaml
│   ├── datasets/                    # Processed datasets (train/val/test splits)
│   │   ├── ffb_localization/       # RGB dataset (YOLO format)
│   │   ├── ffb_localization_depth/ # Depth dataset
│   │   ├── depth_processed_rgb/    # 3-channel depth (0-255)
│   │   └── ffb_ripeness/           # Ripeness classification dataset
│   ├── UploadKaggle/               # Kaggle dataset packages (ZIP files)
│   ├── kaggleoutput/               # Training results from Kaggle
│   ├── eda_output/                 # EDA reports & visualizations
│   ├── labeling/                   # Manual annotation workspace
│   ├── EXPERIMENT_GUIDE_V2.md      # Experiment requirements (Indonesian)
│   ├── README.md                   # Technical setup guide (English)
│   └── ablation_study_plan.md      # Ablation study results & analysis
└── Reports/
    └── FFB_Ultimate_Report/        # Final reports & artifacts
        ├── artifacts/              # Snapshot of experiment outputs
        ├── assets/                 # Images, plots, visualizations
        ├── result.md               # Main consolidated report
        └── README.md
```

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

# B.1: Ripeness Classification
python train_b1_classification.py

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
- **A.2**: Depth-only localization (depth normalized to 3-channel RGB)
- **A.3**: RGB+Depth fusion (4-channel input, requires model modification)
- **B.1**: Ripeness detection/classification (2 classes: ripe, unripe)

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
- Baseline experiments: A.1 (RGB), A.2 (Depth), B.1 (Classification) completed
- Ablation studies: YOLOv11s vs YOLOv11n, SGD vs AdamW, 50e vs 300e completed
- Best model: YOLOv11s + SGD + 300 epochs (mAP50-95: 0.433 on test set)
- Reports: Consolidated in `Reports/FFB_Ultimate_Report/`
