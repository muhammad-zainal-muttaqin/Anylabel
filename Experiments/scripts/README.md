# Scripts Organization Guide

All scripts are organized by category in numbered subdirectories for easy navigation.

## Directory Structure

### `00_data_prep/` - Data Preparation & Preprocessing
Scripts for dataset preparation, splitting, and preprocessing.

| Script | Purpose |
|--------|---------|
| `simple_eda.py` | Exploratory Data Analysis (EDA) of raw dataset |
| `split_localization_data.py` | Split dataset into train/val/test (70:20:10) |
| `convert_json_to_yolo.py` | Convert JSON annotations to YOLO format |
| `prepare_depth_data.py` | Process real depth maps (normalize 0-255, convert 1→3 channel) |
| `prepare_synthetic_depth_data.py` | Organize synthetic depth data into train/val/test splits |
| `cleanup_ffb_localization_structure.py` | Fix dataset directory structure if needed |

**Typical Usage Order**:
```bash
python 00_data_prep/simple_eda.py                    # Understand data
python 00_data_prep/split_localization_data.py       # Create splits
python 00_data_prep/convert_json_to_yolo.py          # Convert format
python 00_data_prep/prepare_depth_data.py            # Process depth
```

---

### `01_training/` - Training Scripts (V3 Experiments)
Active training scripts for FFB detection and ripeness classification.

| Script | Experiment | Input | Classes |
|--------|-----------|-------|---------|
| `train_a1_rgb.py` | A.1 | RGB | 1 (FFB) |
| `train_a2_depth.py` | A.2 | Real Depth | 1 (FFB) |
| `train_a3_rgbd.py` | A.3 | RGB+Depth | 1 (FFB) |
| `train_a4a_synthetic_depth.py` | A.4a | Synthetic Depth | 1 (FFB) |
| `train_a4b_rgbd_synthetic.py` | A.4b | RGB+Synth Depth | 1 (FFB) |
| `train_b1_classification.py` | B.1 | RGB | 2 (Ripe/Unripe) |

**Note**: For B.2 Two-Stage, use notebook: `notebooks/train_b2_twostage.ipynb`

**Typical Usage**:
```bash
python 01_training/train_a1_rgb.py
python 01_training/train_a2_depth.py
python 01_training/train_a3_rgbd.py
# ... etc
```

All scripts run from `Experiments/` directory.

---

### `02_data_generation/` - Data Generation & Processing
Scripts for generating synthetic depth, creating fused datasets, and comparisons.

| Script | Purpose |
|--------|---------|
| `generate_synthetic_depth.py` | Generate synthetic depth using Depth-Anything-V2 |
| `prepare_synthetic_depth_data.py` | *(Deprecated - use 00_data_prep version)* |
| `custom_rgbd_dataset.py` | Custom RGBD dataloader with synchronized augmentation |
| `compare_real_vs_synthetic.py` | Compare A.2 (real) vs A.4a (synthetic) depth performance |

**Typical Usage**:
```bash
python 02_data_generation/generate_synthetic_depth.py  # ~20-30 min on GPU
python 02_data_generation/compare_real_vs_synthetic.py
```

---

### `03_evaluation/` - Evaluation & Analysis
Scripts for model evaluation, failure analysis, and metric extraction.

| Script | Purpose |
|--------|---------|
| `evaluate_all.py` | Evaluate all trained models on test set |
| `failure_analysis.py` | Visualize false positives and false negatives |
| `find_best_map.py` | Extract best model by mAP from all runs |

**Typical Usage**:
```bash
python 03_evaluation/evaluate_all.py
python 03_evaluation/failure_analysis.py
python 03_evaluation/find_best_map.py
```

---

### `04_kaggle_upload/` - Kaggle Dataset Packaging
Scripts for building and uploading datasets to Kaggle.

| Script | Purpose |
|--------|---------|
| `build_uploadkaggle_depth_only.py` | Package real depth dataset (A.2) |
| `build_uploadkaggle_rgbd_pairs.py` | Package RGBD pairs dataset (A.3) |
| `build_uploadkaggle_synthetic_depth.py` | Package synthetic depth dataset (A.4a/A.4b) |
| `build_uploadkaggle_ripeness_detection.py` | Package ripeness detection dataset (B.1) |
| `build_uploadkaggle_ripeness_classification.py` | Package ripeness classification dataset |
| `build_uploadkaggle_ripeness_crops.py` | Package ripeness crops dataset (B.2 Stage2) |
| `run_eda.bat` | Batch script to run EDA and show results |

**Note**: These scripts are optional - only needed if re-uploading to Kaggle.

---

## Quick Reference

### To run a complete experiment pipeline:
```bash
# From Experiments/ directory
python scripts/01_training/train_a1_rgb.py      # Train A.1
python scripts/03_evaluation/evaluate_all.py    # Evaluate
python scripts/03_evaluation/failure_analysis.py # Analyze failures
```

### To process a new dataset:
```bash
python scripts/00_data_prep/simple_eda.py
python scripts/00_data_prep/split_localization_data.py
python scripts/00_data_prep/convert_json_to_yolo.py
```

### To generate synthetic depth:
```bash
python scripts/02_data_generation/generate_synthetic_depth.py
python scripts/00_data_prep/prepare_synthetic_depth_data.py
```

---

## For Jupyter Notebooks

Most experiments now have Jupyter notebook versions for interactive execution:
- `notebooks/train_a*.ipynb` - Detection experiments
- `notebooks/train_b*.ipynb` - Classification experiments
- `notebooks/generate_synthetic_depth.ipynb` - Synthetic depth generation

These are preferred for **Kaggle execution** (better interactivity, auto-save output).

---

*Documentation updated: 2026-01-27*
