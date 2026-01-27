# Experiments Folder - Structure Guide (2026-01-27)

This document outlines the reorganized folder structure for clarity and maintainability.

## Directory Tree

```
Experiments/
│
├── 📔 NOTEBOOKS (V3 Experiments - Kaggle Ready)
│   ├── notebooks/
│   │   ├── generate_synthetic_depth.ipynb     # Data generation
│   │   ├── train_a1_rgb.ipynb                 # A.1: RGB only
│   │   ├── train_a2_depth.ipynb               # A.2: Real depth
│   │   ├── train_a3_rgbd.ipynb                # A.3: RGB+Real depth
│   │   ├── train_a4a_synthetic_depth.ipynb    # A.4a: Synthetic depth
│   │   ├── train_a4b_rgbd_synthetic.ipynb     # A.4b: RGB+Synthetic depth
│   │   ├── train_b1_ripeness.ipynb            # B.1: Ripeness detection
│   │   ├── train_b2_twostage.ipynb            # B.2: Two-stage pipeline
│   │   └── README.md                          # Notebooks guide
│   │
│   └── configs/
│       ├── ffb_localization.yaml
│       ├── ffb_localization_rgbd.yaml
│       ├── ffb_localization_depth_synthetic.yaml
│       └── ... (YOLO dataset configs)
│
├── 🐍 SCRIPTS (Organized by Category)
│   ├── scripts/
│   │   ├── 00_data_prep/
│   │   │   ├── simple_eda.py
│   │   │   ├── split_localization_data.py
│   │   │   ├── convert_json_to_yolo.py
│   │   │   ├── prepare_depth_data.py
│   │   │   ├── prepare_synthetic_depth_data.py
│   │   │   └── cleanup_ffb_localization_structure.py
│   │   │
│   │   ├── 01_training/
│   │   │   ├── train_a1_rgb.py
│   │   │   ├── train_a2_depth.py
│   │   │   ├── train_a3_rgbd.py
│   │   │   ├── train_a4a_synthetic_depth.py
│   │   │   ├── train_a4b_rgbd_synthetic.py
│   │   │   └── train_b1_classification.py
│   │   │
│   │   ├── 02_data_generation/
│   │   │   ├── generate_synthetic_depth.py
│   │   │   ├── custom_rgbd_dataset.py
│   │   │   └── compare_real_vs_synthetic.py
│   │   │
│   │   ├── 03_evaluation/
│   │   │   ├── evaluate_all.py
│   │   │   ├── failure_analysis.py
│   │   │   └── find_best_map.py
│   │   │
│   │   ├── 04_kaggle_upload/
│   │   │   ├── build_uploadkaggle_depth_only.py
│   │   │   ├── build_uploadkaggle_rgbd_pairs.py
│   │   │   ├── build_uploadkaggle_synthetic_depth.py
│   │   │   ├── build_uploadkaggle_ripeness_detection.py
│   │   │   ├── build_uploadkaggle_ripeness_classification.py
│   │   │   ├── build_uploadkaggle_ripeness_crops.py
│   │   │   └── run_eda.bat
│   │   │
│   │   └── README.md                          # Scripts guide
│   │
│   └── _archive/ (empty, for future cleanup)
│
├── 📦 DATASETS (Processed & Organized)
│   ├── datasets/
│   │   ├── ffb_localization/                 # A.1: RGB baseline
│   │   ├── ffb_localization_depth/           # A.2: Real depth
│   │   ├── ffb_localization_rgbd/            # A.3: RGB+Real depth
│   │   ├── depth_processed_rgb/              # Real depth (normalized)
│   │   ├── ffb_synthetic_depth_yolo.zip      # A.4a: Synthetic depth
│   │   ├── ffb_ripeness/                     # B.1/B.2: Ripeness data
│   │   ├── ffb_ripeness_twostage_crops/      # B.2: Extracted crops
│   │   └── labels/                           # Old labels (to cleanup)
│
├── 📊 RESULTS & OUTPUT
│   ├── kaggleoutput/                         # Training results
│   ├── eda_output/                           # EDA reports
│   ├── runs/                                 # YOLO training runs (auto-generated)
│   │   ├── detect/
│   │   └── classify/
│   │
│   └── UploadKaggle/                         # Packaged datasets for Kaggle
│
├── 🏷️ LABELING
│   ├── labeling/                             # Manual annotation workspace
│
├── 📚 LEGACY & ARCHIVED
│   └── legacy/
│       ├── notebooks/
│       │   └── train_a3_rgbd_fix.ipynb       # Old A.3 version
│       │
│       ├── scripts/
│       │   ├── train_ablation.py             # Ablation study (archived)
│       │   ├── train_scaling_adamw.py        # AdamW study (archived)
│       │   ├── train_b2_stage1_detector.py   # Old B.2 stage 1
│       │   ├── train_b2_stage2_classifier.py # Old B.2 stage 2
│       │   ├── inference_b2_twostage.py      # Old B.2 inference
│       │   └── extract_crops_b2.py           # Old B.2 crop extraction
│       │
│       ├── docs/
│       │   └── ablation_study_plan.md        # Ablation results
│       │
│       └── README.md                         # Legacy guide
│
└── 📖 DOCUMENTATION
    ├── README.md                             # Setup & overview
    ├── EXPERIMENT_GUIDE_V2.md                # Experiment requirements (Indonesian)
    ├── EXPERIMENT_GUIDE_V3.md                # V3 experiments guide
    ├── STRUCTURE_GUIDE.md                    # This file
    └── README (root)                         # CLAUDE.md
```

## Quick Navigation

### 🚀 To Run Experiments

**Option 1: Jupyter Notebooks (Recommended for Kaggle)**
```bash
jupyter lab notebooks/
# Run notebooks/train_a1_rgb.ipynb, etc.
```

**Option 2: Python Scripts (Local)**
```bash
python scripts/01_training/train_a1_rgb.py
python scripts/01_training/train_a2_depth.py
# ... etc
```

### 🔧 To Prepare Data
```bash
python scripts/00_data_prep/simple_eda.py
python scripts/00_data_prep/split_localization_data.py
python scripts/00_data_prep/prepare_depth_data.py
```

### 🤖 To Generate Synthetic Depth
```bash
python scripts/02_data_generation/generate_synthetic_depth.py
```

### 📊 To Evaluate Models
```bash
python scripts/03_evaluation/evaluate_all.py
python scripts/03_evaluation/failure_analysis.py
```

## What's New in This Reorganization

### ✨ Changes Made

1. **Notebooks**
   - Consolidated all V3 notebooks in `notebooks/` folder
   - Removed old notebook `train_a3_rgbd_fix.ipynb` → moved to `legacy/notebooks/`
   - Added `README.md` with usage guide

2. **Scripts**
   - Reorganized 25 scripts into **5 categories**:
     - `00_data_prep/` - Data preparation (6 scripts)
     - `01_training/` - Training V3 (6 scripts)
     - `02_data_generation/` - Data generation (3 scripts)
     - `03_evaluation/` - Evaluation & analysis (3 scripts)
     - `04_kaggle_upload/` - Kaggle utilities (7 scripts)
   - Added `README.md` with category descriptions

3. **Legacy**
   - Created `legacy/` folder for archived code:
     - Old notebooks (train_a3_rgbd_fix.ipynb)
     - Ablation study scripts (train_ablation.py, train_scaling_adamw.py)
     - Old B.2 scripts (stage1/2, inference, crop extraction)
   - Added `legacy/README.md` explaining what's archived and why

4. **Documentation**
   - `notebooks/README.md` - Jupyter notebooks guide
   - `scripts/README.md` - Python scripts reference
   - `legacy/README.md` - Archived code explanation
   - `STRUCTURE_GUIDE.md` - This file (overall layout)

## Active vs. Legacy

### 🟢 Active (Use These)
- `notebooks/train_a*.ipynb` - V3 experiments
- `notebooks/train_b*.ipynb` - Ripeness experiments
- `scripts/0X_*/` - All categorized scripts
- `datasets/ffb_*/ ` - Dataset folders

### 🔴 Legacy (Reference Only)
- `legacy/notebooks/train_a3_rgbd_fix.ipynb`
- `legacy/scripts/train_ablation.py`
- `legacy/scripts/train_scaling_adamw.py`
- `legacy/scripts/train_b2_stage*.py`
- `legacy/docs/ablation_study_plan.md`

## File Organization Rules

1. **Notebooks**: One notebook per experiment
2. **Scripts**: Grouped by functionality in numbered directories
3. **Legacy**: Anything > 2 weeks old or superseded by newer approach
4. **Datasets**: Organized by experiment (A.1, A.2, etc.)
5. **Results**: Auto-generated in `kaggleoutput/` and `runs/`

## Moving Forward

### Recommended Workflow
1. Check `notebooks/README.md` to understand available experiments
2. Run notebooks directly (Kaggle/JupyterLab) or scripts
3. Check `scripts/README.md` for script-specific guidance
4. Review `legacy/` only if reproducing old experiments
5. See main `Reports/FFB_Ultimate_Report/result.md` for results

### Adding New Experiments
1. Create notebook in `notebooks/train_xx_name.ipynb`
2. Or add script to appropriate `scripts/0X_*/` directory
3. Update relevant README.md files
4. Never pollute root directories - use subdirectories

---

## Summary of Changes

| What | Before | After | Status |
|------|--------|-------|--------|
| Notebooks | Scattered + old versions | Clean 8 active + legacy | ✅ Done |
| Scripts | 25 files in root `scripts/` | 25 files organized in 5 categories | ✅ Done |
| Legacy | Nowhere | `legacy/` with docs | ✅ Done |
| Documentation | Minimal | README.md in each folder | ✅ Done |

---

*Reorganization completed: 2026-01-27*

For detailed information, see:
- `notebooks/README.md` - Jupyter guide
- `scripts/README.md` - Python scripts guide
- `legacy/README.md` - Archived code explanation
- `CLAUDE.md` - Project instructions (main repo root)
