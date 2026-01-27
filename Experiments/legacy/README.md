# Legacy Archive - Experiments

This folder contains old/deprecated code, documents, and notebooks that are no longer actively maintained.

## Why Files Are Here

### Notebooks (`legacy/notebooks/`)

- **`train_a3_rgbd_fix.ipynb`** - Old version of A.3 experiment
  - Replaced by: `notebooks/train_a3_rgbd.ipynb`
  - Reason: Standardized to 5 seeds [42, 123, 456, 789, 101] with default YOLO parameters
  - Created: 2026-01-22

### Scripts (`legacy/scripts/`)

- **`train_ablation.py`** - Old ablation study training script
  - Reason: Ablation studies (YOLOv11s, SGD, 300 epochs) completed and archived
  - Results documented in: `legacy/docs/ablation_study_plan.md`
  - Created: 2026-01-20

- **`train_scaling_adamw.py`** - Scaling & AdamW optimization study
  - Reason: Part of completed ablation study
  - Superseded by: `legacy/docs/ablation_study_plan.md`
  - Created: 2026-01-21

- **`train_b2_stage1_detector.py`**, **`train_b2_stage2_classifier.py`**, **`inference_b2_twostage.py`**, **`extract_crops_b2.py`**
  - Reason: B.2 Two-Stage training/inference replaced by notebook approach
  - Replaced by: `notebooks/train_b2_twostage.ipynb`
  - Approach: Notebooks provide better interactivity for Kaggle execution
  - Created: 2026-01-21, 2026-01-22

### Documentation (`legacy/docs/`)

- **`ablation_study_plan.md`** - Detailed ablation study results and analysis
  - Covers: YOLOv11n vs YOLOv11s, SGD vs AdamW, 50 epochs vs 300 epochs
  - Best model: YOLOv11s + SGD + 300 epochs (mAP50-95: 0.433 on test set)
  - Status: Completed and archived
  - Created: 2026-01-20

## Active Code Location

For current experiments, use:

- **Notebooks**: `Experiments/notebooks/`
  - V3 experiments with 5 seeds
  - Kaggle-ready format

- **Scripts**: `Experiments/scripts/0X_*/`
  - Data preparation: `00_data_prep/`
  - Training: `01_training/`
  - Data generation: `02_data_generation/`
  - Evaluation: `03_evaluation/`
  - Kaggle upload: `04_kaggle_upload/`

## Reference & Reproducibility

If you need to reproduce ablation study results, see `ablation_study_plan.md` for details.
For understanding differences between old and new approaches, consult the main project report:
`Reports/FFB_Ultimate_Report/result.md`

---
*Archive created: 2026-01-27*
