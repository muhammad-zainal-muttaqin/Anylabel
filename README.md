# FFB Detection & Ripeness Classification

Research project untuk deteksi Fresh Fruit Bunch (FFB) kelapa sawit dan klasifikasi kematangan menggunakan YOLO models dengan input RGB, Depth, dan RGBD.

## Prerequisites

- **Python 3.10+** (disarankan Python 3.12)
- **Windows 10/11**
- **NVIDIA GPU** (opsional, untuk training lebih cepat)

## Quick Start

### 1. Setup Environment

```powershell
# Aktivasi virtual environment
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
cd Experiments\scripts

# Run EDA
python simple_eda.py

# Split dataset (70:20:10)
python split_localization_data.py

# Process depth maps
python prepare_depth_data.py
```

### 3. Run Experiments

```bash
# RGB baseline
python train_a1_rgb.py

# Depth only
python train_a2_depth.py

# Ripeness classification
python train_b1_classification.py
```

## Project Structure

```
Anylabel/
├── Dataset/                 # Raw dataset & collection scripts
├── Experiments/            # Training experiments
│   ├── scripts/           # All Python scripts
│   ├── configs/           # YOLO config YAML files
│   ├── datasets/          # Processed train/val/test splits
│   ├── UploadKaggle/      # Kaggle dataset packages
│   ├── kaggleoutput/      # Training results from Kaggle
│   └── eda_output/        # EDA reports
├── Reports/               # Final reports & artifacts
├── CLAUDE.md              # Guide for Claude Code
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

Lihat [CLAUDE.md](CLAUDE.md) untuk dokumentasi lengkap.

## Key Commands

```bash
# Activate venv
.\venv\Scripts\Activate

# Data preparation
cd Experiments\scripts
python simple_eda.py
python split_localization_data.py
python prepare_depth_data.py

# Training
python train_a1_rgb.py              # RGB baseline
python train_a2_depth.py            # Depth only
python train_b1_classification.py  # Ripeness classification

# Evaluation
python evaluate_all.py
python failure_analysis.py
python find_best_map.py

# YOLO manual validation
cd Experiments
yolo detect val model=runs/detect/exp_name/weights/best.pt data=configs/ffb_localization.yaml split=test
```

## Experiments

- **A.1**: RGB-only detection (1 class: FFB)
- **A.2**: Depth-only detection (normalized to 3-channel)
- **A.3**: RGB+Depth fusion (4-channel input)
- **B.1**: Ripeness classification (2 classes: ripe/unripe)
- **Ablation Studies**: Model size, optimizer, training duration

## Results

Best model: **YOLOv11s + SGD + 300 epochs**
- mAP50: 0.875
- mAP50-95: 0.433

Lihat [Reports/FFB_Ultimate_Report/result.md](Reports/FFB_Ultimate_Report/result.md) untuk detail lengkap.

## Troubleshooting

### PowerShell Execution Error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### CUDA Out of Memory
Edit training script, reduce batch size (16 → 8 → 4)

### No Labels Found
Verify annotations exist in `Experiments/datasets/*/labels/train/`

## Documentation

- [CLAUDE.md](CLAUDE.md) - Comprehensive project documentation
- [Experiments/README.md](Experiments/README.md) - Technical setup guide
- [Experiments/EXPERIMENT_GUIDE_V2.md](Experiments/EXPERIMENT_GUIDE_V2.md) - Experiment requirements (Indonesian)

---

*Research Project - Fresh Fruit Bunch Detection & Ripeness Classification*
