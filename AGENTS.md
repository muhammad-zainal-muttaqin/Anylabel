# AGENTS.md - AI Coding Agent Guide

This file provides essential information for AI coding agents working with this repository. It complements [CLAUDE.md](CLAUDE.md) with project structure and conventions for automated agents.

---

## Project Overview

**Name**: FFB Detection & Ripeness Classification (Anylabel)  
**Domain**: Computer Vision / Deep Learning  
**Language**: Python  
**Documentation Languages**: English and Indonesian (Bahasa Indonesia)

This is a research project for Fresh Fruit Bunch (FFB) oil palm detection and ripeness classification using YOLO models. The project experiments with RGB, depth, and RGBD inputs for both object detection (localization) and ripeness classification tasks.

---

## Technology Stack

### Core Dependencies
- **Python**: 3.10+ (3.12 recommended)
- **Deep Learning**: Ultralytics (YOLOv8/YOLOv11), PyTorch, Transformers
- **Computer Vision**: OpenCV, Pillow
- **Data Processing**: NumPy, Pandas, Matplotlib, Seaborn
- **Augmentation**: Albumentations (for RGBD synced augmentation)
- **Annotation**: AnyLabeling (manual bounding box annotation)

### Platform
- **OS**: Windows 10/11 (primary), Linux compatible
- **Hardware**: NVIDIA GPU recommended for training
- **Cloud**: Kaggle (for GPU training)

---

## Project Structure

```
Anylabel/
├── Dataset/                    # Raw dataset storage
│   ├── 28574489.zip           # Original dataset (4.3GB)
│   ├── gohjinyu-oilpalm-ffb-dataset-d66eb99/  # Extracted data
│   │   ├── ffb-localization/
│   │   │   ├── rgb_images/    # RGB images (1280x720)
│   │   │   ├── depth_maps/    # Raw depth maps (16-bit PNG)
│   │   │   └── point_clouds/  # PLY point cloud files
│   │   └── ffb-ripeness/      # Ripeness classification data
│   └── scripts/               # Data collection utilities
│       ├── ffb_data_collection.py
│       ├── opencv_pc.py
│       └── stitching.py
│
├── Experiments/               # Training experiments
│   ├── scripts/              # Python training scripts
│   │   ├── Data Preparation:
│   │   │   ├── simple_eda.py
│   │   │   ├── split_localization_data.py
│   │   │   ├── prepare_depth_data.py
│   │   │   ├── convert_json_to_yolo.py
│   │   │   ├── generate_synthetic_depth.py
│   │   │   └── prepare_synthetic_depth_data.py
│   │   ├── Training:
│   │   │   ├── train_a1_rgb.py
│   │   │   ├── train_a2_depth.py
│   │   │   ├── train_a3_rgbd.py
│   │   │   ├── train_a4a_synthetic_depth.py
│   │   │   ├── train_a4b_rgbd_synthetic.py
│   │   │   ├── train_b1_classification.py
│   │   │   ├── train_b2_stage1_detector.py
│   │   │   └── train_b2_stage2_classifier.py
│   │   ├── Data Generation:
│   │   │   ├── custom_rgbd_dataset.py
│   │   │   └── extract_crops_b2.py
│   │   ├── Evaluation:
│   │   │   ├── evaluate_all.py
│   │   │   ├── failure_analysis.py
│   │   │   ├── find_best_map.py
│   │   │   ├── inference_b2_twostage.py
│   │   │   └── compare_real_vs_synthetic.py
│   │   └── Kaggle Upload:
│   │       └── build_uploadkaggle_*.py
│   │
│   ├── configs/              # YOLO dataset YAML configs
│   │   ├── ffb_localization.yaml
│   │   ├── ffb_localization_rgbd.yaml
│   │   └── ffb_localization_uploadkaggle.yaml
│   │
│   ├── datasets/             # Processed train/val/test splits
│   │   ├── ffb_localization/
│   │   ├── ffb_localization_depth/
│   │   ├── depth_processed_rgb/
│   │   └── ffb_ripeness/
│   │
│   ├── notebooks/            # Jupyter notebooks (V3 experiments)
│   │   ├── train_a3_rgbd_fix.ipynb
│   │   ├── generate_synthetic_depth.ipynb
│   │   ├── train_a4a_synthetic_depth.ipynb
│   │   ├── train_a4b_rgbd_synthetic.ipynb
│   │   └── train_b2_twostage.ipynb
│   │
│   ├── runs/                 # Training outputs (YOLO default)
│   ├── eda_output/           # EDA reports
│   ├── UploadKaggle/         # Kaggle dataset packages
│   └── kaggleoutput/         # Training results from Kaggle
│
├── Reports/                  # Final reports and artifacts
│   └── FFB_Ultimate_Report/
│       ├── artifacts/        # Experiment outputs snapshot
│       ├── assets/           # Images and visualizations
│       ├── result.md         # Main consolidated report
│       └── README.md
│
├── venv/                     # Python virtual environment
├── requirements.txt          # Python dependencies
├── README.md                 # Quick start guide
├── CLAUDE.md                 # Comprehensive documentation
└── AGENTS.md                 # This file
```

---

## Experiment Types

### Localization (Object Detection)
| Experiment | Input | Channels | Classes | Description |
|:-----------|:------|:---------|:--------|:------------|
| **A.1** | RGB | 3 | 1 (FFB) | RGB baseline |
| **A.2** | Depth (real) | 3 | 1 | Real depth only |
| **A.3** | RGB+Depth | 4 | 1 | RGB+Depth fusion with synced augmentation |
| **A.4a** | Synthetic Depth | 3 | 1 | Synthetic depth (Depth-Anything-V2) |
| **A.4b** | RGB+Synthetic Depth | 4 | 1 | RGB + Synthetic depth fusion |

### Ripeness Classification
| Experiment | Approach | Classes | Description |
|:-----------|:---------|:--------|:------------|
| **B.1** | End-to-End | 2 (ripe/unripe) | Single-stage detection |
| **B.2** | Two-Stage | 2 | Detect → Crop → Classify |

---

## Development Conventions

### File Paths
- **Always use absolute paths** in YOLO dataset YAML configs
- Use Windows-style backslashes for local paths
- Scripts expect to be run from `Experiments/` directory
- Dataset paths: `D:\Work\Assisten Dosen\Anylabel\Experiments\datasets\...`

### Code Style
- Follow PEP 8 for Python code
- Use type hints where practical
- Include docstrings for functions
- No emojis in code comments (use in markdown only)
- Clean, direct technical communication

### Naming Conventions
- Scripts: `snake_case.py`
- Experiments: `train_{experiment_id}_{description}.py`
- Results: `exp_{experiment_id}_{description}_seed{number}`
- Images: `rgb_{xxxx}.png`, `depth_{xxxx}.png`

---

## Build and Run Commands

### Environment Setup
```powershell
# Activate virtual environment (REQUIRED)
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
```bash
cd Experiments/scripts

# Step 1: EDA
python simple_eda.py

# Step 2: Split dataset (70:20:10)
python split_localization_data.py

# Step 3: Process depth maps
python prepare_depth_data.py

# Step 4: Generate synthetic depth (optional)
python generate_synthetic_depth.py
python prepare_synthetic_depth_data.py
```

### Training Commands
```bash
cd Experiments/scripts

# A.1: RGB baseline (run twice with seeds 42, 123)
python train_a1_rgb.py

# A.2: Depth only
python train_a2_depth.py

# A.3: RGB+Depth
python train_a3_rgbd.py

# A.4a: Synthetic depth only
python train_a4a_synthetic_depth.py

# A.4b: RGB+Synthetic depth
python train_a4b_rgbd_synthetic.py

# B.1: Ripeness detection
python train_b1_classification.py

# B.2: Two-stage (full pipeline)
python train_b2_stage1_detector.py
python extract_crops_b2.py
python train_b2_stage2_classifier.py
```

### Evaluation Commands
```bash
# Evaluate all experiments
python evaluate_all.py

# Failure analysis
python failure_analysis.py

# Find best model by mAP
python find_best_map.py

# Manual YOLO validation
cd Experiments
yolo detect val model=runs/detect/exp_name/weights/best.pt data=configs/ffb_localization.yaml split=test
```

---

## Training Standards

### Reproducibility
- **Seeds**: Always use 42 and 123 (2 runs per experiment)
- **Model**: YOLOv11n (Nano) baseline, YOLOv11s (Small) for ablation
- **Epochs**: 50 (baseline), 300 (ablation studies with early stopping)
- **Batch Size**: 16 (detection), 32 (classification)
- **Image Size**: 640x640 (detection), 224x224 (classification)

### Evaluation Metrics
- **Detection**: mAP50, mAP50-95
- **Classification**: Top-1 Accuracy
- **Report**: mean ± std dev from 2 runs
- **Test Set**: Always evaluate on test split (not validation)

### Depth Processing Pipeline
1. Raw depth maps: 16-bit PNG (mm, 0-65535 range)
2. Filter to valid range: 600-6000mm (0.6m-6m)
3. Normalize to 0-255 using min-max scaling
4. Replicate single channel to 3 channels (R=G=B) for YOLO
5. Save as 3-channel PNG in `depth_processed_rgb/`

---

## Key Configuration Files

### YOLO Dataset Config Pattern
```yaml
# Experiments/configs/ffb_localization.yaml
path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_localization
train: images/train
val: images/val
test: images/test
nc: 1
names: ['fresh_fruit_bunch']
```

**Critical**: Always use absolute paths in YAML configs for YOLO compatibility.

---

## Testing Strategy

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

### Model Validation
- Run 2 training runs per experiment (seeds 42, 123)
- Evaluate on **test set** for final metrics
- Compare mean and standard deviation
- Document failures with visual examples

---

## Security Considerations

### Data Handling
- Dataset contains agricultural imagery only (no PII)
- No sensitive credentials in code
- Model weights (.pt files) are binary artifacts

### Execution Safety
- Scripts should not require elevated privileges
- All operations confined to project directory
- No external network calls except for:
  - HuggingFace model downloads (Depth-Anything-V2)
  - Kaggle dataset uploads (explicit user action)

---

## Common Issues & Troubleshooting

### PowerShell Execution
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### CUDA Out of Memory
- Reduce batch size: 16 → 8 → 4
- Or use CPU: `device='cpu'` in training script

### No Labels Found
- Verify annotations exist in `Experiments/datasets/*/labels/train/`
- Check image/label count mismatch

### Python Not Found
Use full path: `.\venv\Scripts\python.exe`

---

## Reference Documents

| Document | Purpose |
|:---------|:--------|
| `CLAUDE.md` | Comprehensive project documentation |
| `README.md` | Quick start guide (Indonesian/English) |
| `Experiments/README.md` | Technical setup guide |
| `Reports/FFB_Ultimate_Report/result.md` | Final consolidated results |

---

## Notes for AI Agents

1. **Always activate venv** before running any Python commands
2. **Never make assumptions** about dataset paths - use absolute paths
3. **Preserve existing code style** when editing
4. **Clean up temporary files** immediately after failures
5. **Use todo list** for multi-step tasks
6. **No debug artifacts** in final code
7. **Documentation language**: Match existing style (Indonesian for internal docs, English for technical configs)

---

*Last Updated: 2026-01-27*
