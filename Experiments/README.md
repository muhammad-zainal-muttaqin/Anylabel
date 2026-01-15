# FFB Detection Experiments - Technical Setup

Project: Fresh Fruit Bunch Oil Palm Detection using YOLO Models  
Last Updated: 2026-01-15

## Prerequisites

### System Requirements
- Python 3.8+
- GPU recommended (NVIDIA CUDA)
- 10GB+ free disk space
- Windows 10/11 or Linux

### Installation
```bash
# Activate virtual environment
.\venv\Scripts\Activate

# Install core dependencies
pip install ultralytics==8.3.0 opencv-python numpy pandas matplotlib seaborn
```

## Dataset Preparation

### 1. Extract Dataset
```bash
# Source
Dataset\28574489.zip (4.3GB)

# Destination
Dataset\gohjinyu-oilpalm-ffb-dataset-d66eb99\
```

Required structure:
```
ffb-localization/
├── rgb_images/
├── depth_maps/
└── point_clouds/
```

### 2. Manual Annotation
Tool: AnyLabeling  
Target: 300-500+ images minimum

Configuration:
- Input: Dataset/gohjinyu.../ffb-localization/rgb_images/
- Output: Experiments/datasets/ffb_localization/labels/
- Class: fresh_fruit_bunch (1 class)
- Format: YOLO

Quality Checklist:
- All bounding boxes tight around fruit
- Occluded fruits: annotate if >50% visible
- Minimum 300 images with labels
- Label files: .txt format, one per image

## Experiment Phases

### Phase 0: Data Analysis & Preparation
```bash
cd Experiments\scripts

# Step 1: EDA
python simple_eda.py

# Step 2: Split data 70:20:10
python split_localization_data.py

# Step 3: Process depth maps
python prepare_depth_data.py
```

Output Structure:
```
Experiments/datasets/
├── ffb_localization/          # RGB split
├── ffb_localization_depth/    # Depth split
└── depth_processed_rgb/       # 3-channel depth
```

### Phase 1: Experiment A.1 - RGB Baseline
```bash
# Training Run 1 (seed=42)
python train_a1_rgb.py --run 1

# Training Run 2 (seed=123)
python train_a1_rgb.py --run 2
```

Config:
- Model: YOLOv8n
- Epochs: 50
- Batch: 16
- Image size: 640x640
- Output: runs/detect/exp_a1_rgb_*

### Phase 2: Experiment A.2 - Depth Only
```bash
# Training Run 1 (seed=42)
python train_a2_depth.py --run 1

# Training Run 2 (seed=123)
python train_a2_depth.py --run 2
```

Config:
- Model: YOLOv8n
- Input: 3-channel depth (normalized 0-255)
- Same hyperparameters as A.1
- Output: runs/detect/exp_a2_depth_*

### Phase 3: Experiment B.1 - Classification
```bash
# Training Run 1 (seed=42)
python train_b1_classification.py --run 1

# Training Run 2 (seed=123)
python train_b1_classification.py --run 2
```

Config:
- Model: YOLOv8n-cls
- Classes: ripe_ffb, unripe_ffb
- Epochs: 50
- Batch: 32
- Image size: 224x224
- Output: runs/classify/exp_b1_cls_*

## Evaluation & Reporting

### Evaluate All Models
```bash
python evaluate_all.py
```

Generates:
- LAPORAN_EKSPERIMEN.md
- experiment_results.csv
- Statistical analysis (mean, std dev)

### Failure Analysis
```bash
python failure_analysis.py
```

Generates:
- failure_analysis/ directory
- False Positive visualizations
- False Negative visualizations
- Correct detection examples
- Error analysis by conditions

## Expected Results

Target Metrics:
- A.1 RGB: mAP50 0.70 - 0.85
- A.2 Depth: mAP50 0.60 - 0.80
- B.1 Cls: Top1 Acc 0.80 - 0.95

Output Files Location:
```
Experiments/
├── runs/
│   ├── detect/exp_a1_rgb_baseline/weights/best.pt
│   ├── detect/exp_a2_depth_only/weights/best.pt
│   └── classify/exp_b1_rgb_classification/weights/best.pt
├── LAPORAN_EKSPERIMEN.md
├── experiment_results.csv
└── failure_analysis/
```

## Execution Order Summary

1. Extract Dataset\28574489.zip
2. Annotate 300+ images (AnyLabeling)
3. Run simple_eda.py
4. Run split_localization_data.py
5. Run prepare_depth_data.py
6. Train A.1 RGB (2 runs)
7. Train A.2 Depth (2 runs)
8. Train B.1 Classification (2 runs)
9. Run evaluate_all.py
10. Run failure_analysis.py

## Configuration Files

ffb_localization.yaml:
```yaml
path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_localization
train: images/train
val: images/val
test: images/test
nc: 1
names: ['fresh_fruit_bunch']
```

## Troubleshooting

CUDA Out of Memory:
- Edit training script: reduce batch size (8 or 4)
- Or use CPU: device: cpu

No Labels Found:
- Verify annotation count matches images
- Check Experiments/datasets/ffb_localization/labels/train

Training Too Slow:
- Use GPU (device: 0)
- Ensure CUDA installed

## File Reference

| File | Purpose |
|------|---------|
| EXPERIMENT_GUIDE_V2.md | Detailed Indonesian guide |
| ffb_localization.yaml | YOLO dataset config |
| scripts/simple_eda.py | Dataset analysis |
| scripts/split_localization_data.py | Train/Val/Test split |
| scripts/prepare_depth_data.py | Depth normalization |
| scripts/train_a1_rgb.py | RGB training |
| scripts/train_a2_depth.py | Depth training |
| scripts/train_b1_classification.py | Classification training |
| scripts/evaluate_all.py | Metrics & reporting |
| scripts/failure_analysis.py | Visual error analysis |

## Notes

- All experiments use seeds 42 and 123 for reproducibility
- Depth maps normalized from 0.6m-6m to 0-255 range
- Training logs saved to runs/ directory
- Evaluation generates statistical analysis
- Failure analysis creates visual reports

---
**Reference:** EXPERIMENT_GUIDE_V2.md (Indonesian, detailed)
