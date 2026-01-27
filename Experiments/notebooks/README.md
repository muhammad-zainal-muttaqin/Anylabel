# Jupyter Notebooks - V3 Experiments

Jupyter notebooks for interactive FFB detection and ripeness classification experiments. These notebooks are optimized for **Kaggle execution** but can also run locally.

## Experiment Notebooks

### Data Generation

| Notebook | Description | Input | Output |
|----------|-------------|-------|--------|
| `generate_synthetic_depth.ipynb` | Generate synthetic depth maps using Depth-Anything-V2 | RGB images | 3-channel depth maps |

**Usage**: Run this first if you need synthetic depth data.

---

### Detection Experiments (1-class: FFB)

| Notebook | Experiment | Input | Dataset | Purpose |
|----------|-----------|-------|---------|---------|
| `train_a1_rgb.ipynb` | A.1 | 3-ch RGB | `ffb-localization` | RGB baseline detection |
| `train_a2_depth.ipynb` | A.2 | 3-ch Real Depth | `ffb-localization-depth` | Real depth only detection |
| `train_a3_rgbd.ipynb` | A.3 | 4-ch RGB+Depth | `ffb-localization-rgbd` | RGB+Real depth fusion |
| `train_a4a_synthetic_depth.ipynb` | A.4a | 3-ch Synth Depth | `ffb-synthetic-depth` | Synthetic depth only detection |
| `train_a4b_rgbd_synthetic.ipynb` | A.4b | 4-ch RGB+Synth | `ffb-rgbd-synthetic` | RGB+Synthetic depth fusion |

**Training Standard**:
- Seeds: 5 × [42, 123, 456, 789, 101]
- Epochs: 100, Patience: 30
- Model: YOLOv11n
- Metrics: mAP50, mAP50-95, Precision, Recall
- Evaluation: Test set only

---

### Ripeness Classification Experiments

| Notebook | Experiment | Type | Input | Classes |
|----------|-----------|------|-------|---------|
| `train_b1_ripeness.ipynb` | B.1 | End-to-end | 3-ch RGB | 2 (Ripe/Unripe) |
| `train_b2_twostage.ipynb` | B.2 | Two-stage | 3-ch RGB | 2 (Ripe/Unripe) |

**B.1 - End-to-End Detection**:
- Detects and classifies FFBs in one stage
- Dataset: `ffb-ripeness-detect`

**B.2 - Two-Stage Pipeline**:
- Stage 1: Detect FFBs (ripe/unripe)
- Stage 2: Extract crops with 10% margin and classify
- Evaluates full pipeline end-to-end
- Dataset: `ffb-ripeness-detect`

---

## How to Use

### Local Execution

```bash
# Activate virtual environment
.\venv\Scripts\Activate

# Start Jupyter
jupyter lab

# Or use Jupyter Notebook
jupyter notebook

# Navigate to Experiments/notebooks/ and open a notebook
```

### Kaggle Execution

1. Create new Kaggle notebook
2. Copy-paste notebook content
3. Add required datasets:
   - `ffb-localization` (for A.1, B.1, B.2)
   - `ffb-localization-depth` (for A.2)
   - `ffb-localization-rgbd` (for A.3)
   - `ffb-synthetic-depth` (for A.4a)
   - `ffb-rgbd-synthetic` (for A.4b)
   - `ffb-ripeness-detect` (for B.1, B.2)
4. Enable GPU accelerator
5. Run all cells

---

## Notebook Structure

Each notebook follows this 10-13 cell pattern:

| Cell # | Content |
|--------|---------|
| 0 | Markdown header with experiment description |
| 1 | Setup (paths, environment, dataset verification) |
| 2 | Create YAML config (%%writefile) |
| 3 | Verify YAML and check inputs |
| 4 | Install dependencies (!pip install) |
| 5 | Training config (SEEDS, EXP_PREFIX) |
| 6 | Training loop (5 seeds) |
| 7 | Evaluation on test set |
| 8 | Results summary (mean ± std deviation) |
| 9 | Save results to `kaggleoutput/*.txt` |
| 10 | Create ZIP archives for download |

---

## Output Files

Each notebook generates:

```
Experiments/kaggleoutput/
├── {exp}_results.txt     # Results summary with mean ± std
├── {exp}_runs.zip        # Training runs archive
└── {exp}_output.zip      # Output archive
```

Results format example:
```
EXPERIMENT RESULTS
Per-Seed Results:
Seed  mAP50  mAP50-95  Precision  Recall
42    0.873  0.430     0.806      0.814
123   0.877  0.436     0.819      0.817
...
Summary (Mean ± Std):
  mAP50: 0.875 ± 0.002
  mAP50-95: 0.433 ± 0.003
  Precision: 0.813 ± 0.007
  Recall: 0.816 ± 0.002
```

---

## Key Features

✅ **Auto-detect environment** (Kaggle vs Local)
✅ **Automatic result saving** to kaggleoutput
✅ **5-seed training** for statistical robustness
✅ **Comprehensive evaluation** with comparison tables
✅ **Default YOLO parameters** (no custom tuning)
✅ **Mean ± std deviation** reporting

---

## Dependencies

- **ultralytics** - YOLO models
- **torch** - Deep learning framework
- **opencv-python** - Image processing
- **numpy, pandas** - Data manipulation
- **matplotlib, seaborn** - Visualization

Installed automatically in cell 4.

---

## Comparison with Python Scripts

| Aspect | Notebooks | Scripts |
|--------|-----------|---------|
| Environment | Kaggle + Local | Mostly local |
| Interactivity | High (cells, plots) | Low (terminal only) |
| Code organization | Linear flow | Modular functions |
| Output handling | Auto-save | Manual handling |
| Debugging | Interactive cells | Terminal debugging |

**Recommendation**: Use notebooks for **Kaggle execution** and interactive work. Use scripts for **batch processing** and automation.

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in training config

### Dataset not found
- Verify dataset is added to Kaggle notebook
- Check absolute path in YAML config

### Model not loading
- Ensure best.pt exists in runs/detect/ or runs/classify/
- Verify dataset YAML path is correct

### Results not saving
- Check kaggleoutput/ directory exists
- Verify write permissions

---

## Related Documentation

- **Setup Guide**: `Experiments/README.md`
- **Script Reference**: `Experiments/scripts/README.md`
- **Legacy/Ablation**: `Experiments/legacy/docs/ablation_study_plan.md`
- **Main Report**: `Reports/FFB_Ultimate_Report/result.md`
- **Project Instructions**: `CLAUDE.md`

---

*Notebooks created: 2026-01-27 | Last updated: 2026-01-27*
