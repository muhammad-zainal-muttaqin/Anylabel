# 5 Seed Experiments - Version 2

**Folder:** `5_seed_v2/`
**Created:** 2026-01-28
**Purpose:** Standardized experiments with uniform augmentation and domain adaptation

---

## Changes from 5_seed v1

| Aspect | v1 (Legacy) | v2 (Current) |
|:------|:----------|:----------|
| **Augmentation** | Mixed (A.1 default, A.3/A.4b HSV=0) | Uniform geometric-only (all) |
| **BN Reset** | Not applied | Applied for depth experiments |
| **Late Fusion** | Not available | A.5 (new) |

---

## Folder Structure

```
5_seed_v2/
├── README.md                          # This file
├── ACTION_PLAN.md                     # Detailed implementation plan
├── Results.md                         # Full detailed results
├── Results_v2.md                      # Concise summary
│
├── train_a1_rgb/                      # A.1: RGB Only
│   ├── runs/detect/exp_a1_rgb_seed{N}/
│   └── kaggleoutput/a1_rgb_results.txt
│
├── train_a2_depth/                    # A.2: Real Depth Only
│   ├── runs/detect/exp_a2_depth_seed{N}/
│   └── kaggleoutput/a2_depth_results.txt
│
├── train_a3_rgbd/                     # A.3: RGB + Real Depth (4-ch)
│   ├── runs/detect/exp_a3_rgbd_seed{N}/
│   └── kaggleoutput/a3_rgbd_results.txt
│
├── train_a4a_synthetic_depth/         # A.4a: Synthetic Depth Only
│   ├── runs/detect/exp_a4a_depth_seed{N}/
│   └── kaggleoutput/a4a_depth_results.txt
│
├── train_a4b_rgbd_synthetic/          # A.4b: RGB + Synthetic Depth (4-ch)
│   ├── runs/detect/exp_a4b_rgbd_seed{N}/
│   └── kaggleoutput/a4b_rgbd_results.txt
│
└── train_a5_late_fusion/              # A.5: Late Fusion (NEW)
    ├── runs/detect/exp_a5_fusion_seed{N}/
    └── kaggleoutput/a5_fusion_results.txt
```

---

## Training Configuration (v2)

### Augmentation (All Experiments)

```yaml
translate: 0.1    # ✅ Enabled
scale: 0.5        # ✅ Enabled
fliplr: 0.5       # ✅ Enabled
hsv_h: 0.0        # ❌ Disabled
hsv_s: 0.0        # ❌ Disabled
hsv_v: 0.0        # ❌ Disabled
erasing: 0.0      # ❌ Disabled
mosaic: 0.0       # ❌ Disabled
mixup: 0.0        # ❌ Disabled
```

### BatchNorm Reset (A.2, A.3, A.4a, A.4b)

- Forward pass: 100 batches of training data
- Timing: After loading pretrained weights, before main training
- Method: Real training images (not synthetic/dummy data)

### Late Fusion (A.5)

- RGB Backbone: Frozen (A.1 weights)
- Depth Backbone: Frozen (A.2 weights)
- Fusion layer: Trainable
- Architecture: Multi-scale (P3, P4, P5) with 1x1 convolutions

---

## Seeds

All experiments use 5 seeds for statistical reliability:

```python
seeds = [42, 123, 456, 789, 101]
```

---

## Status

| Experiment | Status | Seeds Completed |
|:-----------|:------:|:---------------:|
| A.1 RGB | ✅ Complete | 5/5 |
| A.2 Depth | ✅ Complete | 5/5 |
| A.3 RGBD | ✅ Complete | 5/5 |
| A.4a Synthetic | ✅ Complete | 5/5 |
| A.4b RGBD Synthetic | ✅ Complete | 5/5 |
| A.5 Late Fusion | ✅ Complete | 5/5 |

**Legend:**
- ⏳ Pending: Not started
- 🔄 In Progress: Running
- ✅ Complete: Finished

---

## Quick Results Summary

| Metric | Best Value | Experiment |
|:-------|:----------:|:-----------|
| **Best mAP50** | 0.8403 | A.3 (RGB+Real Depth) |
| **RGB Baseline** | 0.8385 | A.1 (RGB Only) |
| **Late Fusion** | 0.8084 | A.5 (Ranks #4) |
| **Depth-Only Best** | 0.7325 | A.2 (Real Depth) |
| **Most Stable** | 0.0122 std | A.4b (RGB+Synthetic) |

### Comparison Table

| Rank | Experiment | mAP50 | mAP50-95 | Std Dev | Status |
|:----:|:-----------|:-----:|:--------:|:-------:|:------:|
| 1 | A.3 RGB+Real Depth | 0.8403±0.0161 | 0.3687±0.0104 | 0.0161 | ✅ |
| 2 | A.1 RGB Only | 0.8385±0.0249 | 0.3645±0.0111 | 0.0249 | ✅ |
| 3 | A.4b RGB+Synthetic | 0.8233±0.0122 | 0.3676±0.0074 | 0.0122 | ✅ |
| 4 | A.5 Late Fusion | 0.8084±0.0304 | 0.3176±0.0160 | 0.0304 | ✅ |
| 5 | A.2 Real Depth | 0.7325±0.0419 | 0.2915±0.0118 | 0.0419 | ✅ |
| 6 | A.4a Synthetic Depth | 0.6533±0.0363 | 0.2754±0.0252 | 0.0363 | ✅ |

See `Results.md` and `Results_v2.md` for complete analysis.

---

## Documentation

| File | Description |
|:-----|:------------|
| `ACTION_PLAN.md` | Technical implementation details and architecture |
| `Results.md` | Full detailed report with visualizations |
| `Results_v2.md` | Concise summary and key findings |

---

## Notes

- This folder contains **standardized** results with uniform parameters
- Legacy results remain in `5_seed/` for reference
- A.5 is a new experiment not present in v1
- All experiments use identical training parameters for fair comparison

---

*For questions or issues, refer to the main repository documentation.*
