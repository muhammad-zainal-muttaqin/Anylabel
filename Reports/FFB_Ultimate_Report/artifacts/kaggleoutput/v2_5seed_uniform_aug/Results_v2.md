# FFB Oil Palm Detection - 5 Seed V2 Results

**Generated:** 2026-01-29 | **Seeds:** 42, 123, 456, 789, 101 | **Model:** YOLOv11n, 100 epochs
**Revision:** Uniform augmentation + BatchNorm Reset for depth experiments

---

## Executive Summary

| Finding | Details |
|---------|---------|
| **Best Localization** | A.3 (RGB+Real Depth) - **0.8403 mAP50** |
| **Best Depth-Only** | A.2 (Real Depth) - **0.7325 mAP50** |
| **Best RGBD Fusion** | A.3 (RGB+Real Depth) - **0.8403 mAP50** |
| **Real vs Synthetic Depth** | Real (0.7325) > Synthetic (0.6533) by 7.9% |
| **RGBD Fusion (A.3)** | Slightly better than RGB baseline (+0.2%) |
| **RGB+Synthetic (A.4b)** | Underperforms RGB baseline by 1.8% |
| **Late Fusion (A.5)** | **0.8084 mAP50** - Ranks #4 |
| **Most Stable** | A.4b RGB+Synthetic (std=0.0122) |

---

## A Series: Localization Results (V2)

### Summary Comparison

| Exp | Input | mAP50 | mAP50-95 | Precision | Recall | Rank |
|:----|:------|:-----:|:--------:|:---------:|:------:|:----:|
| A.3 | RGB + Real Depth | **0.8403±0.0161** | 0.3687±0.0104 | 0.8019±0.0501 | **0.7743±0.0177** | 1 |
| A.1 | RGB Only | 0.8385±0.0249 | 0.3645±0.0111 | 0.8028±0.0456 | 0.7605±0.0566 | 2 |
| A.4b | RGB + Synthetic Depth | 0.8233±0.0122 | **0.3676±0.0074** | **0.7959±0.0341** | 0.7387±0.0403 | 3 |
| **A.5** | **Late Fusion** | **0.8084±0.0304** | **0.3176±0.0150** | - | - | **4** |
| A.2 | Real Depth Only | 0.7325±0.0419 | 0.2915±0.0118 | 0.7147±0.0379 | 0.7267±0.0125 | 5 |
| A.4a | Synthetic Depth Only | 0.6533±0.0363 | 0.2754±0.0252 | 0.7176±0.0396 | 0.6170±0.0818 | 6 |

**Note:** All experiments use **uniform augmentation**: translate=0.1, scale=0.5, fliplr=0.5, HSV/mosaic/mixup disabled.
**Note:** A.5 Precision & Recall not available in raw notebook output; requires re-run evaluation.

### Per-Seed Details

<details>
<summary><b>A.1 - RGB Only (V2)</b></summary>

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | **0.8809** | 0.3662 | 0.7571 | **0.8313** |
| 123 | 0.8324 | **0.3697** | **0.8684** | 0.6913 |
| 456 | 0.8148 | 0.3657 | 0.7717 | 0.8047 |
| 789 | 0.8325 | 0.3751 | 0.8296 | 0.7417 |
| 101 | 0.8317 | 0.3458 | 0.7873 | 0.7333 |

**Mean±Std:** 0.8385±0.0249

</details>

<details>
<summary><b>A.2 - Real Depth Only (V2) + BN Reset</b></summary>

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.6908 | 0.2742 | 0.7001 | **0.7238** |
| 123 | 0.7169 | 0.2987 | 0.6682 | **0.7479** |
| **456** | **0.8005** | **0.3054** | **0.7636** | 0.7238 |
| 789 | 0.7138 | 0.2881 | 0.6996 | 0.7238 |
| 101 | 0.7402 | 0.2910 | 0.7419 | 0.7143 |

**Mean±Std:** 0.7325±0.0419 | **BN Reset:** Using 100 real training images

</details>

<details>
<summary><b>A.3 - RGB + Real Depth (4-Ch) (V2) + BN Reset</b></summary>

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.8386 | 0.3549 | 0.8019 | **0.8000** |
| 123 | 0.8344 | 0.3618 | 0.7939 | 0.7810 |
| 456 | 0.8337 | **0.3798** | 0.7488 | 0.7666 |
| **789** | **0.8681** | 0.3701 | **0.8838** | 0.7524 |
| 101 | 0.8267 | 0.3770 | 0.7809 | 0.7714 |

**Mean±Std:** 0.8403±0.0161 | **BN Reset:** Callback on_train_start with 100 real images

</details>

<details>
<summary><b>A.4a - Synthetic Depth Only (V2) + BN Reset</b></summary>

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.6066 | 0.2333 | **0.7244** | 0.4952 |
| 123 | 0.6309 | 0.2789 | 0.6483 | **0.7048** |
| 456 | 0.6707 | 0.2762 | 0.7408 | 0.6190 |
| 789 | 0.6577 | **0.2981** | 0.7291 | 0.5897 |
| **101** | **0.7009** | 0.2906 | 0.7452 | 0.6762 |

**Mean±Std:** 0.6533±0.0363 | **Depth Source:** Depth-Anything-V2

</details>

<details>
<summary><b>A.4b - RGB + Synthetic Depth (4-Ch) (V2) + BN Reset</b></summary>

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.8210 | 0.3636 | 0.7489 | **0.7669** |
| 123 | 0.8149 | 0.3704 | 0.8019 | 0.7143 |
| 456 | 0.8103 | 0.3614 | 0.7775 | 0.6857 |
| **789** | **0.8410** | **0.3793** | **0.8381** | 0.7397 |
| 101 | 0.8293 | 0.3632 | 0.8130 | 0.7868 |

**Mean±Std:** 0.8233±0.0122 | **Depth Source:** Depth-Anything-V2

</details>

<details>
<summary><b>A.5 - Late Fusion (V2) - COMPLETE</b></summary>

| Seed | mAP50 | mAP50-95 |
|:----:|:-----:|:--------:|
| 42 | 0.7610 | 0.2977 |
| 123 | 0.7955 | 0.3120 |
| 456 | 0.8279 | **0.3390** |
| **789** | **0.8347** | 0.3205 |
| 101 | 0.8229 | 0.3188 |

**Mean±Std (mAP50):** 0.8084±0.0304 | **Mean±Std (mAP50-95):** 0.3176±0.0150 | **Architecture:** Dual frozen backbones + trainable fusion
**Note:** Precision & Recall not available in raw notebook output; requires re-run evaluation.

</details>

---

## Key Insights V2 vs V1

### Comparison with V1 (Before Revision)

| Exp | V1 mAP50 | V2 mAP50 | Δ Change | Note |
|:----|:--------:|:--------:|:--------:|:-----|
| A.1 RGB | 0.869 | 0.839 | -3.5% | Uniform aug (HSV disabled) |
| A.2 Depth | 0.748 | 0.733 | -2.0% | + BN Reset (100 real images) |
| A.3 RGBD | 0.842 | 0.840 | -0.2% | + BN Reset real images |
| A.4a Syn | 0.708 | 0.653 | -7.8% | + BN Reset (100 real images) |
| A.4b Syn | 0.813 | 0.823 | +1.2% | + BN Reset real images |

### Key Findings V2

1. **RGBD Fusion Improvement:** A.3 now nearly equal to RGB baseline (only 0.2% difference)
2. **BN Reset Effectiveness:** A.4b improved 1.2% with BN reset using real images
3. **Uniform Augmentation Impact:** A.1 decreased due to disabling HSV augmentation
4. **Real > Synthetic Consistent:** Real depth (0.733) still outperforms synthetic (0.653) by 7.9%
5. **Late Fusion Complete:** A.5 achieves 0.8084 mAP50, ranking #4 among all experiments
6. **A.5 Analysis:** Highest variance (std=0.0304) indicates seed sensitivity; best seed (789: 0.8347) nearly matches RGB baseline

### Stability Analysis (Std Dev of mAP50)

| Experiment | Std Dev | Rating |
|:-----------|:-------:|:------:|
| A.4b RGB+Synthetic | 0.0122 | Most Stable |
| A.3 RGB+Real Depth | 0.0161 | Very Stable |
| A.1 RGB Only | 0.0249 | Stable |
| A.5 Late Fusion | 0.0304 | Moderate |
| A.4a Synthetic Depth | 0.0363 | Moderate-Low |
| A.2 Real Depth | 0.0419 | Least Stable |

---

## Recommendations

| Scenario | V2 Recommendation | Rationale |
|:---------|:------------------|:----------|
| **Best Overall Detection** | A.3 RGB+Real Depth | Highest mAP50 (0.8403), very stable |
| **Simplest Setup** | A.1 RGB Only | Nearly equal to A.3, simpler setup |
| **No Depth Sensor** | A.4b RGB+Synthetic | Only 1.7% below RGB, no sensor needed |
| **Late Fusion Architecture** | A.5 Late Fusion | Viable (0.8084) but not superior to A.3/A.4b |
| **Depth Only** | A.2 Real Depth | 0.7325 mAP50 with BN reset |

---

## Technical Details

### Uniform Augmentation (All Experiments)

```python
translate=0.1    # Geometric ✅
scale=0.5        # Geometric ✅
fliplr=0.5       # Geometric ✅
hsv_h=0.0        # Disabled (non-geometric)
hsv_s=0.0        # Disabled (non-geometric)
hsv_v=0.0        # Disabled (non-geometric)
mosaic=0.0       # Disabled
mixup=0.0        # Disabled
erasing=0.0      # Disabled
```

### BatchNorm Reset (A.2, A.3, A.4a, A.4b)

- **Method:** Forward pass 100 real training images
- **A.2, A.4a:** PIL + transforms, batch size 16
- **A.3, A.4b:** Callback `on_train_start` with train_loader

### A.5 Late Fusion Architecture

```
Input RGB (3ch) ──► RGB Backbone (Frozen A.1)
                      └──► P3, P4, P5 ──┐
                                       ├──► Concat ──► 1x1 Conv ──► Detect Head
Input Depth (3ch) ──► Depth Backbone (Frozen A.2)    (Trainable)
                      └──► P3, P4, P5 ──┘
```

---

## Appendix: Model Weights & Results

**Weights location:** `<exp_dir>/runs/detect/exp_<name>_seed<N>/weights/best.pt`

**Raw results:** `<exp_dir>/kaggleoutput/<exp>_results.txt`

| Experiment | Directory | Status |
|:-----------|:----------|:-------|
| A.1 | train_a1_rgb | ✅ Valid |
| A.2 | train_a2_depth | ✅ Valid |
| A.3 | train_a3_rgbd | ✅ Valid |
| A.4a | train_a4a_synthetic_depth | ✅ Valid |
| A.4b | train_a4b_rgbd_synthetic | ✅ Valid |
| A.5 | train_a5_late_fusion | ✅ Valid |

---

*Report from 5-seed V2 experiments (42, 123, 456, 789, 101) with uniform augmentation and BN reset*
