# FFB Oil Palm Detection & Ripeness Classification - 5 Seed Results

**Generated:** 2026-01-28 | **Seeds:** 42, 123, 456, 789, 101 | **Model:** YOLOv11n, 100 epochs

---

## Executive Summary

| Finding | Details |
|---------|---------|
| **Best Localization** | A.1 (RGB Only) - **0.869 mAP50** |
| **Worst Localization** | A.4a (Synthetic Depth Only) - **0.708 mAP50** |
| **Real vs Synthetic Depth** | Real (0.748) > Synthetic (0.708) by 5.7% |
| **RGBD Fusion (A.3)** | Underperforms RGB baseline by 3.1% |
| **RGB+Synthetic (A.4b)** | Underperforms RGB baseline by 6.4% |
| **Best Ripeness** | B.2 Two-Stage - **96.1% classification accuracy** |
| **Most Stable** | A.3 RGBD (std=0.013) |

---

## A Series: Localization Results

### Summary Comparison

| Exp | Input | mAP50 | mAP50-95 | Precision | Recall | Rank |
|:----|:------|:-----:|:--------:|:---------:|:------:|:----:|
| A.1 | RGB Only | **0.869±0.018** | **0.377±0.013** | **0.839±0.045** | 0.765±0.026 | 1 |
| A.3* | RGB + Real Depth | 0.842±0.013 | 0.363±0.010 | 0.780±0.041 | **0.779±0.035** | 2 |
| A.4b* | RGB + Synthetic Depth | 0.813±0.023 | 0.361±0.015 | 0.785±0.030 | 0.729±0.035 | 3 |
| A.2 | Real Depth Only | 0.748±0.038 | 0.276±0.018 | 0.743±0.031 | 0.726±0.042 | 4 |
| A.4a | Synthetic Depth Only | 0.708±0.029 | 0.273±0.009 | 0.716±0.060 | 0.661±0.050 | 5 |

*\*A.3 and A.4b use `erasing=0.0` and HSV disabled while A.1, A.2, A.4a use default settings. This ensures fair comparison between RGBD fusion experiments.*

### Per-Seed Details

<details>
<summary><b>A.1 - RGB Only (Baseline)</b></summary>

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.873 | 0.393 | 0.852 | 0.752 |
| 123 | 0.855 | 0.374 | 0.862 | 0.774 |
| 456 | 0.847 | 0.364 | 0.764 | 0.790 |
| 789 | 0.880 | 0.364 | 0.836 | 0.726 |
| 101 | 0.889 | 0.388 | 0.879 | 0.781 |

</details>

<details>
<summary><b>A.2 - Real Depth Only</b></summary>

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.787 | 0.291 | 0.765 | 0.774 |
| 123 | 0.728 | 0.256 | 0.770 | 0.695 |
| 456 | 0.751 | 0.278 | 0.759 | 0.721 |
| 789 | 0.779 | 0.295 | 0.725 | 0.762 |
| 101 | 0.696 | 0.259 | 0.698 | 0.676 |

</details>

<details>
<summary><b>A.3 - RGB + Real Depth (4-Channel)*</b></summary>

**Note:** Uses `erasing=0.0` vs default `erasing=0.4`. HSV disabled correctly for depth channel protection.

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.858 | 0.377 | 0.842 | 0.762 |
| 123 | 0.825 | 0.359 | 0.755 | 0.790 |
| 456 | 0.838 | 0.353 | 0.742 | 0.781 |
| 789 | 0.842 | 0.370 | 0.802 | 0.733 |
| 101 | 0.849 | 0.359 | 0.759 | 0.829 |

</details>

<details>
<summary><b>A.4a - Synthetic Depth Only</b></summary>

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.673 | 0.271 | 0.652 | 0.625 |
| 123 | 0.699 | 0.263 | 0.701 | 0.647 |
| 456 | 0.705 | 0.265 | 0.805 | 0.629 |
| 789 | 0.753 | 0.285 | 0.680 | 0.748 |
| 101 | 0.710 | 0.278 | 0.741 | 0.657 |

</details>

<details>
<summary><b>A.4b - RGB + Synthetic Depth (4-Channel)*</b></summary>

**Note:** Uses same settings as A.3 (`erasing=0.0`, HSV disabled) for fair comparison.

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.830 | 0.359 | 0.826 | 0.768 |
| 123 | 0.836 | 0.375 | 0.777 | 0.762 |
| 456 | 0.801 | 0.371 | 0.806 | 0.714 |
| 789 | 0.779 | 0.337 | 0.758 | 0.686 |
| 101 | 0.820 | 0.362 | 0.758 | 0.714 |

</details>

---

## B Series: Ripeness Classification Results

### B.1 - End-to-End (Joint Detection + Classification)

Single model detects AND classifies FFBs as ripe/unripe simultaneously.

| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42 | 0.846 | 0.554 | 0.861 | 0.780 |
| 123 | 0.805 | 0.538 | 0.718 | 0.806 |
| 456 | 0.807 | 0.549 | 0.797 | 0.712 |
| 789 | 0.811 | 0.504 | 0.784 | 0.734 |
| 101 | 0.800 | 0.518 | 0.732 | 0.799 |
| **Mean±Std** | **0.814±0.018** | **0.532±0.021** | **0.779±0.057** | **0.766±0.042** |

*Note: mAP50 includes both localization and classification performance (2-class detection).*

### B.2 - Two-Stage (Detect → Crop → Classify)

Separates detection and classification into specialized models.

**Stage 1 (Detection):** Same architecture as B.1 (0.814±0.018 mAP50)

**Stage 2 (Classification on Crops):**

| Seed | Top-1 Accuracy | Top-5 Accuracy |
|:----:|:--------------:|:--------------:|
| 42 | 0.964 | 1.000 |
| 123 | 0.971 | 1.000 |
| 456 | 0.957 | 1.000 |
| 789 | 0.957 | 1.000 |
| 101 | 0.957 | 1.000 |
| **Mean±Std** | **0.961±0.006** | **1.000±0.000** |

### B Series Comparison

| Approach | Method | Detection mAP50 | Classification Metric |
|:---------|:-------|:---------------:|:---------------------:|
| B.1 End-to-End | Joint detect+classify | 0.814±0.018 | Embedded in mAP |
| B.2 Two-Stage | Separate classifier | 0.814±0.018 | **96.1% Top-1 Acc** |

**Key difference:** B.1 measures classification implicitly via 2-class mAP, while B.2 provides explicit classification accuracy on cropped regions.

---

## Key Insights

### Localization (A Series)

1. **RGB is optimal** - Best mAP50 (0.869) with simplest setup
2. **RGBD fusion underperforms** - A.3 (real, 0.842, -3.1%) and A.4b (synthetic, 0.813, -6.4%) both worse than RGB
3. **Real > Synthetic fusion** - A.3 outperforms A.4b by 3.6%
4. **Depth alone insufficient** - Both real (0.748) and synthetic (0.708) significantly worse than RGB
5. **Real > Synthetic depth-only** - Real depth outperforms synthetic by 5.7%

### Ripeness (B Series)

1. **Two-stage highly effective** - 96.1% accuracy, 100% top-5
2. **Very stable** - Classification std=0.006 across seeds
3. **Trade-off** - B.2 adds complexity but yields interpretable results

---

## Recommendations

| Scenario | Approach | Rationale |
|:---------|:---------|:----------|
| Simple localization | A.1 RGB | Best performance, simplest setup |
| No RGB camera | A.2 Real Depth | Best depth-only option |
| Ripeness classification | B.2 Two-Stage | 96.1% accuracy |
| Resource constrained | A.1 RGB | Single modality |

---

## Stability Analysis (Std Dev of mAP50)

| Experiment | Std Dev | Rating |
|:-----------|:-------:|:------:|
| A.3 RGB+Real Depth | 0.013 | Most Stable |
| A.1 RGB, B.1/B.2 | 0.018 | Very Stable |
| A.4b RGB+Synthetic | 0.023 | Stable |
| A.4a Synthetic Depth | 0.029 | Moderate |
| A.2 Real Depth | 0.038 | Least Stable |

---

## Appendix: Model Weights & Results

**Weights location:** `<exp_dir>/runs/detect/exp_<name>_seed<N>/weights/best.pt`

**Raw results:** `<exp_dir>/kaggleoutput/<exp>_results.txt`

| Experiment | Directory | Status |
|:-----------|:----------|:-------|
| A.1 | train_a1_rgb | Valid |
| A.2 | train-a2-depth | Valid |
| A.3 | train-a3-rgbd | Valid |
| A.4a | train-a4a-synthetic-depth | Valid |
| A.4b | train-a4b-rgbd-synthetic | Valid |
| B.1 | train-b1-ripeness | Valid |
| B.2 | train-b2-twostage | Valid |

---

*Report from 5-seed experiments (42, 123, 456, 789, 101)*
