# FFB Detection Report - Summary

> **Latest Update:** 2026-01-29 | **Model:** YOLOv11n | **Evaluation:** Test Set

---

## Quick Results (V2 - 5 Seeds, Uniform Augmentation)

### A Series: Localization (1-Class Detection)

| Rank | Experiment | Input | mAP50 | mAP50-95 | Note |
|:----:|:-----------|:------|:-----:|:--------:|:-----|
| 1 | **A.3** | RGB + Real Depth (4-ch) | **0.840** | 0.369 | Best overall |
| 2 | A.1 | RGB Only | 0.839 | 0.365 | Baseline |
| 3 | A.4b | RGB + Synthetic Depth | 0.823 | 0.368 | No sensor needed |
| 4 | A.5 | Late Fusion | 0.808 | 0.318 | Dual backbone |
| 5 | A.2 | Real Depth Only | 0.733 | 0.292 | + BN Reset |
| 6 | A.4a | Synthetic Depth Only | 0.653 | 0.275 | Depth-Anything-V2 |

### B Series: Ripeness Classification (2-Class Detection)

| Experiment | Approach | mAP50 | mAP50-95 | Note |
|:-----------|:---------|:-----:|:--------:|:-----|
| **B.2** | Two-Stage (Detect + Classify) | - | - | **96.1% accuracy** |
| B.1 | End-to-End Detection | 0.801 | 0.514 | 2-seed baseline |

---

## Key Findings

| Finding | Details |
|:--------|:--------|
| **Best Localization** | A.3 RGB+Depth (0.840 mAP50) |
| **RGB is Strong** | A.1 RGB-only nearly matches A.3 (0.839) |
| **Real > Synthetic** | Real depth (0.733) > Synthetic (0.653) by 7.9% |
| **RGBD Fusion** | A.3 slightly better than RGB (+0.1%) |
| **No Sensor Option** | A.4b (0.823) viable alternative |
| **Best Ripeness** | B.2 Two-Stage achieves 96.1% classification |

---

## Ablation Study (A.1 RGB - 2 Seeds)

| Rank | Config | Model | Optimizer | Epochs | mAP50 | mAP50-95 |
|:----:|:-------|:-----:|:---------:|:------:|:-----:|:--------:|
| 1 | **Ablation 4** | Small | SGD | 300 | 0.875 | **0.433** |
| 2 | Ablation 1 | Small | SGD | 50 | 0.899 | 0.418 |
| 3 | Ablation 2 | Nano | AdamW | 50 | 0.860 | 0.391 |
| 4 | Ablation 5 | Small | AdamW | 300 | 0.833 | 0.374 |
| 5 | Baseline | Nano | SGD | 50 | 0.873 | 0.370 |
| 6 | Ablation 3 | Nano | SGD | 300 | 0.849 | 0.363 |

**Insight:** Small + SGD + 300 epochs = best mAP50-95 (0.433)

---

## Recommendations

| Scenario | Recommended | Why |
|:---------|:------------|:----|
| Production (speed) | A.1 RGB | Simplest, near-best performance |
| Best precision | A.3 RGB+Depth | Highest mAP50 |
| No depth sensor | A.4b RGB+Synthetic | Only -1.7% vs RGB |
| Ripeness task | B.2 Two-Stage | 96.1% accuracy |
| Maximum mAP50-95 | Ablation 4 | 0.433 (best ever) |

---

## Experiment Versions

| Version | Seeds | Epochs | Key Feature | Report |
|:--------|:-----:|:------:|:------------|:-------|
| V0 (Baseline) | 2 | 50 | Initial experiments | `result_full.md` |
| V1 (5-seed) | 5 | 100 | Extended seeds | `v1_5seed_basic/` |
| **V2 (5-seed)** | 5 | 100 | Uniform aug + BN Reset | `v2_5seed_uniform_aug/` |

---

## File Structure

```
Reports/FFB_Ultimate_Report/
├── result_summary.md          # This file (quick reference)
├── result_full.md             # Detailed report with visuals
├── artifacts/kaggleoutput/
│   ├── exp_a1_rgb_test.txt
│   ├── exp_a2_depth_test.txt
│   ├── exp_a3_rgbd_test.txt
│   ├── exp_b1_ripeness_test.txt
│   ├── ablation1_small_sgd_50e_test.txt
│   ├── ablation2_nano_adamw_50e_test.txt
│   ├── ablation3_nano_sgd_300e_test.txt
│   ├── ablation4_small_sgd_300e_test.txt
│   ├── ablation5_small_adamw_300e_test.txt
│   ├── v1_5seed_basic/        # 5-seed V1 results
│   └── v2_5seed_uniform_aug/  # 5-seed V2 results (latest)
```

---

*Summary from V2 experiments (5 seeds: 42, 123, 456, 789, 101)*
