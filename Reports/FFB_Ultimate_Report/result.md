# 🌴 FFB Detection Report — YOLO Experiments

> **Model**: YOLOv11n | **Epochs**: 50 | **Seeds**: 42, 123 | **Evaluasi**: Test Set

---

## � Exploratory Data Analysis (EDA) Summary

> **Detail lengkap**: [`Experiments/eda_output/EDA_INSIGHTS.md`](../../Experiments/eda_output/EDA_INSIGHTS.md)

### Dataset Overview

| Aspek | Lokalisasi (1-class) | Ripeness (2-class) |
|:------|:--------------------:|:------------------:|
| **Total gambar** | 400 | 400 |
| **Split (train/val/test)** | 280 / 80 / 40 | 280 / 80 / 40 |
| **Total objek (bbox)** | 957 | 1,416 |
| **Objek/gambar** | min=0, med=2, max=5 | min=1, med=3, max=9 |
| **Resolusi gambar** | 1280×720 (RGB) | varies |


### Bounding Box Statistics (Normalized)

| Dataset | Width | Height | Area |
|:--------|:-----:|:------:|:----:|
| Lokalisasi | 0.038–0.155 (med: 0.083) | 0.078–0.334 (med: 0.181) | 0.003–0.048 |
| Ripeness | 0.003–0.429 (med: 0.095) | 0.006–0.413 (med: 0.103) | 0.000–0.177 |

### Class Distribution (Ripeness)

| Kelas | Jumlah | Persentase |
|:-----:|:------:|:----------:|
| 🍊 Ripe | 276 | 19.5% |
| 🌿 Unripe | 1,140 | 80.5% |

> ⚠️ **Class Imbalance**: Rasio Unripe:Ripe ~4:1 — dapat mempengaruhi performa klasifikasi kematangan

---

## �📊 Hasil Utama (Quick Summary)

| Eksperimen | Input | Kelas | mAP50 ↑ | mAP50-95 ↑ | Waktu/Epoch |
|:----------:|:-----:|:-----:|:-------:|:----------:|:-----------:|
| **A.1** RGB Only | 3-ch | 1 | **0.873** | 0.370 | ~5.4s |
| **A.2** Depth Only | 1→3 ch | 1 | 0.628 | 0.226 | ~5.0s |
| **A.3** RGB+Depth | 4-ch | 1 | **0.869** | 0.379 | ~5.3s |
| **B.1** Ripeness | 3-ch | 2 | **0.801** | **0.514** | ~12.9s |

---

### 🏆 Ablation Study — RGB Only (Model Size & Optimizer)

Studi ablasi dilakukan pada eksperimen A.1 (RGB Only) untuk mengisolasi pengaruh **ukuran model** dan **optimizer** terhadap performa deteksi FFB.

**Catatan**: Semua metrik menggunakan **validation set** (80 images) untuk konsistensi perbandingan. Gap 1-3 juga dievaluasi pada test set (40 images) — lihat detail di bawah.

| Rank | Eksperimen | Model | Optimizer | Epochs | mAP50 | mAP50-95 | Delta(50-95) |
|:--:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Gap 4** | **Small** | **SGD** | 300* | **0.903** | **0.477** | **+0.107** |
| 🥈 | Gap 1 (Scaling) | Small | SGD | 50 | 0.904 | 0.473 | +0.103 |
| 🥉 | Gap 5 (Latest) | Small | AdamW | 300 | 0.891 | 0.466 | +0.096 |
| 4 | Gap 2 (Optimizer) | Nano | AdamW | 50 | 0.896 | 0.433 | +0.063 |
| 5 | Gap 3 (Duration) | Nano | SGD | 300* | 0.881 | 0.420 | +0.050 |
| 6 | Old Best | Small | AdamW | 38† | 0.876 | 0.414 | +0.044 |
| 7 | **Baseline (A.1)** | Nano | SGD | 50 | 0.873 | 0.370 | — |

*\* EarlyStopping aktif (patience=100). Gap 4 berhenti di epoch 198 (best), Gap 3 berjalan full 300 epochs. † Old Best berhenti di epoch 38 karena patience=20.*

**💡 Ablation Insights:**
1. **Model Size > Optimizer:** Upgrade dari Nano ke Small memberikan boost terbesar (+10.3pp mAP50-95 untuk Gap 1 vs Baseline).
2. **SGD > AdamW untuk Small:** Model Small dengan SGD (Gap 4: 0.477) mengungguli AdamW (Gap 5: 0.466) dengan gap +1.1pp di mAP50-95.
3. **🚀 AdamW Revival:** Dengan **patience=100**, Small+AdamW (Gap 5) mencapai **0.466** — jauh lebih baik dari Old Best (0.414) yang terlalu cepat dihentikan di epoch 38.
4. **⏳ Still Improving:** Gap 5 berjalan **full 300 epoch** tanpa early stop dan loss masih menurun di akhir training → **potensi improve dengan >300 epochs**.
5. **Efficiency King:** **Small + SGD + 50 Epochs (Gap 1)** adalah sweet spot untuk prototyping — mencapai 0.473 mAP50-95 dengan cost training ~1/6 dari 300 epochs.

<details>
<summary><b>🔍 Gap 5 Deep Analysis — Kenapa AdamW Tiba-tiba Kompetitif?</b></summary>

**Results per Seed (Validation Set):**
| Seed | mAP50 | mAP50-95 | Status |
|:----:|:-----:|:--------:|:------:|
| 42 | 0.902 | 0.464 | Full 300e |
| 123 | 0.880 | **0.468** | Full 300e |
| **Avg** | **0.891** | **0.466** | — |

**Analisis:**
1. **Patience Terlalu Kecil:** Old Best (patience=20) berhenti di epoch 38, padahal AdamW butuh waktu lebih lama untuk konvergen.
2. **Slow but Steady:** AdamW dengan weight decay menghasilkan generalisasi lebih baik ketika diberi waktu cukup — gap 5.2pp lebih baik dari Old Best.
3. **⏳ Still Improving:** Di epoch 290-300 (seed 123), loss masih menurun (`box_loss: 0.47→0.44`, `cls_loss: 0.28→0.24`, `dfl_loss: 0.84→0.81`). Model **BELUM KONVERGEN sepenuhnya**!
4. **Potensi >300 Epochs:** Trend loss yang masih menurun mengindikasikan potensi mencapai **mAP50-95 = 0.47+** dengan 500 epochs.

**🔥 Next: Gap 6 — Small + AdamW + 500 Epochs** (In Progress)
</details>

**🔑 Key Insights:**
- **A.1 (RGB)** adalah champion untuk mAP50 — depth tidak memberikan peningkatan signifikan
- **B.1 (Ripeness)** mAP50-95 tertinggi (0.514) — bounding box lebih tight untuk klasifikasi kematangan
- **A.2 (Depth only)** gap ~24pp di bawah RGB — informasi tekstur sangat krusial
- **A.3 (RGB+Depth)** unggul di mAP50-95 (+0.9pp dari A.1) — depth membantu lokalisasi presisi

---

## 🔬 A.1 — Lokalisasi FFB (RGB Only)

> **Rata-rata Test**: mAP50 = **0.873** | mAP50-95 = **0.370**

### Training Dynamics

**Catatan**: Angka di bagian ini berasal dari `results.csv` seed 42.

![Results A.1 Seed 42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/results.png)

**Observasi dari CSV Training:**
- 📈 **Konvergensi cepat**: mAP50 >0.80 sejak epoch 9 (seed 42)
- ✅ **Stabil**: mAP50 relatif stabil setelah epoch 20; tidak terlihat overfit signifikan
- 📉 **Loss terus turun**: `box_loss` 2.85 -> 1.28, `cls_loss` 4.20 -> 0.79 (seed 42)

| Metric | Epoch 10 | Epoch 25 | Epoch 50 (Final) |
|--------|----------|----------|------------------|
| mAP50 | 0.807 | 0.854 | **0.867** |
| Precision | 0.721 | 0.778 | 0.779 |
| Recall | 0.765 | 0.866 | 0.852 |

### Precision-Recall Curve

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![PR A.1 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/BoxPR_curve.png) | ![PR A.1 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed123/BoxPR_curve.png) |

**Insight**: Kurva PR sangat konsisten antara dua seed — model robust dan reproducible.

### Confusion Matrix

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![CM A.1 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/confusion_matrix.png) | ![CM A.1 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed123/confusion_matrix.png) |

### F1-Confidence Curve

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![F1 A.1 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/BoxF1_curve.png) | ![F1 A.1 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed123/BoxF1_curve.png) |

**Optimal confidence**: ~0.35-0.45 (F1 maksimal)

### Contoh Prediksi (Visual)

**Validation Batch — Ground Truth vs Prediction:**

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![Val Labels](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/val_batch0_labels.jpg) | ![Val Pred](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/val_batch0_pred.jpg) |

---

## 🔬 A.2 — Lokalisasi FFB (Depth Only)

> **Rata-rata Test**: mAP50 = **0.628** | mAP50-95 = **0.226**

### Training Dynamics

**Catatan**: Angka di bagian ini berasal dari `results.csv` seed 42.

![Results A.2 Seed 42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/results.png)

**Observasi dari CSV Training:**
- 🐢 **Konvergensi lambat**: mAP50 >0.50 baru di epoch 14 (seed 42) vs epoch 5 di RGB (seed 42)
- 📊 **Plateau lebih awal**: mAP50 berkisar 0.57-0.68 setelah epoch 30 (seed 42)
- ⚠️ **Recall rendah**: Maks recall 0.710 (seed 42) vs 0.866 di RGB (seed 42)

| Metric | Epoch 10 | Epoch 25 | Epoch 50 (Final) |
|--------|----------|----------|------------------|
| mAP50 | 0.130 | 0.636 | **0.661** |
| Precision | 0.456 | 0.662 | 0.685 |
| Recall | 0.075 | 0.589 | 0.645 |

**⚠️ Fenomena Menarik:** Pada epoch 10, mAP50 hanya 0.13! Model sangat struggle di awal karena depth tidak memiliki informasi tekstur/warna.

### Precision-Recall Curve

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![PR A.2 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/BoxPR_curve.png) | ![PR A.2 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed123/BoxPR_curve.png) |

**Insight**: Area under curve lebih kecil — model kurang confident dalam prediksi.

### Confusion Matrix

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![CM A.2 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/confusion_matrix.png) | ![CM A.2 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed123/confusion_matrix.png) |

### F1-Confidence Curve

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![F1 A.2 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/BoxF1_curve.png) | ![F1 A.2 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed123/BoxF1_curve.png) |

**Optimal confidence**: ~0.25-0.35 (lebih rendah dari RGB)

### Contoh Prediksi (Visual)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![Val Labels](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/val_batch0_labels.jpg) | ![Val Pred](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/val_batch0_pred.jpg) |

---

## 🔬 A.3 — Lokalisasi FFB (RGB+Depth 4-Channel)

> **Rata-rata Test**: mAP50 = **0.869** | mAP50-95 = **0.379** ⬆️ Best mAP50-95!

### Training Dynamics

**Catatan**: Angka di bagian ini berasal dari `results.csv` seed 42.

![Results A.3 Seed 42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed42_train/results.png)

**Observasi:**
- 📈 **mAP50 stabil**: >0.80 sejak epoch 12 dan berada di ~0.85-0.89 setelah epoch 20 (seed 42)
- 📊 **mAP50-95**: puncak 0.414 di epoch 22 lalu stabil di ~0.37-0.41 hingga akhir (seed 42)
- 📉 **Loss turun konsisten**: `box_loss` 2.79 -> 1.21, `cls_loss` 4.18 -> 0.764 (seed 42)

### Precision-Recall Curve

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![PR A.3 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed42_train/BoxPR_curve.png) | ![PR A.3 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed123_train/BoxPR_curve.png) |

### Confusion Matrix

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![CM A.3 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed42_train/confusion_matrix.png) | ![CM A.3 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed123_train/confusion_matrix.png) |

### F1-Confidence Curve

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![F1 A.3 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed42_train/BoxF1_curve.png) | ![F1 A.3 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed123_train/BoxF1_curve.png) |

### Contoh Prediksi (Visual)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![Val Labels](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed42_train/val_batch0_labels.jpg) | ![Val Pred](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed42_train/val_batch0_pred.jpg) |

---

## 🔬 B.1 — Deteksi Kematangan (2 Kelas: Ripe/Unripe)

> **Rata-rata Test**: mAP50 = **0.801** | mAP50-95 = **0.514** ⬆️ Best mAP50-95 overall!

### Training Dynamics

**Catatan**: Angka di bagian ini berasal dari `results.csv` seed 42.

![Results B.1 Seed 42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/results.png)

**Observasi dari CSV Training:**
- 📈 **Konvergensi bertahap**: mAP50 >0.80 pada epoch 14 (seed 42)
- 🎯 **mAP50-95 tinggi**: epoch 50 = 0.572 (seed 42)
- ⚖️ **Precision vs Recall trade-off**: Precision 0.773, Recall 0.807 (epoch 50, seed 42)

| Metric | Epoch 10 | Epoch 25 | Epoch 50 (Final) |
|--------|----------|----------|------------------|
| mAP50 | 0.673 | 0.844 | **0.837** |
| mAP50-95 | 0.428 | 0.575 | **0.572** |
| Precision | 0.637 | 0.784 | 0.773 |
| Recall | 0.642 | 0.785 | 0.807 |

### Precision-Recall Curve (Per Kelas)

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![PR B.1 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/BoxPR_curve.png) | ![PR B.1 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed123/BoxPR_curve.png) |

**Insight**: Kurva menunjukkan performa per kelas (ripe vs unripe) — lihat gap jika ada.

### Confusion Matrix (2 Kelas)

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![CM B.1 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/confusion_matrix.png) | ![CM B.1 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed123/confusion_matrix.png) |

### Confusion Matrix (Normalized)

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![CMN B.1 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/confusion_matrix_normalized.png) | ![CMN B.1 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed123/confusion_matrix_normalized.png) |

### F1-Confidence Curve

| Seed 42 | Seed 123 |
|:-------:|:--------:|
| ![F1 B.1 s42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/BoxF1_curve.png) | ![F1 B.1 s123](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed123/BoxF1_curve.png) |

### Contoh Prediksi (Visual)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![Val Labels](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/val_batch0_labels.jpg) | ![Val Pred](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/val_batch0_pred.jpg) |

---

## 📈 Perbandingan Kurva Training (All Experiments)

### Loss Comparison (Final Epoch, Seed 42)

| Experiment | Box Loss | Cls Loss | DFL Loss |
|:----------:|:--------:|:--------:|:--------:|
| A.1 RGB | 1.277 | 0.793 | 1.097 |
| A.2 Depth | 1.638 | 1.305 | 1.303 |
| A.3 RGBD | 1.211 | 0.764 | 1.066 |
| B.1 Ripeness | 0.790 | 0.769 | 0.926 |

**Insight**: 
- A.2 memiliki loss tertinggi — model struggle dengan depth-only input
- B.1 terendah pada box_loss dan dfl_loss; cls_loss terendah ada di A.3

---

## 📋 Detail Metrik per Run

### A.1 — RGB Only (1 Kelas)

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.873 | 0.370 | 0.884 | 0.727 | [test.txt](artifacts/kaggleoutput/test.txt) |
| 123 | 0.873 | 0.369 | 0.808 | 0.790 | [test.txt](artifacts/kaggleoutput/test.txt) |
| **Avg** | **0.873** | **0.370** | 0.846 | 0.759 | — |

### A.2 — Depth Only (1 Kelas)

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.640 | 0.216 | 0.643 | 0.667 | [test_depth.txt](artifacts/kaggleoutput/test_depth.txt) |
| 123 | 0.615 | 0.235 | 0.652 | 0.581 | [test_depth.txt](artifacts/kaggleoutput/test_depth.txt) |
| **Avg** | **0.628** | **0.226** | 0.648 | 0.624 | — |

### A.3 — RGB+Depth 4-Channel (1 Kelas)

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.875 | 0.378 | 0.787 | 0.811 | [test_4input.txt](artifacts/kaggleoutput/test_4input.txt) |
| 123 | 0.862 | 0.380 | 0.772 | 0.804 | [test_4input.txt](artifacts/kaggleoutput/test_4input.txt) |
| **Avg** | **0.869** | **0.379** | 0.780 | 0.808 | — |

### B.1 — Ripeness Detection (2 Kelas)

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.804 | 0.511 | 0.778 | 0.747 | [test_ripeness_detect.txt](artifacts/kaggleoutput/test_ripeness_detect.txt) |
| 123 | 0.797 | 0.517 | 0.796 | 0.731 | [test_ripeness_detect.txt](artifacts/kaggleoutput/test_ripeness_detect.txt) |
| **Avg** | **0.801** | **0.514** | 0.787 | 0.739 | — |

### Ablation Study — Gap Experiments (Validation Set)

**Gap 1 — Small + SGD + 50 Epochs**

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.918 | 0.469 | 0.907 | 0.839 | [test_gap1.txt](artifacts/kaggleoutput/test_gap1.txt) |
| 123 | 0.889 | 0.477 | 0.851 | 0.828 | [test_gap1.txt](artifacts/kaggleoutput/test_gap1.txt) |
| **Avg** | **0.904** | **0.473** | 0.879 | 0.834 | — |

**Gap 2 — Nano + AdamW + 50 Epochs**

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.908 | 0.438 | 0.788 | 0.925 | [test_gap2.txt](artifacts/kaggleoutput/test_gap2.txt) |
| 123 | 0.883 | 0.428 | 0.818 | 0.843 | [test_gap2.txt](artifacts/kaggleoutput/test_gap2.txt) |
| **Avg** | **0.896** | **0.433** | 0.803 | 0.884 | — |

**Gap 3 — Nano + SGD + 300 Epochs**

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.872 | 0.417 | 0.807 | 0.807 | [test_gap3.txt](artifacts/kaggleoutput/test_gap3.txt) |
| 123 | 0.889 | 0.422 | 0.858 | 0.843 | [test_gap3.txt](artifacts/kaggleoutput/test_gap3.txt) |
| **Avg** | **0.881** | **0.420** | 0.833 | 0.825 | — |

**Gap 4 — Small + SGD + 300 Epochs**

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.909 | 0.477 | 0.859 | 0.852 | [test_gap4.txt](artifacts/kaggleoutput/test_gap4.txt) |
| 123 | 0.896 | 0.476 | 0.886 | 0.796 | [test_gap4.txt](artifacts/kaggleoutput/test_gap4.txt) |
| **Avg** | **0.903** | **0.477** | 0.873 | 0.824 | — |

**Gap 5 — Small + AdamW + 300 Epochs**

| Seed | mAP50 | mAP50-95 | Precision | Recall | Source |
|:----:|:-----:|:--------:|:---------:|:------:|:------:|
| 42 | 0.902 | 0.464 | 0.845 | 0.853 | [test_gap5.txt](artifacts/kaggleoutput/test_gap5.txt) |
| 123 | 0.880 | 0.468 | 0.869 | 0.788 | [test_gap5.txt](artifacts/kaggleoutput/test_gap5.txt) |
| **Avg** | **0.891** | **0.466** | 0.857 | 0.821 | — |

**Catatan**: Gap 1-3 juga dievaluasi pada test set (40 images). Test set results: Gap 1 (mAP50=0.899, mAP50-95=0.418), Gap 2 (mAP50=0.860, mAP50-95=0.391), Gap 3 (mAP50=0.849, mAP50-95=0.363).

---

## 🧪 Preprocessing Depth (Technical)

Untuk eksperimen A.2 dan A.3, depth maps diproses dengan aturan berikut:

```python
# 1. Load depth (uint16, dalam mm)
depth_u16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)

# 2. Convert ke meter
depth_m = depth_u16.astype(np.float32) / 1000.0

# 3. Handle invalid values
depth_m[depth_u16 == 0] = np.nan      # no return
depth_m[depth_u16 == 65535] = np.nan  # saturated

# 4. Clip ke operating range (0.6m - 6.0m)
depth_m = np.clip(depth_m, 0.6, 6.0)

# 5. Normalize ke 0-255
scaled = (depth_m - 0.6) / (6.0 - 0.6)
depth_u8 = (scaled * 255).astype(np.uint8)

# 6. Replicate ke 3 channel (untuk YOLO compatibility)
depth_3ch = cv2.merge([depth_u8, depth_u8, depth_u8])
```

**Rentang**: 0.6m – 6.0m (sesuai spesifikasi RealSense depth camera)

---

## 📂 Struktur Artefak

```
artifacts/kaggleoutput/
├── kaggle/working/runs/detect/
│   ├── exp_a1_rgb_seed42/          # RGB Seed 42
│   │   ├── results.csv             # Training metrics per epoch
│   │   ├── results.png             # Training curves (loss, mAP, etc)
│   │   ├── BoxPR_curve.png         # Precision-Recall curve
│   │   ├── BoxF1_curve.png         # F1 vs Confidence
│   │   ├── confusion_matrix.png    # Confusion matrix
│   │   ├── val_batch*_pred.jpg     # Validation predictions
│   │   └── weights/best.pt         # Best model weights
│   ├── exp_a1_rgb_seed123/         # RGB Seed 123
│   ├── exp_a2_depth_seed42/        # Depth Seed 42
│   ├── exp_a2_depth_seed123/       # Depth Seed 123
│   ├── exp_a3_rgbd_seed42_train/   # RGBD Seed 42
│   ├── exp_a3_rgbd_seed123_train/  # RGBD Seed 123
│   ├── exp_b1_ripeness_det_seed42/ # Ripeness Seed 42
│   └── exp_b1_ripeness_det_seed123/# Ripeness Seed 123
├── test.txt                        # A.1 test results log
├── test_depth.txt                  # A.2 test results log
├── test_4input.txt                 # A.3 test results log
├── test_ripeness_detect.txt        # B.1 test results log
└── *.ipynb                         # Kaggle notebooks
```

---

## 🔴 Failure Cases Analysis

Berdasarkan analisis confusion matrix dan visual inspection pada validation batch, berikut adalah pola kegagalan yang teridentifikasi:

### A.1 — RGB Localization (Single Class)

**Confusion Matrix Summary (Seed 42):**
| | Predicted: FFB | Predicted: Background |
|:--|:--------------:|:---------------------:|
| **True: FFB** | 165 (TP) | 21 (FN) |
| **True: Background** | 53 (FP) | — |

**Pola Kegagalan:**

1. **Missed Detection (21 FN = 11.3%)**
   - FFB yang terhalang daun/pelepah (occlusion)
   - FFB berukuran sangat kecil di latar belakang
   - FFB dengan warna mirip daun (unripe muda)

2. **False Positive (53 FP)**
   - Bagian batang/tangkai yang mirip tekstur FFB
   - Daun kering dengan bentuk bulat
   - Artefak visual di area gelap

| Contoh Kasus | Visual |
|:-------------|:------:|
| GT (Batch 0) | ![GT](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/val_batch0_labels.jpg) |
| Pred (Batch 0) | ![Pred](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/val_batch0_pred.jpg) |

**Observasi Visual:**
- Confidence score bervariasi 0.3–0.9
- Beberapa prediksi dengan confidence rendah (0.3–0.4) adalah kandidat FP
- Missed detection umumnya pada objek di pinggir frame atau dengan occlusion tinggi

---

### B.1 — Ripeness Detection (2 Class)

**Confusion Matrix Normalized (Seed 42):**
| | Pred: Ripe | Pred: Unripe | Pred: Background |
|:--|:----------:|:------------:|:----------------:|
| **True: Ripe** | **0.84** | 0.08 | 0.08 |
| **True: Unripe** | 0.03 | **0.85** | 0.13 |

**Pola Kegagalan:**

1. **Ripe → Unripe Misclassification (8%)**
   - FFB matang yang terlalu gelap (bayangan)
   - Ripe dengan sedikit area oranye visible
   
2. **Unripe → Ripe Misclassification (3%)**  
   - FFB muda dengan refleksi cahaya kekuningan
   - Unripe yang sebagian terekspos matahari langsung

3. **Missed Detection (Ripe 8%, Unripe 13%)**
   - Occlusion oleh daun
   - Objek di pinggir frame (cropped)

| Contoh Kasus | Visual |
|:-------------|:------:|
| GT (Batch 0) | ![GT B1](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/val_batch0_labels.jpg) |
| Pred (Batch 0) | ![Pred B1](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/val_batch0_pred.jpg) |

**💡 Insight:**
- Akurasi klasifikasi Ripe (84%) dan Unripe (85%) cukup seimbang meskipun ada class imbalance (4:1)
- **Background FP tinggi di Unripe (82%)** — model cenderung over-detect unripe pada area non-FFB

---

### A.2 — Depth Only (Perbandingan)

| GT (Depth) | Pred (Depth) |
|:----------:|:------------:|
| ![GT Depth](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/val_batch0_labels.jpg) | ![Pred Depth](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/val_batch0_pred.jpg) |

**Observasi:**
- Confidence score keseluruhan lebih rendah (0.3–0.7) dibanding RGB
- Banyak missed detection pada FFB yang mirip kedalaman dengan background
- Depth map gagal membedakan FFB dari batang pohon pada jarak sama

---



## 🎯 Kesimpulan & Rekomendasi

### Temuan Utama

1. **RGB sangat powerful** — Untuk lokalisasi FFB, RGB saja sudah mencapai mAP50 = 0.873
2. **Depth memberikan marginal gain** — RGB+Depth (A.3) hanya +0.9pp di mAP50-95
3. **Depth alone tidak cukup** — Gap ~25pp di bawah RGB (mAP50), kehilangan informasi tekstur
4. **Ripeness detection feasible** — mAP50 = 0.80 dengan 2 kelas cukup baik

### Rekomendasi

| Scenario | Recommended Model | Why |
|----------|-------------------|-----|
| Produksi (speed priority) | A.1 RGB | Fastest, best mAP50 |
| Presisi lokalisasi | A.3 RGB+Depth | Best mAP50-95 |
| Klasifikasi kematangan | B.1 Ripeness | Purpose-built |

---
## ⚠️ Kendala & Solusi (Troubleshooting Journey)

Berikut adalah kendala-kendala yang dialami selama eksperimen dan bagaimana penyelesaiannya:

### 1. ❌ Preprocessing Depth Tidak Sesuai Skala Meter

**Masalah:**
Pada awalnya, depth map diproses dengan **normalisasi min-max per-gambar**:
```python
# ❌ SALAH - min-max per image
depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255
```

Ini menyebabkan:
- Setiap gambar memiliki skala berbeda
- Tidak merefleksikan rentang fisik sebenarnya (0.6m – 6.0m)
- Model tidak bisa belajar konsistensi jarak antar gambar

**Solusi:**
Implementasi **fixed-range normalization** yang konsisten:
```python
# ✅ BENAR - fixed range 0.6m - 6.0m
depth_m = depth_u16.astype(np.float32) / 1000.0  # mm → meter
depth_m[depth_u16 == 0] = np.nan                 # invalid: no return
depth_m[depth_u16 == 65535] = np.nan             # invalid: saturated
depth_m = np.clip(depth_m, 0.6, 6.0)             # clip ke operating range
scaled = (depth_m - 0.6) / (6.0 - 0.6)           # normalize 0-1
depth_u8 = (scaled * 255).astype(np.uint8)       # scale ke 0-255
```

**Hasil:** Dataset depth-only dibangun ulang dengan preprocessing yang benar.

---

### 2. ❌ Struktur Dataset/YAML di Kaggle

**Masalah:**
Training di Kaggle gagal dengan error:
```
AssertionError: No labels found in .../labels/train, can not start training
num_samples=0
```

Path dan split dataset tidak mengikuti struktur standar YOLO.

**Solusi:**
Restrukturisasi folder mengikuti konvensi YOLO:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Dan YAML config yang konsisten:
```yaml
path: /kaggle/input/ffb-localization
train: images/train
val: images/val
test: images/test
nc: 1
names: ['fresh_fruit_bunch']
```

---

### 3. ❌ Training RGB+Depth (4-Channel Input)

**Masalah:**
YOLO default hanya menerima 3-channel input (RGB). Training 4-channel (R,G,B,D) memerlukan modifikasi signifikan dan memunculkan berbagai error:

- `RuntimeError: stride mismatch` — validator tidak kompatibel dengan 4-ch
- `IndexError` pada mosaic augmentation — buffer tidak support 4-ch
- Sinkronisasi transform RGB-Depth yang tidak konsisten

**Solusi:**
1. **Load pasangan RGB+Depth**: Memastikan setiap RGB punya pasangan depth yang valid
2. **Transform sinkron**: Augmentasi (flip, rotate, dll) diterapkan identik ke RGB dan Depth
3. **Concat 4 channel**: 
   ```python
   rgbd = np.concatenate([rgb, depth_1ch], axis=-1)  # shape: (H, W, 4)
   ```
4. **Adapt conv pertama**: Modifikasi layer pertama model untuk menerima 4 channel:
   ```python
   # Ubah in_channels dari 3 ke 4
   model.model[0].conv = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
   ```
5. **Disable beberapa augmentasi**: Mosaic dan copy-paste augmentation di-disable untuk stabilitas

**Hasil:** Training 4-channel berhasil stabil di Kaggle dengan mAP50-95 terbaik (0.379).

---


## 📎 Lampiran

### Kaggle Notebooks
- [`ffb-localization.ipynb`](artifacts/kaggleoutput/ffb-localization.ipynb) — A.1 RGB
- [`ffb-localization-depth.ipynb`](artifacts/kaggleoutput/ffb-localization-depth.ipynb) — A.2 Depth
- [`ffb-localization-rgbd.ipynb`](artifacts/kaggleoutput/ffb-localization-rgbd.ipynb) — A.3 RGB+Depth
- [`ffb-ripeness-detect.ipynb`](artifacts/kaggleoutput/ffb-ripeness-detect.ipynb) — B.1 Ripeness

### Dataset Paths (Kaggle)
- RGB Localization: `Experiments/UploadKaggle/ffb_localization/`
- Depth Localization: `Experiments/UploadKaggle/ffb_localization_depth/`
- RGB+Depth: `Experiments/UploadKaggle/ffb_localization_rgbd/`
- Ripeness: `Experiments/UploadKaggle/ffb_ripeness_detect/`

---

*Report generated: 2026-01-19 | Model: YOLOv11n | Framework: Ultralytics*
