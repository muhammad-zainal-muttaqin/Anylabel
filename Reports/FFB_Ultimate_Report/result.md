# 🌴 FFB Detection Report — YOLO Experiments

> **Model**: YOLOv11n | **Epochs**: 50 | **Seeds**: 42, 123 | **Evaluasi**: Test Set

---

## 📊 Hasil Utama (Quick Summary)

| Eksperimen | Input | Kelas | mAP50 ↑ | mAP50-95 ↑ | Waktu/Epoch |
|:----------:|:-----:|:-----:|:-------:|:----------:|:-----------:|
| **A.1** RGB Only | 3-ch | 1 | **0.873** | 0.370 | ~5.4s |
| **A.2** Depth Only | 1→3 ch | 1 | 0.628 | 0.226 | ~5.0s |
| **A.3** RGB+Depth | 4-ch | 1 | **0.869** | 0.379 | ~5.3s |
| **B.1** Ripeness | 3-ch | 2 | **0.801** | **0.514** | ~12.9s |

**🔑 Key Insights:**
- **A.1 (RGB)** adalah champion untuk mAP50 — depth tidak memberikan peningkatan signifikan
- **B.1 (Ripeness)** mAP50-95 tertinggi (0.514) — bounding box lebih tight untuk klasifikasi kematangan
- **A.2 (Depth only)** gap ~24pp di bawah RGB — informasi tekstur sangat krusial
- **A.3 (RGB+Depth)** unggul di mAP50-95 (+0.9pp dari A.1) — depth membantu lokalisasi presisi

---

## 🔬 A.1 — Lokalisasi FFB (RGB Only)

> **Rata-rata Test**: mAP50 = **0.873** | mAP50-95 = **0.370**

### Training Dynamics

![Results A.1 Seed 42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a1_rgb_seed42/results.png)

**Observasi dari CSV Training:**
- 📈 **Konvergensi cepat**: mAP50 mencapai >0.80 pada epoch 12-13
- ✅ **Stabil**: Fluktuasi minimal setelah epoch 20, model tidak overfit
- 📉 **Loss terus turun**: `box_loss` dari 2.85 → 1.28, `cls_loss` dari 4.20 → 0.79

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

![Results A.2 Seed 42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a2_depth_seed42/results.png)

**Observasi dari CSV Training:**
- 🐢 **Konvergensi lambat**: mAP50 baru >0.50 pada epoch 14-16 (vs epoch 5-6 di RGB)
- 📊 **Plateau lebih awal**: Model stuck di ~0.66 setelah epoch 30
- ⚠️ **Recall rendah**: Maksimal ~0.71 vs ~0.87 di RGB

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

![Results A.3 Seed 42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_a3_rgbd_seed42_train/results.png)

**Observasi:**
- ✅ **Kombinasi terbaik**: mAP50-95 tertinggi di semua eksperimen lokalisasi
- 📈 **Konvergensi mirip RGB**: Depth tidak memperlambat training
- 🎯 **Presisi lebih baik**: Bounding box lebih akurat (mAP50-95 ↑)

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

![Results B.1 Seed 42](artifacts/kaggleoutput/kaggle/working/runs/detect/exp_b1_ripeness_det_seed42/results.png)

**Observasi dari CSV Training:**
- 📈 **Konvergensi bertahap**: mAP50 >0.80 pada epoch 14-15
- 🎯 **mAP50-95 tinggi**: Rata-rata 0.514 — bbox sangat akurat untuk klasifikasi
- ⚖️ **Precision vs Recall trade-off**: Precision ~0.77, Recall ~0.81

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

### Loss Comparison (Final Epoch)

| Experiment | Box Loss | Cls Loss | DFL Loss |
|:----------:|:--------:|:--------:|:--------:|
| A.1 RGB | 1.277 | 0.793 | 1.097 |
| A.2 Depth | 1.638 | 1.305 | 1.303 |
| A.3 RGBD | ~1.3 | ~0.85 | ~1.1 |
| B.1 Ripeness | 0.790 | 0.769 | 0.926 |

**Insight**: 
- A.2 memiliki loss tertinggi — model struggle dengan depth-only input
- B.1 memiliki loss terendah — task classification lebih "mudah" dengan 2 kelas

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
