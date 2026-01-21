# Panduan Eksperimen V3 — Eksperimen Lanjutan

> **Status**: 🚧 DALAM PROSES
> **Tanggal Mulai**: 2026-01-22
> **Prerequisite**: Eksperimen A.1, A.2, A.3 (lama), B.1 sudah selesai (lihat EXPERIMENT_GUIDE_V2.md)

Dokumen ini mencakup **4 eksperimen baru** yang akan dikerjakan sebagai lanjutan penelitian FFB detection.

---

## 🎯 Ringkasan Eksperimen Baru

| ID | Nama | Deskripsi | Status | Output File |
|:--:|:-----|:----------|:------:|:------------|
| **A.3** | RGBD Fix | RGB+Depth dengan augmentasi yang diperbaiki | ⬜ | `test_rgbd_fixed.txt` |
| **A.4a** | Synthetic Depth Only | Depth sintetis (Depth-Anything-V2) saja | ⬜ | `test_synthetic_depth.txt` |
| **A.4b** | RGB+Synthetic Depth | RGB + Depth sintetis (4-channel) | ⬜ | `test_rgbd_synthetic.txt` |
| **B.2** | Two-Stage Classification | Detect → Crop → Classify | ⬜ | `test_twostage.txt` |

**Legend**: ⬜ Belum dikerjakan | 🚧 Sedang berjalan | ✅ Selesai

---

## 📋 A.3 — RGBD Fix (Augmentasi Diperbaiki)

### Masalah dari A.3 Lama
- Augmentasi HSV di-disable untuk depth channel
- Inconsistent augmentation antara RGB dan Depth
- Hasil tidak optimal karena augmentasi tidak tersingkron

### Yang Akan Dikerjakan
- [x] **Notebook**: `notebooks/train_a3_rgbd_fix.ipynb`
- [x] Config sudah ada: `ffb_localization_rgbd.yaml`
- [x] Custom dataloader: `custom_rgbd_dataset.py` (augmentasi tersingkron)
- [ ] **Training Run 1** (seed 42)
- [ ] **Training Run 2** (seed 123)
- [ ] **Evaluasi Test Set** (simpan ke `test_rgbd_fixed.txt`)
- [ ] **Bandingkan** dengan A.3 lama dan A.1 (RGB only)

### Expected Output
```
kaggleoutput/
├── runs/detect/
│   ├── exp_a3_fixed_seed42/
│   │   ├── weights/best.pt
│   │   ├── results.csv
│   │   ├── BoxPR_curve.png
│   │   └── confusion_matrix.png
│   └── exp_a3_fixed_seed123/
└── test_rgbd_fixed.txt
```

### Metrik Target
- **Hipotesis**: mAP50-95 > A.3 lama (0.379) karena augmentasi lebih baik
- **Baseline**: A.1 RGB = 0.873 (mAP50), 0.370 (mAP50-95)

---

## 📋 A.4a — Synthetic Depth Only

### Konsep
Menggunakan **Depth-Anything-V2** untuk generate synthetic depth dari RGB, kemudian training dengan depth sintetis saja (3-channel).

### Tujuan
- Apakah synthetic depth bisa menggantikan real depth sensor?
- Bandingkan dengan A.2 (Real Depth Only)

### Yang Akan Dikerjakan
- [x] **Notebook generation**: `notebooks/generate_synthetic_depth.ipynb` (20-30 min GPU)
- [x] **Notebook training**: `notebooks/train_a4a_synthetic_depth.ipynb`
- [x] Config: `ffb_localization_depth_synthetic.yaml`
- [ ] **Generate synthetic depth** dari semua RGB images (run notebook)
- [ ] **Organize dataset** dengan `scripts/prepare_synthetic_depth_data.py`
- [ ] **Training Run 1** (seed 42)
- [ ] **Training Run 2** (seed 123)
- [ ] **Evaluasi Test Set** (simpan ke `test_synthetic_depth.txt`)
- [ ] **Analisis komparasi** A.2 vs A.4a (script: `compare_real_vs_synthetic.py`)

### Expected Output
```
datasets/
├── depth_synthetic_da2/        # Synthetic depth maps (3-channel)
│   ├── train/
│   ├── val/
│   └── test/
kaggleoutput/
├── runs/detect/
│   ├── exp_a4a_synthetic_seed42/
│   └── exp_a4a_synthetic_seed123/
├── test_synthetic_depth.txt
└── comparison_real_vs_synthetic.md
```

### Metrik Target
- **Hipotesis**: mAP50 < A.2 Real Depth (0.628) karena synthetic kurang akurat
- **Tapi**: Jika gap kecil (<5pp), synthetic depth viable untuk deployment

---

## 📋 A.4b — RGB+Synthetic Depth (4-Channel)

### Konsep
Fusion RGB + Synthetic Depth (4-channel) seperti A.3, tapi menggunakan depth sintetis.

### Tujuan
- Apakah RGB+Synthetic Depth bisa match RGB+Real Depth (A.3)?
- Eliminasi kebutuhan depth sensor fisik

### Yang Akan Dikerjakan
- [x] **Notebook**: `notebooks/train_a4b_rgbd_synthetic.ipynb`
- [x] Config: `ffb_localization_rgbd_synthetic.yaml`
- [ ] **Pastikan synthetic depth sudah di-generate** (dari A.4a)
- [ ] **Training Run 1** (seed 42)
- [ ] **Training Run 2** (seed 123)
- [ ] **Evaluasi Test Set** (simpan ke `test_rgbd_synthetic.txt`)
- [ ] **Bandingkan** dengan A.3 (Real RGBD) dan A.1 (RGB only)

### Expected Output
```
kaggleoutput/
├── runs/detect/
│   ├── exp_a4b_rgbd_synthetic_seed42/
│   └── exp_a4b_rgbd_synthetic_seed123/
└── test_rgbd_synthetic.txt
```

### Metrik Target
- **Hipotesis**: mAP50-95 sedikit < A.3 Real RGBD (0.379) tapi > A.1 RGB (0.370)
- **Jika berhasil**: Bisa deploy tanpa depth sensor fisik

---

## 📋 B.2 — Two-Stage Ripeness Classification

### Konsep
**Pipeline Dua Tahap**:
1. **Stage 1**: Detect semua FFB (ripe/unripe) dengan detector
2. **Crop** bounding box dengan margin 10%
3. **Stage 2**: Classify crops dengan classifier khusus

### Tujuan
- Apakah two-stage lebih akurat dari B.1 (end-to-end)?
- Isolasi deteksi dan klasifikasi untuk performa lebih baik

### Yang Akan Dikerjakan
- [x] **Notebook (All-in-One)**: `notebooks/train_b2_twostage.ipynb`
  - Stage 1: Train detector
  - Extract crops (integrated)
  - Stage 2: Train classifier
  - End-to-end inference
- [x] Config Stage 1: `ffb_ripeness_detect.yaml`
- [ ] **Run Full Pipeline** (Stage 1 → Crop → Stage 2 → Inference)
- [ ] **Evaluasi** (simpan ke `test_twostage.txt`)
- [ ] **Bandingkan** dengan B.1 (end-to-end)

### Expected Output
```
datasets/
└── ffb_ripeness_twostage_crops/  # Crops untuk Stage 2
    ├── train/
    │   ├── ripe/
    │   └── unripe/
    ├── val/
    └── test/
kaggleoutput/
├── runs/detect/
│   ├── exp_b2_stage1_seed42/       # Detector
│   └── exp_b2_stage1_seed123/
├── runs/classify/
│   ├── exp_b2_stage2_seed42/       # Classifier
│   └── exp_b2_stage2_seed123/
└── test_twostage.txt
```

### Metrik Target
- **Stage 1 (Detection)**: mAP50 > 0.80 (seperti B.1)
- **Stage 2 (Classification)**: Top-1 Accuracy > 85%
- **End-to-End**: Accuracy keseluruhan > B.1

---

## 🔬 Metode Validasi & Pelaporan (Konsisten dengan V2)

### Training Protocol
- **Model**: YOLOv11n (Nano) baseline
- **Seeds**: 42 dan 123 (2 runs per eksperimen)
- **Epochs**: 50 (detection), 100 (classification)
- **Batch Size**: 16 (detection), 32 (classification)
- **Image Size**: 640x640 (detection), 224x224 (classification)
- **Device**: CUDA (Kaggle GPU)

### Evaluasi
- **Test set** untuk semua eksperimen (BUKAN validation set)
- Laporkan: **mAP50**, **mAP50-95** (detection) atau **Top-1 Accuracy** (classification)
- **Rata-rata** dari 2 runs (seed 42 & 123) dengan **standar deviasi**
- Sertakan **confusion matrix** dan **contoh kasus gagal** (FP/FN)

---

## 📊 Rencana Analisis Komparatif

Setelah semua eksperimen selesai, buat tabel perbandingan:

| Eksperimen | Input | mAP50 | mAP50-95 | Δ vs Baseline | Insight |
|:-----------|:------|:-----:|:--------:|:-------------:|:--------|
| A.1 (baseline) | RGB | 0.873 | 0.370 | — | — |
| A.3 Fixed | RGB+Depth | ? | ? | ? | Augmentasi fix |
| A.4a | Synthetic Depth | ? | ? | ? | vs A.2 (0.628) |
| A.4b | RGB+Syn.Depth | ? | ? | ? | vs A.3 |
| B.1 (baseline) | RGB (2-class) | 0.801 | 0.514 | — | — |
| B.2 | Two-stage | ? | ? | ? | Detect+Classify |

**Pertanyaan Riset**:
1. Apakah augmentasi fix meningkatkan A.3?
2. Apakah synthetic depth viable untuk deployment?
3. Apakah two-stage lebih baik dari end-to-end?

---

## 📂 File Tracking

### Jupyter Notebooks (Primary) ✅
- `Experiments/notebooks/train_a3_rgbd_fix.ipynb` - A.3 RGBD Fix training
- `Experiments/notebooks/generate_synthetic_depth.ipynb` - Generate synthetic depth maps
- `Experiments/notebooks/train_a4a_synthetic_depth.ipynb` - A.4a training
- `Experiments/notebooks/train_a4b_rgbd_synthetic.ipynb` - A.4b training
- `Experiments/notebooks/train_b2_twostage.ipynb` - B.2 full pipeline

### Python Scripts (Supporting) ✅
- `Experiments/scripts/prepare_synthetic_depth_data.py` - Organize synthetic depth dataset
- `Experiments/scripts/compare_real_vs_synthetic.py` - Analysis A.2 vs A.4a
- `Experiments/scripts/custom_rgbd_dataset.py` - RGBD dataloader helper

### Configs ✅
- `Experiments/configs/ffb_localization_rgbd.yaml`
- `Experiments/configs/ffb_localization_depth_synthetic.yaml`
- `Experiments/configs/ffb_localization_rgbd_synthetic.yaml`
- `Experiments/configs/ffb_ripeness_detect.yaml`

### Output Files (Belum Ada) ⬜
- `Experiments/kaggleoutput/test_rgbd_fixed.txt`
- `Experiments/kaggleoutput/test_synthetic_depth.txt`
- `Experiments/kaggleoutput/test_rgbd_synthetic.txt`
- `Experiments/kaggleoutput/test_twostage.txt`
- `Experiments/kaggleoutput/comparison_real_vs_synthetic.md`

---

## 🚀 Urutan Eksekusi yang Disarankan

### Fase 1: Data Preparation
1. ✅ Pastikan RGB dataset sudah ready (dari V2)
2. ⬜ **Run notebook**: `generate_synthetic_depth.ipynb` (20-30 min GPU)
   - Generate synthetic depth untuk semua RGB images
3. ⬜ **Run script**: `prepare_synthetic_depth_data.py` (organize dataset)

### Fase 2: Training Experiments

**Track A (RGBD) - Can run in parallel:**
- ⬜ **Notebook**: `train_a3_rgbd_fix.ipynb` (A.3 Fix - bisa langsung, real depth sudah ada)
- ⬜ **Notebook**: `train_a4b_rgbd_synthetic.ipynb` (A.4b - setelah synthetic depth ready)

**Track B (Depth Only) - Depends on Fase 1:**
- ⬜ **Notebook**: `train_a4a_synthetic_depth.ipynb` (A.4a - setelah synthetic depth ready)

**Track C (Classification) - Independent:**
- ⬜ **Notebook**: `train_b2_twostage.ipynb` (B.2 - full pipeline dalam 1 notebook)

### Fase 3: Evaluation & Analysis
- ⬜ All notebooks auto-evaluate on test set (hasil di `kaggleoutput/*.txt`)
- ⬜ **Run script**: `compare_real_vs_synthetic.py` (analisis A.2 vs A.4a)
- ⬜ Update `Reports/FFB_Ultimate_Report/result.md` dengan semua hasil
- ⬜ Mulai penulisan skripsi (lihat section FORMAT PENULISAN AKHIR)

---

## 📝 Checklist Sebelum Mulai

- [x] Virtual environment aktif (`.\venv\Scripts\Activate`)
- [ ] Dependencies terinstall (check `requirements.txt` untuk torch, transformers, albumentations)
- [ ] Dataset RGB lokalisasi ready (dari V2)
- [ ] Kaggle environment ready (jika training di Kaggle)
- [ ] Git status clean (commit file-file baru ini terlebih dahulu)

---

## 🎓 Expected Contributions

Eksperimen ini akan menjawab:
1. **Trade-off augmentasi** pada RGBD fusion
2. **Viability synthetic depth** untuk eliminasi hardware sensor
3. **Performa two-stage** vs end-to-end untuk klasifikasi kematangan

Hasil akan di-merge ke `Reports/FFB_Ultimate_Report/result.md` setelah semua eksperimen selesai.

---

## 📓 Jupyter Notebooks - Panduan Penggunaan

### Struktur Notebooks

Semua eksperimen telah dibuat dalam format **Jupyter Notebook (.ipynb)** untuk kemudahan eksekusi dan reproduktifitas. Notebooks dapat dijalankan baik di **lokal** maupun di **Kaggle**.

```
Experiments/notebooks/
├── train_a3_rgbd_fix.ipynb           # A.3 RGBD Fix
├── generate_synthetic_depth.ipynb    # Data prep: Generate synthetic depth
├── train_a4a_synthetic_depth.ipynb   # A.4a Synthetic Depth Only
├── train_a4b_rgbd_synthetic.ipynb    # A.4b RGB+Synthetic Depth
└── train_b2_twostage.ipynb           # B.2 Two-Stage (full pipeline)
```

### Fitur Notebooks

Setiap notebook memiliki:
1. ✅ **Auto-detect environment** (Kaggle vs Local)
2. ✅ **Path handling otomatis** untuk dataset
3. ✅ **GPU detection** & fallback ke CPU
4. ✅ **Progress tracking** dengan tqdm
5. ✅ **Comprehensive evaluation** dengan pandas DataFrame
6. ✅ **Comparison tables** otomatis dengan baseline
7. ✅ **Auto-save results** ke `kaggleoutput/*.txt`
8. ✅ **Markdown documentation** di setiap cell
9. ✅ **Training 2 seeds** (42, 123) untuk reproduktifitas

### Cara Menjalankan - Lokal

```bash
# 1. Aktifkan virtual environment
.\venv\Scripts\Activate

# 2. Install Jupyter (jika belum)
pip install jupyter notebook jupyterlab

# 3. Jalankan Jupyter Lab atau Notebook
jupyter lab
# atau
jupyter notebook

# 4. Navigate ke Experiments/notebooks/
# 5. Buka dan jalankan notebook yang diinginkan
```

### Cara Menjalankan - Kaggle

1. **Upload dataset** ke Kaggle Datasets:
   - `ffb-localization` (RGB images)
   - `ffb-localization-depth` (Real depth maps)
   - `ffb-synthetic-depth` (Synthetic depth - setelah di-generate)
   - `ffb-ripeness` (Ripeness dataset)

2. **Create new Kaggle Notebook**:
   - Copy-paste isi `.ipynb` dari repo
   - Add datasets ke notebook
   - Enable GPU accelerator
   - Run all cells

3. **Download results**:
   - Results auto-saved ke `/kaggle/working/kaggleoutput/`
   - Download ZIP archive di akhir notebook

### Catatan Penting

**A.3 & A.4b (RGBD Notebooks)**:
- Notebooks ini adalah **template**
- Untuk 4-channel RGBD yang proper, lihat `scripts/train_a3_rgbd.py` dan `scripts/train_a4b_rgbd_synthetic.py`
- Requires modification:
  - Modify first conv layer YOLOv11n (3→4 channels)
  - Implement custom dataloader dari `custom_rgbd_dataset.py`

**B.2 Two-Stage**:
- Full pipeline dalam 1 notebook (Stage 1 → Crop → Stage 2 → Inference)
- Sequential execution (tidak bisa dijalankan parallel)

**Synthetic Depth Generation**:
- `generate_synthetic_depth.ipynb` memerlukan **GPU** (Depth-Anything-V2-Large)
- Estimated time: 20-30 menit untuk ~400 images
- Setelah generate, run `scripts/prepare_synthetic_depth_data.py` untuk organize dataset

### Output yang Dihasilkan

Setelah menjalankan notebooks, output akan tersimpan di:

```
Experiments/
├── runs/                              # Training runs
│   ├── detect/
│   │   ├── exp_a3_fixed_seed42/
│   │   ├── exp_a3_fixed_seed123/
│   │   ├── exp_a4a_synthetic_seed42/
│   │   ├── exp_a4a_synthetic_seed123/
│   │   ├── exp_a4b_rgbd_synthetic_seed42/
│   │   ├── exp_a4b_rgbd_synthetic_seed123/
│   │   ├── exp_b2_stage1_seed42/
│   │   └── exp_b2_stage1_seed123/
│   └── classify/
│       ├── exp_b2_stage2_seed42/
│       └── exp_b2_stage2_seed123/
├── kaggleoutput/                      # Test results
│   ├── test_rgbd_fixed.txt
│   ├── test_synthetic_depth.txt
│   ├── test_rgbd_synthetic.txt
│   ├── test_twostage.txt
│   └── comparison_real_vs_synthetic.md
└── datasets/
    └── depth_synthetic_da2_raw/       # Generated synthetic depth
```

### Troubleshooting

**CUDA Out of Memory**:
- Reduce batch size di notebook (16 → 8 → 4)
- Close other programs using GPU

**Path not found errors**:
- Verify dataset paths di cell pertama notebook
- Adjust `DATASET_PATH` variable sesuai lokasi dataset

**Kaggle timeout**:
- Enable GPU accelerator
- Reduce epochs untuk testing
- Save intermediate checkpoints

---

## 📖 FORMAT PENULISAN AKHIR: HASIL PENELITIAN (SKRIPSI)

> **⚠️ PENTING**: Semua eksperimen dari A.1 sampai B.2 (termasuk eksperimen lama dan baru) harus ditulis dalam format **hasil penelitian skripsi** yang lengkap dan akademis.

### Cakupan Eksperimen yang Dilaporkan

**Eksperimen Sudah Selesai (dari V2):**
- A.1: RGB Only (mAP50: 0.873, mAP50-95: 0.370)
- A.2: Depth Only (mAP50: 0.628, mAP50-95: 0.226)
- A.3: RGB+Depth (lama, dengan augmentasi issue) (mAP50: 0.869, mAP50-95: 0.379)
- B.1: Ripeness Detection 2-class (mAP50: 0.801, mAP50-95: 0.514)
- Ablation Study: Gap 1-5 (model scaling, optimizer)

**Eksperimen Baru (dari V3):**
- A.3 Fix: RGB+Depth dengan augmentasi diperbaiki
- A.4a: Synthetic Depth Only (Depth-Anything-V2)
- A.4b: RGB+Synthetic Depth
- B.2: Two-Stage Ripeness Classification

### Struktur Dokumen Skripsi

Dokumen skripsi harus mengikuti struktur standar penelitian akademis:

```
BAB I: PENDAHULUAN
├── 1.1 Latar Belakang
├── 1.2 Rumusan Masalah
├── 1.3 Tujuan Penelitian
├── 1.4 Manfaat Penelitian
├── 1.5 Batasan Masalah
└── 1.6 Sistematika Penulisan

BAB II: TINJAUAN PUSTAKA
├── 2.1 Fresh Fruit Bunch (FFB) Oil Palm
├── 2.2 Computer Vision untuk Deteksi Objek
├── 2.3 YOLO (You Only Look Once)
├── 2.4 Depth Sensing & RGBD Fusion
├── 2.5 Depth-Anything-V2 (Monocular Depth Estimation)
├── 2.6 Ripeness Classification Methods
├── 2.7 Two-Stage Detection-Classification Pipeline
└── 2.8 Penelitian Terkait (Related Works)

BAB III: METODOLOGI PENELITIAN
├── 3.1 Kerangka Penelitian
├── 3.2 Dataset & Preprocessing
│   ├── 3.2.1 Sumber Data (RealSense Camera)
│   ├── 3.2.2 Struktur Dataset
│   ├── 3.2.3 EDA (Exploratory Data Analysis)
│   ├── 3.2.4 Data Split (70:20:10)
│   ├── 3.2.5 Preprocessing Depth Maps (0.6m-6.0m normalization)
│   └── 3.2.6 Synthetic Depth Generation (Depth-Anything-V2)
├── 3.3 Arsitektur Model
│   ├── 3.3.1 YOLOv11n Baseline
│   ├── 3.3.2 4-Channel RGBD Modification
│   └── 3.3.3 Two-Stage Pipeline Architecture
├── 3.4 Eksperimen Design
│   ├── 3.4.1 A.1: RGB Only (Baseline)
│   ├── 3.4.2 A.2: Depth Only
│   ├── 3.4.3 A.3: RGB+Depth Fusion
│   ├── 3.4.4 A.4a: Synthetic Depth Only
│   ├── 3.4.5 A.4b: RGB+Synthetic Depth
│   ├── 3.4.6 B.1: End-to-End Ripeness Detection
│   ├── 3.4.7 B.2: Two-Stage Ripeness Classification
│   └── 3.4.8 Ablation Study (Model Scaling & Optimizer)
├── 3.5 Training Protocol
│   ├── 3.5.1 Hyperparameters
│   ├── 3.5.2 Augmentation Strategy
│   ├── 3.5.3 Loss Functions
│   └── 3.5.4 Random Seeds (42, 123)
├── 3.6 Evaluation Metrics
│   ├── 3.6.1 mAP50 & mAP50-95
│   ├── 3.6.2 Precision & Recall
│   ├── 3.6.3 Confusion Matrix
│   └── 3.6.4 Failure Case Analysis
└── 3.7 Computational Environment

BAB IV: HASIL DAN PEMBAHASAN
├── 4.1 Hasil Exploratory Data Analysis
├── 4.2 Hasil Eksperimen Lokalisasi
│   ├── 4.2.1 A.1: RGB Only Baseline
│   ├── 4.2.2 A.2: Depth Only
│   ├── 4.2.3 A.3: RGB+Depth Fusion (Lama vs Fix)
│   ├── 4.2.4 A.4a: Synthetic Depth Only
│   ├── 4.2.5 A.4b: RGB+Synthetic Depth
│   └── 4.2.6 Perbandingan Semua Eksperimen Lokalisasi
├── 4.3 Hasil Eksperimen Klasifikasi Kematangan
│   ├── 4.3.1 B.1: End-to-End Ripeness Detection
│   ├── 4.3.2 B.2: Two-Stage Classification
│   └── 4.3.3 Perbandingan B.1 vs B.2
├── 4.4 Hasil Ablation Study
│   ├── 4.4.1 Model Scaling (Nano vs Small)
│   ├── 4.4.2 Optimizer Comparison (SGD vs AdamW)
│   ├── 4.4.3 Training Duration (50e vs 300e)
│   └── 4.4.4 Overfitting Analysis
├── 4.5 Analisis Real Depth vs Synthetic Depth
├── 4.6 Failure Case Analysis
│   ├── 4.6.1 False Positives
│   ├── 4.6.2 False Negatives
│   └── 4.6.3 Misclassification Patterns
└── 4.7 Diskusi & Interpretasi Hasil

BAB V: KESIMPULAN DAN SARAN
├── 5.1 Kesimpulan
├── 5.2 Kontribusi Penelitian
├── 5.3 Keterbatasan Penelitian
└── 5.4 Saran untuk Penelitian Selanjutnya

DAFTAR PUSTAKA
LAMPIRAN
├── A. Source Code
├── B. Training Logs
├── C. Hasil Visualisasi Lengkap
└── D. Dataset Statistics
```

### Format Akademis yang Harus Diikuti

1. **Bahasa**: Bahasa Indonesia formal (atau Bahasa Inggris jika diperlukan)
2. **Kutipan**: Gunakan citation style (IEEE atau APA) untuk semua referensi
3. **Tabel & Gambar**:
   - Setiap tabel/gambar harus diberi nomor dan caption
   - Format: "Tabel 4.1 Hasil Perbandingan mAP50 pada Test Set"
   - Format: "Gambar 4.2 Confusion Matrix Eksperimen A.1 (Seed 42)"
4. **Persamaan**: Gunakan notasi matematika untuk loss functions, metrics, dll
5. **Sitasi Related Works**: BAB II harus mencakup minimal 20-30 referensi penelitian terkait

### Konten yang Harus Ada di Setiap Eksperimen (BAB IV)

Untuk setiap eksperimen (A.1, A.2, A.3, A.4a, A.4b, B.1, B.2):

```markdown
#### X.X.X Eksperimen [Nama]

**Deskripsi**: [Penjelasan singkat eksperimen]

**Hipotesis**: [Hipotesis penelitian]

**Hasil Kuantitatif**:
| Seed | mAP50 | mAP50-95 | Precision | Recall |
|:----:|:-----:|:--------:|:---------:|:------:|
| 42   | X.XXX | X.XXX    | X.XXX     | X.XXX  |
| 123  | X.XXX | X.XXX    | X.XXX     | X.XXX  |
| **Rata-rata** | **X.XXX** | **X.XXX** | **X.XXX** | **X.XXX** |

**Training Dynamics**: [Analisis kurva training - loss, mAP]

**Confusion Matrix**: [Gambar + interpretasi]

**Precision-Recall Curve**: [Gambar + interpretasi]

**Failure Cases**: [Visual examples + analisis]

**Pembahasan**: [Diskusi hasil, perbandingan dengan baseline/hipotesis]
```

### Deliverables Akhir

Setelah semua eksperimen selesai, buat dokumen-dokumen berikut:

1. **Skripsi Lengkap** (format .docx atau .tex):
   - File: `Reports/FFB_Skripsi/FFB_Thesis_Final.docx`
   - Minimal 60-80 halaman
   - Sesuai template skripsi universitas

2. **Summary Report** (format .md, untuk repo):
   - File: `Reports/FFB_Ultimate_Report/result.md` (update existing)
   - Versi ringkas untuk dokumentasi teknis

3. **Presentasi** (format .pptx):
   - File: `Reports/FFB_Presentation/FFB_Defense.pptx`
   - 20-30 slides untuk sidang/presentasi

4. **Artifacts Archive**:
   - Semua model weights, training logs, visualizations
   - File: `Reports/FFB_Skripsi/artifacts.zip`

### Timeline Penulisan

Fase penulisan sebaiknya dilakukan **setelah semua eksperimen selesai**:

```
Week 1-2: Eksperimen A.3 Fix, A.4a, A.4b
Week 3:   Eksperimen B.2
Week 4:   Evaluasi & Analisis Komparatif
Week 5-6: Penulisan BAB I-III (Pendahuluan, Tinjauan Pustaka, Metodologi)
Week 7-8: Penulisan BAB IV (Hasil & Pembahasan) - BAGIAN TERPANJANG
Week 9:   Penulisan BAB V (Kesimpulan & Saran)
Week 10:  Review, Revisi, Finalisasi
```

### Catatan Penting

- **Semua angka dan tabel** harus konsisten dengan data di `Reports/FFB_Ultimate_Report/result.md`
- **Grafik dan visualisasi** harus high-quality (300 DPI untuk cetak)
- **Citation** untuk Depth-Anything-V2, YOLOv11, RealSense camera, etc.
- **Ethical statement**: Jelaskan bahwa dataset dikumpulkan dengan proper authorization
- **Reproducibility**: Sertakan semua hyperparameters, seeds, preprocessing steps

---

*Guide created: 2026-01-22 | Updated: 2026-01-22 (Added Jupyter Notebooks) | Status: READY | Next update: After experiments completed*