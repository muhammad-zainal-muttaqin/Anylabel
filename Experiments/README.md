# 🧪 Eksperimen Deteksi TBS Kelapa Sawit

**Proyek:** Anylabel - Asisten Dosen  
**Tanggal:** 2026-01-15  
**Versi:** 2.0 (Clean)

---

## 📋 Apa yang Dibuat?

SEMUA FILE TELAH DIBERSIHKAN - HANYA FILE PENTING SAJA!

### ✅ Files Utama
| File | Deskripsi |
|------|-----------|
| `EXPERIMENT_GUIDE_V2.md` | **Panduan lengkap** (WAJIB BACA!) |
| `ffb_localization.yaml` | Config dataset YOLO |
| `README.md` | File ini |

### ✅ Scripts Python

**Persiapan Data:**
- `scripts/simple_eda.py` - Analisis dataset
- `scripts/split_localization_data.py` - Split 70:20:10
- `scripts/prepare_depth_data.py` - Proses depth

**Training:**
- `scripts/train_a1_rgb.py` - RGB Only (2 runs)
- `scripts/train_a2_depth.py` - Depth Only (2 runs)
- `scripts/train_b1_classification.py` - Classification (2 runs)

**Evaluasi:**
- `scripts/evaluate_all.py` - Semua model
- `scripts/failure_analysis.py` - FP/FN

---

## 🚀 CEPAT MULAI (Quick Start)

### Langkah 0: Install Dependencies
```bash
.\venv\Scripts\Activate
pip install ultralytics opencv-python numpy pandas matplotlib seaborn
```

### Langkah 1: Ekstrak Dataset
```bash
# Ekstrak: Dataset\28574489.zip
# Tujuan: Dataset\gohjinyu-oilpalm-ffb-dataset-d66eb99\
# Pastikan ada folder:
# - ffb-localization/rgb_images/
# - ffb-localization/depth_maps/
```

### Langkah 2: Anotasi (Manual!)
```bash
# Buka AnyLabeling
anylabeling

# Atur:
# - Output: Experiments/datasets/ffb_localization/labels/
# - Input: Dataset/gohjinyu.../ffb-localization/rgb_images/
# - Class: 1 class (fresh_fruit_bunch)
# - Format: YOLO

# Target: 300-500+ gambar teranotasi
```

### Langkah 3: Persiapan Data
```bash
cd Experiments\scripts
python simple_eda.py
python split_localization_data.py
python prepare_depth_data.py
```

### Langkah 4: Training
```bash
# A.1 RGB Only
python train_a1_rgb.py

# A.2 Depth Only  
python train_a2_depth.py

# B.1 Classification
python train_b1_classification.py
```

### Langkah 5: Evaluasi & Laporan
```bash
python evaluate_all.py
python failure_analysis.py

# Hasil:
# - Experiments/LAPORAN_EKSPERIMEN.md
# - Experiments/failure_analysis/
```

---

## 🧪 Eksperimen Yang Dijalankan

### A. Lokalisasi (Object Detection)

| # | Input | Model | Output |
|---|-------|-------|--------|
| **A.1** | RGB Only | YOLOv8n | Bounding Box |
| **A.2** | Depth (3-ch) | YOLOv8n | Bounding Box |
| **A.3** | RGB+Depth (4-ch) | YOLOv8n (modified) | Bounding Box |

### B. Klasifikasi

| # | Input | Model | Output |
|---|-------|-------|--------|
| **B.1** | RGB Only | YOLOv8n-cls | Ripe vs Unripe |

**Catatan:**
- Setiap eksperimen: **2 kali training** (seed 42 & 123)
- Total: **6 training runs** (3 eksperimen × 2 runs)

---

## 📁 Struktur Folder

```
Experiments/
├── 📄 EXPERIMENT_GUIDE_V2.md        # Panduan lengkap
├── 📄 ffb_localization.yaml         # Config YOLO
├── 📄 README.md                     # File ini
│
├── 📂 datasets/
│   ├── ffb_localization/            # Split data RGB
│   │   ├── images/{train,val,test}
│   │   └── labels/{train,val,test}
│   ├── ffb_localization_depth/      # Split data Depth (A.2)
│   └── ffb_ripeness/                # Split data Classification (B.1)
│
├── 📂 scripts/
│   ├── simple_eda.py
│   ├── split_localization_data.py
│   ├── prepare_depth_data.py
│   ├── train_a1_rgb.py
│   ├── train_a2_depth.py
│   ├── train_b1_classification.py
│   ├── evaluate_all.py
│   ├── failure_analysis.py
│   └── run_all_experiments.py       # Master script
│
├── 📂 runs/                          # Hasil training (auto-created)
│   ├── detect/exp_a1_rgb_*/         # Model A.1
│   ├── detect/exp_a2_depth_*/       # Model A.2
│   └── classify/exp_b1_cls_*/       # Model B.1
│
└── 📂 failure_analysis/              # Hasil analisis (auto-created)
    ├── false_positives/
    ├── false_negatives/
    └── correct_detections/
```

---

## 📊 Hasil Yang Diharapkan

### Training Metrics (Target)
- **A.1 RGB**: mAP50 ~0.7-0.85 (baseline)
- **A.2 Depth**: mAP50 ~0.6-0.8 (tergantung kualitas depth)
- **B.1 Cls**: Top1 Acc ~0.8-0.95

### Output Files
1. **Model Weights** (`runs/*/weights/best.pt`)
2. **Training Results** (`runs/*/results.png`)
3. **Laporan** (`LAPORAN_EKSPERIMEN.md`)
4. **Failure Analysis** (`failure_analysis/` + images)
5. **CSV Results** (`experiment_results.csv`)

---

## 🔧 Troubleshooting

### ❌ "Dataset not found"
```bash
# Cek struktur folder
ls Dataset\gohjinyu-oilpalm-ffb-dataset-d66eb99\ffb-localization\
# Harus ada: rgb_images/ depth_maps/
```

### ❌ "CUDA out of memory"
```bash
# Edit config, turunkan batch size
batch: 8  # atau 4
# atau gunakan CPU
device: cpu
```

### ❌ "No labels found"
```bash
# Pastikan anotasi selesai di AnyLabeling
# Cek jumlah file .txt vs jumlah gambar
```

### ❌ Training lama
```bash
# Gunakan GPU! (10x lebih cepat)
device: 0

# Turunkan epochs untuk test
epochs: 10  # bukan 50
```

---

## 📚 Referensi

### Dokumentasi
- **YOLOv8**: https://docs.ultralytics.com/
- **AnyLabeling**: https://github.com/vietanhdev/anylabeling
- **Dataset**: `Dataset\dataset Goh 2025.md`

### Tools
- **Annotasi**: AnyLabeling
- **Training**: Ultralytics YOLO
- **Analysis**: OpenCV, Pandas, Matplotlib

---

## 👥 Tim
- **Dosen Pembimbing**: [Nama Dosen]
- **Asisten Dosen**: Anda
- **AI Assistant**: Factory Droid

---

## 🎯 Checklist Sebelum Training

- [ ] Dataset terextract (4.3GB zip)
- [ ] Python dependencies terinstall
- [ ] Anotasi selesai (300+ gambar)
- [ ] Run `simple_eda.py` ✅
- [ ] Run `split_localization_data.py` ✅
- [ ] Run `prepare_depth_data.py` ✅
- [ ] Config `ffb_localization.yaml` ada ✅
- [ ] GPU tersedia (opsional, tapi recommended)
- [ ] Disk space cukup (~10GB)

---

## 💡 Tips

1. **Mulai dari A.1** (RGB Only) dulu sebagai baseline
2. **Gunakan GPU** untuk training cepat
3. **Cek results.png** di folder runs/ untuk lihat learning curve
4. **Simpan weights terbaik** setiap eksperimen
5. **Jangan lupa seed berbeda** (42 & 123) untuk validasi

---

**Last Updated:** 2026-01-15  
**Status:** Ready to Use 🚀
