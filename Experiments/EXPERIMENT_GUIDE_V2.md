# Eksperimen Deteksi TBS & Klasifikasi Kematangan Sawit
**Versi 2.2 - Panduan Ringkas (sesuai rencana eksperimen)**

---

## Ringkasan Eksperimen

| Eksperimen | Input | Model (Baseline) | Target | Output |
|------------|-------|------------------|--------|--------|
| **A.1** | RGB | **YOLO11n** | Object Detection | Bounding Box TBS |
| **A.2** | Depth (3-channel) | **YOLO11n** | Object Detection | Bounding Box TBS |
| **A.3 (opsional)** | RGB+Depth (4-ch) | **Custom** | Object Detection | Bounding Box TBS |
| **B.1** | RGB | **YOLO11n-cls** | Classification (2 kelas: ripe/unripe) | Label kematangan |

### Keputusan Model (Disarankan)
- **Model (baseline)**: **YOLO11n atau YOLOv8n**. Gunakan YOLO11 sebagai baseline bila tidak ada alasan khusus memakai YOLOv8.
- **Target deploy mobile (kandidat)**: **YOLO26** (utamanya varian `n`) untuk uji latensi end-to-end di perangkat.
- **Catatan penting**: beberapa laporan menunjukkan **YOLO26 (NMS-free head)** bisa butuh **epoch lebih tinggi** agar menyamai YOLO11 pada dataset yang tidak terlalu besar. Untuk baseline cepat, mulai dari YOLO11 dulu.

### Ruang Lingkup Eksperimen
- **A (Lokalisasi)**: single-class (TBS), jalankan RGB-only, Depth-only (replikasi 1→3), lalu RGB+Depth (4 channel).
- **B (Klasifikasi)**: **2 kelas (ripe/unripe)**, RGB-only pakai dataset classification.

---

## FASE 0: Persiapan Awal (Wajib!)

### 0.1. Install Dependencies
```bash
# Aktifkan virtual environment
.\venv\Scripts\Activate

# Install semua dependencies dari requirements.txt (recommended untuk reproducibility)
pip install -r requirements.txt

# ATAU install manual (jika tidak pakai requirements.txt):
# pip install ultralytics opencv-python numpy pandas matplotlib seaborn
# pip install anylabeling  # Untuk anotasi
# pip install pyrealsense2  # Jika ingin collect data baru (opsional)
```

### 0.2. Ekstrak Dataset
```bash
# Ekstrak zip dataset ke folder yang benar
# File: Dataset\28574489.zip
# Tujuan: Dataset\gohjinyu-oilpalm-ffb-dataset-d66eb99\

# Pastikan folder berisi:
# - ffb-localization/rgb_images/
# - ffb-localization/depth_maps/
# - ffb-localization/point_clouds/
```

### 0.3. Cek Struktur Dataset
```bash
# Jalankan EDA script untuk cek data
cd Experiments\scripts
python simple_eda.py
```

---

## FASE 1: Anotasi Data (CRITICAL!)

### 1.1. Anotasi Lokalisasi (Object Detection)
**Tools:** AnyLabeling (AnyLabeling.exe)

**Langkah:**
1. **Buka AnyLabeling**
   ```bash
   .\venv\Scripts\Activate
   anylabeling
   ```

2. **Load Dataset**
   - Menu: `File` → `Change Output Directory` → Pilih: `Experiments/labeling/ffb_localization/json/`
   - Menu: `File` → `Open Dir` → Pilih: `Dataset/gohjinyu-oilpalm-ffb-dataset-d66eb99/ffb-localization/rgb_images/`

3. **Atur Kelas**
   - Menu: `File` → `Change Save Format` → Pilih: `LabelMe (JSON)`
   - Class Editor: Tambah 1 class
     - Name: `fresh_fruit_bunch`
     - Color: (Pilih warna)

4. **Mulai Anotasi**
   - Tekan `W` → Gambar bounding box di sekitar TBS
   - Tekan `D` → Next image
   - Tekan `A` → Previous image
   - **Tips:** Gunakan `Auto Labeling` (YOLO/SAM) untuk estimasi awal, lalu **periksa dan koreksi manual** (hasil autolabel harus diperiksa manual)

5. **Target Anotasi**
   - Minimal: 300-500 gambar teranotasi
   - Ideal: 1000+ gambar
   - Status: Pastikan jumlah file `.json` = jumlah gambar

6. **Konversi JSON → YOLO (.txt)**
   ```bash
   cd Experiments\scripts
   python convert_json_to_yolo.py
   ```
   Output YOLO akan ada di: `Experiments/labeling/ffb_localization/yolo_all/`

### 1.2. Anotasi Klasifikasi (Jika Diperlukan)
Jika dataset klasifikasi (`ffb-ripeness-classification`) belum ada:
- Kumpulkan data terpisah
- Siapkan label **2 kelas**: `ripe` dan `unripe` (atau gunakan nama folder yang Anda pakai), dan pastikan konsisten dari awal

---

## FASE 2: Data Preparation

### 2.1. EDA (Exploratory Data Analysis)
```bash
cd Experiments\scripts
python simple_eda.py
```

**Cek untuk dataset lokalisasi:**
- Jumlah gambar teranotasi
- Distribusi ukuran bounding box
- Jumlah objek per gambar

**Cek untuk dataset klasifikasi:**
- Jumlah gambar per kelas (ripe/unripe)
- Distribusi kelas (apakah seimbang atau tidak)
- Kualitas gambar (resolusi, blur, dll)

### 2.2. Split Dataset
```bash
cd Experiments\scripts
python split_localization_data.py
```

**Output Structure:**
```
Experiments/datasets/ffb_localization/
├── images/
│   ├── train/   (70%)
│   ├── val/     (20%)
│   └── test/    (10%)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 2.3. Prepare Depth Data (Untuk Eksperimen A.2)
```bash
cd Experiments\scripts
python prepare_depth_data.py
```

**Process:**
- Baca depth map asli (16-bit)
- Normalisasi min-max ke rentang 0-255 dengan range depth **0.6m - 6.0m**
- Replikasi channel depth tunggal sebanyak 3 kali dengan pola: **R=G, G=B, B=R**
- Output: `Experiments/datasets/depth_processed_rgb/`

### 2.4. Buat Dataset Config YOLO
**File:** `Experiments/ffb_localization.yaml`

```yaml
path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_localization
train: images/train
val: images/val
test: images/test

nc: 1  # number of classes
names: ['fresh_fruit_bunch']
```

---

## FASE 3: Eksperimen A - Lokalisasi (Detection)

### **Eksperimen A.1: RGB Only (Baseline)**

#### Step 1: Setup Config
**File:** `Experiments/config_a1_rgb.yaml`
```yaml
task: detect
mode: train
model: yolo11n.pt
data: ffb_localization.yaml
epochs: 50
patience: 10
batch: 16
imgsz: 640
save: True
device: 0  # or 'cpu' if no GPU
workers: 4
project: runs/detect
name: exp_a1_rgb_baseline
seed: 42
```

**Opsi deploy mobile (uji banding):**
- Ganti `model: yolo11n.pt` → `model: yolo26n.pt`
- Jika hasil YOLO26 lebih rendah, coba **naikkan epoch** (mis. 100) sebelum menyimpulkan.

#### Step 2: Training (Run #1)
```bash
cd Experiments
yolo detect train config=config_a1_rgb.yaml
```

#### Step 3: Training (Run #2 dengan seed berbeda)
**Edit:** `config_a1_rgb.yaml` → `seed: 123`

```bash
yolo detect train config=config_a1_rgb.yaml
```

#### Step 4: Evaluasi
```bash
# Evaluate pada test set
yolo detect val model=runs/detect/exp_a1_rgb_baseline/weights/best.pt data=ffb_localization.yaml split=test

# Simpan hasil:
# - mAP50
# - mAP50-95
# - Confusion matrix
# - Precision, Recall
```

---

### **Eksperimen A.2: Depth Only**

#### Step 1: Siapkan Data Depth
Pastikan `prepare_depth_data.py` sudah dijalankan → `Experiments/datasets/depth_processed_rgb/`

#### Step 2: Buat Dataset Config Depth
**File:** `Experiments/ffb_localization_depth.yaml`
```yaml
path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_localization
train: images/train  # Gunakan folder yang sama, tapi gambar dari depth_processed_rgb
val: images/val
test: images/test

nc: 1
names: ['fresh_fruit_bunch']
```

**Catatan:** Anda perlu copy gambar dari `depth_processed_rgb` ke folder `images/train/val/test`

#### Step 3: Setup Config
**File:** `Experiments/config_a2_depth.yaml`
```yaml
task: detect
mode: train
model: yolo11n.pt
data: ffb_localization_depth.yaml
epochs: 50
patience: 10
batch: 16
imgsz: 640
save: True
device: 0
workers: 4
project: runs/detect
name: exp_a2_depth_only
seed: 42
```

#### Step 4: Training (Run #1 & #2)
```bash
# Run 1: seed 42
yolo detect train config=config_a2_depth.yaml

# Run 2: seed 123 (edit config)
yolo detect train config=config_a2_depth.yaml
```

---

### **Eksperimen A.3: RGB + Depth (4-Channel)**

**Status:** opsional (lanjutkan setelah A.1/A.2 stabil).

**Tujuan:** menguji apakah depth membantu jika digabung dengan RGB (input 4 channel).

**Ringkas langkahnya:**
1. Buat dataset 4-channel (gabungkan RGB + depth menjadi `[R,G,B,D]`).
2. Modifikasi layer input pertama model agar menerima 4 channel.
3. Train + evaluasi seperti A.1.

**Catatan:** ini pekerjaan “custom” (tidak drop-in seperti A.1/A.2), jadi jangan dijadikan baseline awal.

---

## FASE 4: Eksperimen B - Klasifikasi

### **Eksperimen B.1: RGB Only (2 Kelas: ripe/unripe)**

#### Step 1: Siapkan Dataset Klasifikasi
- Dataset: `ffb-ripeness-classification`
- Struktur: `images/ripe/`, `images/unripe/`
- Pastikan penamaan folder dan label konsisten.

#### Step 2: Split Data Klasifikasi
- Split **70/20/10** dengan **stratified per kelas** (jaga proporsi `ripe`/`unripe`).
- Jika perlu otomatisasi, buat script `Experiments/scripts/split_classification_data.py` (opsional).

#### Step 3: Dataset Config
**File:** `Experiments/ffb_ripeness.yaml`
```yaml
path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_ripeness
train: images/train
val: images/val
test: images/test

nc: 2
names: ['ripe', 'unripe']
```

#### Step 4: Setup Config
**File:** `Experiments/config_b1_cls.yaml`
```yaml
task: classify
mode: train
model: yolo11n-cls.pt
data: ffb_ripeness.yaml
epochs: 50
patience: 10
batch: 32
imgsz: 224
save: True
device: 0
workers: 4
project: runs/classify
name: exp_b1_rgb_classification
seed: 42
```

**Opsi deploy mobile (uji banding):**
- Ganti `model: yolo11n-cls.pt` → `model: yolo26n-cls.pt`
- Jika kelas tidak seimbang, prioritaskan perbaikan data/sampling dulu sebelum ganti model.

#### Step 5: Training (Run #1 & #2)
```bash
# Run 1
yolo classify train config=config_b1_cls.yaml

# Run 2 (seed 123)
yolo classify train config=config_b1_cls.yaml
```

---

## FASE 5: Validasi & Reporting

### 5.1. Evaluasi Semua Model
**Script:** `Experiments/scripts/evaluate_all.py`

Jalankan:

```bash
cd Experiments\scripts
python evaluate_all.py
```

Output:
- File ringkasan metrik (CSV), mis. `experiment_results.csv`
- Pastikan path weights dan file dataset YAML di dalam script sudah sesuai folder `runs/` Anda

### 5.2. Analisis Kegagalan
**Script:** `Experiments/scripts/failure_analysis.py`

Jalankan:

```bash
cd Experiments\scripts
python failure_analysis.py
```

Fokus analisis:
- False Positive / False Negative
- Kondisi sulit: pencahayaan, occlusion, ukuran objek, blur

### 5.3. Report Template
**File:** `Experiments/LAPORAN_HASIL.md`

```markdown
# Laporan Eksperimen Deteksi TBS Kelapa Sawit

## Ringkasan Eksekusi
| Eksperimen | Run #1 (mAP50) | Run #2 (mAP50) | Rata-rata mAP50 | Run #1 (mAP50-95) | Run #2 (mAP50-95) | Rata-rata mAP50-95 |
|------------|----------------|----------------|-----------------|-------------------|-------------------|-------------------|
| A.1 RGB    |                |                |                 |                   |                   |                   |
| A.2 Depth  |                |                |                 |                   |                   |                   |
| B.1 Cls    |                |                |                 |                   |                   |                   |

## Analisis Kegagalan
[Contoh gambar failure + analisis]

## Kesimpulan
[Rekomendasi]
```

---

## CEKLIST EKSEKUSI

### Tahap 1: Persiapan
- [ ] Extract dataset 28574489.zip
- [ ] Install dependencies (ultralytics, opencv, dll)
- [ ] Run `simple_eda.py`

### Tahap 2: Anotasi
- [ ] Anotasi 300-500+ gambar di AnyLabeling
- [ ] Verifikasi jumlah label = jumlah gambar

### Tahap 3: Data Processing
- [ ] Run `split_localization_data.py`
- [ ] Run `prepare_depth_data.py`
- [ ] Create `ffb_localization.yaml`

### Tahap 4: Eksperimen A.1 (RGB Baseline)
- [ ] Config: `config_a1_rgb.yaml` (seed 42)
- [ ] Training Run #1
- [ ] Config: `config_a1_rgb.yaml` (seed 123)
- [ ] Training Run #2
- [ ] Eval & Save metrics
- [ ] (Opsional) Uji `yolo26n.pt` untuk kandidat deploy mobile + epoch lebih tinggi bila perlu

### Tahap 5: Eksperimen A.2 (Depth Only)
- [ ] Prepare depth data
- [ ] Copy depth images ke folder split
- [ ] Config: `config_a2_depth.yaml` (seed 42)
- [ ] Training Run #1
- [ ] Config: `config_a2_depth.yaml` (seed 123)
- [ ] Training Run #2
- [ ] Eval & Save metrics

### Tahap 6: Eksperimen A.3 (RGB+Depth 4-ch)
- [ ] (Opsional) Siapkan dataset 4-channel (RGB+Depth)
- [ ] (Opsional) Modifikasi model untuk input 4 channel
- [ ] (Opsional) Training 2x runs + evaluasi

### Tahap 7: Eksperimen B.1 (Classification)
- [ ] Prepare classification dataset
- [ ] Split data
- [ ] Config: `config_b1_cls.yaml` (seed 42)
- [ ] Training Run #1
- [ ] Config: `config_b1_cls.yaml` (seed 123)
- [ ] Training Run #2
- [ ] Eval & Save metrics
- [ ] (Opsional) Uji `yolo26n-cls.pt` untuk kandidat deploy mobile

### Tahap 8: Reporting
- [ ] Run `evaluate_all.py`
- [ ] Run `failure_analysis.py`
- [ ] Create `LAPORAN_HASIL.md`
- [ ] Buat presentasi

---

## Tips & Best Practices

1. **GPU:** Jika tersedia, gunakan GPU (device: 0). Training akan jauh lebih cepat.
2. **Monitor:** Gunakan `tensorboard --logdir runs/` untuk monitor training.
3. **Backup:** Simpan weights terbaik setiap eksperimen.
4. **Konsistensi:** Gunakan random seed yang sama untuk reproducbility.
5. **Disk Space:** Training 50 epoch butuh ~2-5GB per eksperimen.
6. **Interrupted Training:** Gunakan `patience` (auto-stop jika tidak improve).

---

## Troubleshooting

### Error: "Label not found"
- Pastikan anotasi selesai sebelum split
- Cek jumlah file .txt vs .png

### Error: "CUDA out of memory"
- Turunkan batch size: `batch: 8` atau `batch: 4`
- Turunkan imgsz: `imgsz: 512`

### Error: "Module not found"
- Aktifkan venv: `.\venv\Scripts\Activate`
- Install: `pip install ultralytics opencv-python`

### Training lama
- Gunakan model nano: `yolo11n.pt` / `yolo26n.pt` (bukan varian m/l/x)
- Untuk uji cepat, turunkan epochs (mis. 30). Untuk hasil serius, kembali ke epoch rekomendasi di config.
- Gunakan GPU!

---

**Created by:** Factory Droid  
**Last Updated:** 2026-01-15  
**Project:** Anylabel - Deteksi TBS Kelapa Sawit
