# 🧪 Eksperimen Deteksi Tandan Buah Segar (TBS) Kelapa Sawit
**Versi 2.0 - Panduan Lengkap Step-by-Step**

---

## 📋 Ringkasan Eksperimen

| Eksperimen | Input | Model | Target | Output |
|------------|-------|-------|--------|--------|
| **A.1** | RGB Only | YOLOv8n | Object Detection | Bounding Box TBS |
| **A.2** | Depth Only (3-channel) | YOLOv8n | Object Detection | Bounding Box TBS |
| **A.3** | RGB + Depth (4-channel) | YOLOv8n (modified) | Object Detection | Bounding Box TBS |
| **B.1** | RGB Only | YOLOv8n-cls | Classification | Ripe vs Unripe |

---

## 🎯 FASE 0: Persiapan Awal (Wajib!)

### 0.1. Install Dependencies
```bash
# Aktifkan virtual environment
.\venv\Scripts\Activate

# Install required packages
pip install ultralytics opencv-python numpy pandas matplotlib seaborn
pip install pyrealsense2  # Jika ingin collect data baru
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

## 🎯 FASE 1: Anotasi Data (CRITICAL!)

### 1.1. Anotasi Lokalisasi (Object Detection)
**Tools:** AnyLabeling (AnyLabeling.exe)

**Langkah:**
1. **Buka AnyLabeling**
   ```bash
   # Pastikan venv aktif
   .\venv\Scripts\Activate
   anylabeling
   ```

2. **Load Dataset**
   - Menu: `File` → `Change Output Directory` → Pilih: `Experiments/datasets/ffb_localization/labels/`
   - Menu: `File` → `Open Dir` → Pilih: `Dataset/gohjinyu-oilpalm-ffb-dataset-d66eb99/ffb-localization/rgb_images/`

3. **Atur Kelas**
   - Menu: `File` → `Change Save Format` → Pilih: `YOLO`
   - Class Editor: Tambah 1 class
     - Name: `fresh_fruit_bunch`
     - Color: (Pilih warna)

4. **Mulai Anotasi**
   - Tekan `W` → Gambar bounding box di sekitar TBS
   - Tekan `D` → Next image
   - Tekan `A` → Previous image
   - **Tips:** Gunakan `Auto Labeling` (YOLO/SAM) untuk estimasi awal, lalu koreksi manual

5. **Target Anotasi**
   - Minimal: 300-500 gambar teranotasi
   - Ideal: 1000+ gambar
   - Status: Pastikan jumlah file `.txt` = jumlah gambar

### 1.2. Anotasi Klasifikasi (Jika Diperlukan)
Jika dataset klasifikasi (`ffb-ripeness-classification`) belum ada:
- Kumpulkan data terpisah
- Anotasi dengan kelas: `ripe_ffb` dan `unripe_ffb`

---

## 🎯 FASE 2: Data Preparation

### 2.1. EDA (Exploratory Data Analysis)
```bash
cd Experiments\scripts
python simple_eda.py
```

**Cek:**
- [ ] Jumlah gambar teranotasi
- [ ] Distribusi ukuran bounding box
- [ ] Jumlah objek per gambar
- [ ] Distribusi kelas (untuk klasifikasi)

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
- Normalisasi 0-255
- Replikasi 3 channel
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

## 🎯 FASE 3: Eksperimen A - Lokalisasi (Detection)

### **Eksperimen A.1: RGB Only (Baseline)**

#### Step 1: Setup Config
**File:** `Experiments/config_a1_rgb.yaml`
```yaml
task: detect
mode: train
model: yolov8n.pt
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
model: yolov8n.pt
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

**Tantangan:** Modifikasi model YOLO untuk menerima 4 channel input.

#### Opsi 1: Modifikasi Model dari Awal (Recommended untuk eksperimen)
**File:** `Experiments/model_4channel.py`
```python
import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLO4Channel(YOLO):
    def __init__(self, model='yolov8n.pt', ch=4):
        super().__init__(model)
        # Modifikasi first layer untuk 4 channel
        self.model[0].conv1 = nn.Conv2d(
            ch, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Load weights untuk layer lainnya (transfer learning)
        # Note: First layer weights perlu diisi manual atau training dari scratch

# Usage:
# model = YOLO4Channel('yolov8n.pt')
# model.train(data='ffb_localization_4ch.yaml', epochs=50, ...)
```

#### Opsi 2: Stack RGB+Depth sebagai Video (Sederhana)
- Buat data input dengan channel 4: [R, G, B, D]
- Atau: 2 frame [RGB, Depth] sebagai sequence

#### Opsi 3: Fusion Model (Advanced)
- Separate branches untuk RGB dan Depth
- Fusion layer di tengah

#### Config untuk 4-Channel:
**File:** `Experiments/ffb_localization_4ch.yaml`
```yaml
path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_localization_4ch
train: images/train
val: images/val
test: images/test

nc: 1
names: ['fresh_fruit_bunch']

# Tambahan untuk custom model
custom_input_channels: 4
```

**Data Preparation:**
```python
# Buat script untuk stack RGB + Depth menjadi 4-channel
# Experiments/scripts/prepare_4ch_data.py

import cv2
import numpy as np
import os

def create_4channel(rgb_path, depth_path, output_path):
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    # Stack menjadi 4 channel
    # Channel terakhir = depth
    if len(rgb.shape) == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    
    b, g, r = cv2.split(rgb)
    four_channel = cv2.merge([b, g, r, depth])
    
    # Simpan sebagai PNG dengan 4 channel
    cv2.imwrite(output_path, four_channel)

# Jalankan untuk semua data
```

---

## 🎯 FASE 4: Eksperimen B - Klasifikasi

### **Eksperimen B.1: RGB Only (2-Class: Ripe vs Unripe)**

#### Step 1: Siapkan Dataset Klasifikasi
- Dataset: `ffb-ripeness-classification`
- Struktur: `images/ripe/`, `images/unripe/`

#### Step 2: Split Data Klasifikasi
```python
# Experiments/scripts/split_classification_data.py
# Split 70:20:10 untuk ripe dan unripe
```

#### Step 3: Dataset Config
**File:** `Experiments/ffb_ripeness.yaml`
```yaml
path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_ripeness
train: images/train
val: images/val
test: images/test

nc: 2
names: ['ripe_ffb', 'unripe_ffb']
```

#### Step 4: Setup Config
**File:** `Experiments/config_b1_cls.yaml`
```yaml
task: classify
mode: train
model: yolov8n-cls.pt
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

#### Step 5: Training (Run #1 & #2)
```bash
# Run 1
yolo classify train config=config_b1_cls.yaml

# Run 2 (seed 123)
yolo classify train config=config_b1_cls.yaml
```

---

## 📊 FASE 5: Validasi & Reporting

### 5.1. Evaluasi Semua Model
**Script:** `Experiments/scripts/evaluate_all.py`

```python
from ultralytics import YOLO
import os

def evaluate_model(model_path, data_config, task='detect'):
    model = YOLO(model_path)
    results = model.val(data=data_config, task=task)
    
    return {
        'mAP50': results.box.map50 if task == 'detect' else results.top1,
        'mAP50-95': results.box.map if task == 'detect' else results.top5,
        'precision': results.box.mp,
        'recall': results.box.mr
    }

# Evaluasi semua eksperimen
experiments = {
    'A.1 RGB': ('runs/detect/exp_a1_rgb_baseline/weights/best.pt', 'ffb_localization.yaml'),
    'A.2 Depth': ('runs/detect/exp_a2_depth_only/weights/best.pt', 'ffb_localization_depth.yaml'),
    'B.1 Cls': ('runs/classify/exp_b1_rgb_classification/weights/best.pt', 'ffb_ripeness.yaml', 'classify'),
}

results = {}
for name, (model, data, *task) in experiments.items():
    t = task[0] if task else 'detect'
    results[name] = evaluate_model(model, data, t)

# Save to CSV
import pandas as pd
pd.DataFrame(results).to_csv('experiment_results.csv')
```

### 5.2. Analisis Kegagalan
**Script:** `Experiments/scripts/failure_analysis.py`

```python
# Cari False Positive, False Negative
# Visualisasi: Gambar asli + prediksi + ground truth
# Analisis: Pencahayaan, ukuran, occlusion
```

### 5.3. Report Template
**File:** `Experiments/LAPORAN_HASIL.md`

```markdown
# Laporan Eksperimen Deteksi TBS Kelapa Sawit

## Ringkasan Eksekusi
| Eksperimen | Run #1 (mAP50) | Run #2 (mAP50) | Rata-rata | Std Dev |
|------------|----------------|----------------|-----------|---------|
| A.1 RGB    |                |                |           |         |
| A.2 Depth  |                |                |           |         |
| B.1 Cls    |                |                |           |         |

## Analisis Kegagalan
[Contoh gambar failure + analisis]

## Kesimpulan
[Rekomendasi]
```

---

## 🚀 CEKLIST EKSEKUSI

### ✅ Tahap 1: Persiapan
- [ ] Extract dataset 28574489.zip
- [ ] Install dependencies (ultralytics, opencv, dll)
- [ ] Run `simple_eda.py`

### ✅ Tahap 2: Anotasi
- [ ] Anotasi 300-500+ gambar di AnyLabeling
- [ ] Verifikasi jumlah label = jumlah gambar

### ✅ Tahap 3: Data Processing
- [ ] Run `split_localization_data.py`
- [ ] Run `prepare_depth_data.py`
- [ ] Create `ffb_localization.yaml`

### ✅ Tahap 4: Eksperimen A.1 (RGB Baseline)
- [ ] Config: `config_a1_rgb.yaml` (seed 42)
- [ ] Training Run #1
- [ ] Config: `config_a1_rgb.yaml` (seed 123)
- [ ] Training Run #2
- [ ] Eval & Save metrics

### ✅ Tahap 5: Eksperimen A.2 (Depth Only)
- [ ] Prepare depth data
- [ ] Copy depth images ke folder split
- [ ] Config: `config_a2_depth.yaml` (seed 42)
- [ ] Training Run #1
- [ ] Config: `config_a2_depth.yaml` (seed 123)
- [ ] Training Run #2
- [ ] Eval & Save metrics

### ✅ Tahap 6: Eksperimen A.3 (RGB+Depth 4-ch)
- [ ] Create `prepare_4ch_data.py`
- [ ] Generate 4-channel dataset
- [ ] Modifikasi model
- [ ] Training 2x runs
- [ ] Eval & Save metrics

### ✅ Tahap 7: Eksperimen B.1 (Classification)
- [ ] Prepare classification dataset
- [ ] Split data
- [ ] Config: `config_b1_cls.yaml` (seed 42)
- [ ] Training Run #1
- [ ] Config: `config_b1_cls.yaml` (seed 123)
- [ ] Training Run #2
- [ ] Eval & Save metrics

### ✅ Tahap 8: Reporting
- [ ] Run `evaluate_all.py`
- [ ] Run `failure_analysis.py`
- [ ] Create `LAPORAN_HASIL.md`
- [ ] Buat presentasi

---

## 💡 Tips & Best Practices

1. **GPU:** Jika tersedia, gunakan GPU (device: 0). Training akan jauh lebih cepat.
2. **Monitor:** Gunakan `tensorboard --logdir runs/` untuk monitor training.
3. **Backup:** Simpan weights terbaik setiap eksperimen.
4. **Konsistensi:** Gunakan random seed yang sama untuk reproducbility.
5. **Disk Space:** Training 50 epoch butuh ~2-5GB per eksperimen.
6. **Interrupted Training:** Gunakan `patience` (auto-stop jika tidak improve).

---

## 🔧 Troubleshooting

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
- Gunakan model nano: `yolov8n.pt` (bukan yolov8m/l/x)
- Turunkan epochs: `epochs: 30` (untuk testing)
- Gunakan GPU!

---

**Created by:** Factory Droid  
**Last Updated:** 2026-01-15  
**Project:** Anylabel - Deteksi TBS Kelapa Sawit
