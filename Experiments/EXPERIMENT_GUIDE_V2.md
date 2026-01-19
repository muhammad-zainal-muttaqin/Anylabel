# Panduan Eksperimen

Dokumen ini sengaja **singkat**: hanya apa yang wajib dilakukan + format pelaporan.

## 1) Persiapan Data
- **Lokalisasi (YOLO format)**: anotasi bounding box **manual** (autolabel boleh, **wajib cek manual**).
- **EDA**: lakukan EDA untuk dataset lokalisasi dan dataset kematangan.
- **Split**: buat `train/val/test`.

## 2) Model (baseline)
- Gunakan **YOLOv11n**.

## 3) Eksperimen yang diminta

### A) Single-class (Lokalisasi TBS/FFB)
- **A.1 RGB only**: deteksi 1 kelas (FFB/TBS).
- **A.2 Depth only (1→3 channel)**:
  - Normalisasi depth ke **0–255** (min-max) pada rentang **0.6m–6.0m** (kalau belum).
  - Replikasi **1 channel → 3 channel** (agar kompatibel input 3-channel).
- **A.3 RGB + Depth (4 channel)**:
  - Modifikasi input model agar menerima **4 channel** \([R,G,B,D]\).
  - Boleh coba 1–2 varian fusion.

### B) 2 kelas (ripe/unripe)
- **B.1 RGB only** (2 kelas: `ripe`, `unripe`) menggunakan dataset kematangan yang tersedia.

## 4) Metode Validasi & Pelaporan
- Buat **train/val/test split**.
- Untuk setiap eksperimen, lakukan **2 kali training run** (random seed berbeda, contoh: **42** dan **123**).
- Evaluasi pada **test set** dan laporkan:
  - **mAP50**
  - **mAP50-95**
  - **rata-rata** dari 2 run (seed 42 & 123)
- Sertakan **contoh beberapa kasus gagal** (FP/FN) sebagai kualitatif.

## 5) Output yang harus disimpan
- Folder `runs/...` (mis. `runs/detect/exp_.../`) berisi:
  - `weights/best.pt`, `results.csv`, kurva PR/F1, contoh prediksi
- Arsipkan sebagai `.zip` untuk dianalisis lokal.

## 6) Referensi file hasil di repo ini
- Ringkasan/laporan utama: `Reports/FFB_Ultimate_Report/result.md`
- Log test per eksperimen:
  - A.1: `Experiments/kaggleoutput/test.txt`
  - A.2: `Experiments/kaggleoutput/test_depth.txt`
  - A.3: `Experiments/kaggleoutput/test_4input.txt`
  - B.1: `Experiments/kaggleoutput/test_ripeness_detect.txt`
