# EDA Insights - Dataset Goh 2025

## Ringkasan Lokalisasi (Raw)
- Total file: 400 RGB, 400 depth, 400 point clouds.
- Resolusi konsisten: RGB 1280x720 (8-bit, 3-channel), depth 1280x720 (16-bit, 1-channel grayscale).
- Ukuran file konsisten: RGB 2,770,032 bytes; depth 1,846,963 bytes.
- Statistik depth (sample 100): tipe data `uint16`, rentang nilai 0–65535.

## Konsistensi Metadata
- Metadata total 406 baris, tetapi ada ketidaksesuaian ID.
- Metadata hilang untuk ID 0–9.
- Metadata ada untuk ID 400–411 namun file RGB/depth/PC tidak ada.

## Label Lokalisasi (1-Class FFB)
- **Total gambar**: 400 (split: 280 train, 80 val, 40 test)
- **Total objek**: 957 bounding boxes
- **Objek per gambar**: min=0, median=2, max=5, mean=2.39
- **BBox width (normalized)**: min=0.038, median=0.083, max=0.155
- **BBox height (normalized)**: min=0.078, median=0.181, max=0.334
- **BBox area (normalized)**: min=0.003, median=0.015, max=0.048
- **Catatan**: Objek relatif kecil-sedang (area median ~1.5% dari gambar)

## Ringkasan Klasifikasi / Ripeness Detection (2-Class)
- **Total gambar**: 400 (split: 280 train, 80 val, 40 test)
- **Total objek**: 1,416 bounding boxes
- **Distribusi kelas**: 
  - `Ripe` (class 0): 276 objek (19.5%)
  - `Unripe` (class 1): 1,140 objek (80.5%)
- **Objek per gambar**: min=1, median=3, max=9, mean=3.54
- **BBox width (normalized)**: min=0.003, median=0.095, max=0.429
- **BBox height (normalized)**: min=0.006, median=0.103, max=0.413
- **BBox area (normalized)**: min=0.000, median=0.010, max=0.177
- **⚠️ Class Imbalance**: Unripe dominan (~4:1 ratio) — perlu dipertimbangkan untuk weighted loss atau oversampling

## Audit Kecukupan EDA
EDA ini sudah **memenuhi requirement** untuk analisis dataset akademik:
- ✅ Struktur & konsistensi file
- ✅ Statistik bbox lokalisasi (jumlah, ukuran, distribusi)
- ✅ Statistik bbox klasifikasi dengan distribusi kelas
- ⚠️ Class imbalance teridentifikasi (Unripe 4:1 ratio)

## Checklist Manual yang Perlu Dicek
### Lokalisasi
- Verifikasi alignment RGB vs depth (visual 20–30 sampel).
- Cek kualitas anotasi: bbox ketat, tidak missed, konsisten di occlusion.
- Pastikan label YOLO 1:1 dengan gambar di split train/val/test.
- Cek duplikasi/near-duplicate dan variasi kondisi (pencahayaan, jarak, sudut).
- Audit file yang tidak punya metadata (ID 0–9) dan metadata tanpa file (ID 400–411).

### Klasifikasi
- Pastikan definisi label ripe/unripe konsisten (manual review 30–50 sampel).
- Jika model klasifikasi per gambar: pastikan tiap gambar hanya 1 label atau lakukan konversi (crop per objek).
- Cek ketimpangan kelas (unripe dominan) dan rencana penyeimbangan (sampling/augmentation).
- Identifikasi gambar ambiguous (ripe+unripe dalam 1 gambar) dan tentukan aturan label.
