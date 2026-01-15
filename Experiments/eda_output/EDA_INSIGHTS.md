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

## Label Lokalisasi
- Folder label YOLO belum ditemukan, jadi statistik bbox (jumlah objek, ukuran bbox, aspect ratio, out-of-bounds) belum bisa dihitung.

## Ringkasan Klasifikasi (COCO)
- Total gambar: 400 (semua memiliki anotasi).
- Kategori: `Ripe-FFB` (276 anotasi), `Unripe-FFB` (1140 anotasi), `Fresh-Fruit-Bunch` (0 anotasi).
- Komposisi gambar: 17 hanya ripe, 187 hanya unripe, 196 mengandung keduanya.
- Anotasi per gambar: min 1, median 3, max 9.

## Audit Kecukupan EDA
EDA ini sudah cukup untuk **audit struktur & konsistensi file**, namun **belum cukup** untuk memenuhi EDA akademik karena:
- Lokalisasi belum punya analisis bbox (butuh label YOLO).
- Klasifikasi menunjukkan multi-objek per gambar; perlu klarifikasi apakah targetnya klasifikasi per gambar atau deteksi/klasifikasi per objek.

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
