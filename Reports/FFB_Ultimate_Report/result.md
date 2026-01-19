## Laporan Ultimate — Outdoor RGB + Depth Dataset (FFB Lokalisasi & Kematangan)

Dokumen ini adalah versi **lebih detail** dari ringkasan `Experiments/kaggleoutput/result.md` dan dirancang untuk menjadi laporan final (lengkap, terstruktur, dan mudah ditelusuri).

### A) Ringkasan Eksekutif (1 halaman)
- **Tujuan**: membangun model YOLO untuk lokalisasi TBS/FFB dan deteksi kematangan (ripe/unripe).
- **Model baseline**: YOLOv11n.
- **Eksperimen**:
  - **A.1**: RGB only (1 kelas).
  - **A.2**: Depth only (1→3 channel).
  - **A.3**: RGB+Depth (4 channel).
  - **B.1**: Ripeness (2 kelas: ripe/unripe).
- **Metode validasi**: 2 run per eksperimen (seed 42 & 123), metrik **test**: mAP50 & mAP50-95.

### B) Dataset & Struktur (yang dipakai training)
- **RGB lokalisasi**: `Experiments/UploadKaggle/ffb_localization/`
- **Depth lokalisasi (depth-only)**: `Experiments/UploadKaggle/ffb_localization_depth/`
- **RGB+Depth (4 channel)**: `Experiments/UploadKaggle/ffb_localization_rgbd/`
- **Ripeness detect (2 kelas)**: `Experiments/UploadKaggle/ffb_ripeness_detect/`

Struktur YOLO:
- `images/{train,val,test}/*.png|*.jpg`
- `labels/{train,val,test}/*.txt`

### C) Preprocessing Depth (A.2)
#### C.1 Raw depth (dataset jurnal)
Depth asli dari dataset jurnal tersimpan sebagai PNG 16-bit (RealSense `z16`), nilai umum dalam **milimeter**.

#### C.2 Preprocessing yang dipakai untuk dataset training depth-only (versi terbaru)
Script yang bertanggung jawab:
- `Experiments/scripts/prepare_depth_data.py`
- `Experiments/scripts/build_uploadkaggle_depth_only.py`

Aturan preprocessing depth yang diterapkan (sesuai permintaan):
- **uint16 → mm → meter**: `depth_m = depth_u16 / 1000.0`
- **invalid**: nilai `0` dan `65535` dianggap invalid
- **clip** ke rentang \([0.6, 6.0]\) meter
- **scale linear** ke `uint8` \([0, 255]\)
- **replicate** 1 channel → 3 channel (agar kompatibel input YOLO 3-channel)

Contoh snippet (inti logika; bahasa Inggris):

```python
depth_m = depth_u16.astype(np.float32) / 1000.0
depth_m[depth_u16 == 0] = np.nan
depth_m[depth_u16 == 65535] = np.nan
depth_m = np.clip(depth_m, 0.6, 6.0)
scaled = (depth_m - 0.6) / (6.0 - 0.6)  # 0..1
scaled = np.where(np.isfinite(scaled), scaled, 0.0)  # invalid -> 0
depth_u8 = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
depth_3ch = cv2.merge([depth_u8, depth_u8, depth_u8])
```

#### C.3 Catatan kompatibilitas hasil lama A.2
Hasil A.2 yang pernah dicatat sebelumnya kemungkinan berasal dari normalisasi **min-max per-gambar** (lebih “mudah” secara visual tetapi tidak mengunci skala meter).
Setelah perbaikan preprocessing fixed-range \([0.6, 6.0]\) m, hasil A.2 telah diulang (lihat `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test_depth.txt`).

### D) Protokol Training & Evaluasi (Kaggle)
Umum:
- 2 seed: **42** dan **123**
- epoch: 50
- imgsz: 640
- batch: 16
- evaluasi: `split=test`

Contoh code (bahasa Inggris; Kaggle):

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="/kaggle/working/ffb_localization.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    seed=42,
    project="/kaggle/working/runs/detect",
    name="exp_a1_rgb_seed42",
)

YOLO("/kaggle/working/runs/detect/exp_a1_rgb_seed42/weights/best.pt").val(
    data="/kaggle/working/ffb_localization.yaml",
    split="test",
    project="/kaggle/working/runs/detect",
    name="test_seed42",
)
```

## Ringkasan Hasil Kuantitatif (TEST) — Tabel mAP50 & mAP50-95

> **Sumber Log & Artefak:**
> - `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test.txt`
> - `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test_depth.txt`
> - `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test_4input.txt`
> - `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test_ripeness_detect.txt`
> - Artefak kurva/CSV: `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/kaggle/working/runs/detect/...`

Tabel di bawah ini merangkum hasil pengujian model YOLO pada berbagai eksperimen.  
Ada 4 eksperimen utama:
- **A.1**: Deteksi FFB dengan gambar RGB saja.
- **A.2**: Deteksi FFB dengan gambar depth saja (diubah menjadi 3 channel).
- **A.3**: Deteksi FFB gabungan gambar RGB + depth (total 4 channel).
- **B.1**: Deteksi tingkat kematangan (2 kelas: matang & mentah) menggunakan gambar RGB.

Setiap eksperimen dilakukan dua kali (dengan seed 42 dan 123) untuk memastikan hasilnya konsisten.  
Nilai yang dilaporkan:
- **mAP50**: rata-rata presisi untuk IOU ≥ 0.50 (semakin besar semakin baik)
- **mAP50-95**: rata-rata presisi untuk IOU dari 0.50 sampai 0.95 (lebih ketat)

| Kode | Eksperimen               | Seed | mAP50   | mAP50-95 |
|------|-------------------------|------|---------|----------|
| A.1  | RGB saja (1 kelas)      | 42   | 0.873   | 0.370    |
| A.1  | RGB saja (1 kelas)      | 123  | 0.873   | 0.369    |
|      | **Rata-rata A.1**       | -    | 0.87300 | 0.36950  |
| A.2  | Depth saja (1→3 ch)     | 42   | 0.69304 | 0.26463  |
| A.2  | Depth saja (1→3 ch)     | 123  | 0.70972 | 0.26283  |
|      | **Rata-rata A.2**       | -    | 0.70138 | 0.26373  |
| A.3  | RGB+Depth (4 channel)   | 42   | 0.875   | 0.378    |
| A.3  | RGB+Depth (4 channel)   | 123  | 0.862   | 0.380    |
|      | **Rata-rata A.3**       | -    | 0.86850 | 0.37900  |
| B.1  | Deteksi kematangan (2k) | 42   | 0.804   | 0.511    |
| B.1  | Deteksi kematangan (2k) | 123  | 0.797   | 0.517    |
|      | **Rata-rata B.1**       | -    | 0.80050 | 0.51400  |

Keterangan:
- "Rata-rata" adalah nilai rata-rata dari dua seed.
- Nilai pada kolom "Seed" adalah angka acak yang digunakan saat training untuk menguji konsistensi.
- "k" pada "2k" maksudnya "2 kelas".

---

### **Highlight Utama:**
- **A.3 (RGB+Depth 4-ch):** mAP50 tertinggi (**0.8685**), menunjukkan benefit dari input multimodal.
- **A.1 (RGB Only):** Sangat tinggi juga (**0.873**), mendekati A.3.
- **A.2 (Depth Only):** Setelah preprocessing fixed-range, mAP50 rata-rata menjadi **0.70138** (lebih baik dari baseline lama).
- **B.1 (Ripeness):** mAP50-95 relatif tinggi (**0.514**), cukup promising untuk 2 kelas.

---

### **Catatan Penting**
- **A.2** pada tabel ini sudah menggunakan hasil training ulang depth-only (preprocessing fixed-range \([0.6, 6.0]\) m) dari `.../test_depth.txt`.
- Hasil rata-rata dihitung dari 2 seed (**42** dan **123**), sesuai protokol.

---


### F) Bukti Kualitatif (gambar) — template penempelan
Tempel minimal:
- 2 contoh **TP** (deteksi benar)
- 2 contoh **FN** (FFB tidak terdeteksi)
- 2 contoh **FP** (deteksi salah)

Saran penempatan:
- Simpan gambar ke `Reports/FFB_Ultimate_Report/assets/` lalu link-kan di sini.

Contoh format:
- `![A.1 TP example](assets/a1_tp_01.png)`

### G) Lampiran Artefak (log, CSV, zip)
#### G.1 Log ringkas yang sudah ada (lokal)
- `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test.txt`
- `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test_depth.txt`
- `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test_4input.txt`
- `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/test_ripeness_detect.txt`

#### G.2 CSV hasil training (lokal)
- `Reports/FFB_Ultimate_Report/artifacts/kaggleoutput/kaggle/working/runs/detect/**/results.csv`

#### G.3 Zip runs (dari Kaggle)
Simpan zip per run, misalnya:
- `exp_a1_rgb_seed42_train.zip`, `exp_a1_rgb_seed42_test.zip`, dst.
Letakkan di `Reports/FFB_Ultimate_Report/logs/` atau folder terpisah (jangan di-commit jika terlalu besar).

### H) Ringkasan Kendala & Perbaikan (singkat)
- **Preprocessing depth awal tidak sesuai skala meter**: sempat memakai normalisasi **min-max per-gambar** sehingga tidak merefleksikan rentang fisik \(0.6–6.0m\).
- **Perbaikan preprocessing depth**: implementasi preprocessing fixed-range yang benar (uint16 \(mm\) → meter `/1000`, invalid `0/65535`, clip \(0.6–6.0m\), skala linear 0–255, replikasi 1→3 channel) lalu dataset depth-only dibangun ulang.
- **Isu struktur dataset/YAML di Kaggle**: path/split yang tidak mengikuti struktur standar menyebabkan “no labels found/num_samples=0”; diselesaikan dengan struktur `images/{split}` + `labels/{split}` dan YAML yang konsisten.
- **Isu training RGB+Depth (4-channel)**: perlu modifikasi pipeline (load pasangan RGB+depth, transform sinkron, concat 4 channel, adapt conv pertama) serta penanganan beberapa error runtime (mis. validator stride, sampling mosaic/buffer) agar training stabil di Kaggle.

