# FFB Labeling Kit

Toolkit untuk anotasi Fresh Fruit Bunch (FFB) kelapa sawit.

## Workflow Cepat (Untuk User Non-Tech)

Ikuti urutan file bernomor:

1. Jalankan `1_INSTALL.bat` (sekali saja)
2. Jalankan `2_START_LABELING.bat`
3. Di AnyLabeling pilih folder `images` dari kelompok yang ingin dilabel
4. Kelas default otomatis muncul: `B1`, `B2`, `B3`, `B4`

Contoh:
- `Dataset\Damimas\Kelompok 1\images`
- `Dataset\Lonsum\Kelompok 5\images`

## Labeling Video (Disarankan)

AnyLabeling di workflow ini digunakan untuk **folder gambar**, bukan file video langsung.

Gunakan alur berikut:

1. Ekstrak video ke frame gambar dengan `tools\EXTRACT_VIDEO_FRAMES.bat`
2. Hasil default ada di:
   - `Dataset\Video\<nama_video>\images`
3. Jalankan `2_START_LABELING.bat`
4. Di AnyLabeling, buka folder `images` hasil ekstraksi
5. (Opsional) Jalankan `AUTO_LABEL_ALL (opsional).bat` untuk pre-label
6. Review dan koreksi label manual
7. Konversi JSON ke YOLO dengan `tools\CONVERT_TO_YOLO.bat`

Saran awal ekstraksi:
- `FPS=1` sampai `FPS=3` untuk video kebun normal
- Mulai dari `FPS=2`, naikkan jika objek bergerak cepat atau scene berubah drastis

## Auto Label (Opsional)

Jalankan `AUTO_LABEL_ALL (opsional).bat` untuk pre-label otomatis satu folder.

- Saat diminta `Folder gambar`, isi folder `images`, contoh:
  - `Dataset\Lonsum\Kelompok 5\images`
- Model default (jika ada): `models\ffb_autolabel_stage1_seed42_best.pt`
- Confidence: tekan `Enter` untuk default `0.25`
- Output: file `.json` dibuat di samping gambar (di folder `images`)
- Mapping label auto:
  - `ripe` otomatis jadi `B1`
  - `unripe` otomatis jadi `B2`
  - `B3`/`B4` disesuaikan manual saat review

Urutan prompt `AUTO_LABEL_ALL.bat`:
1. Install `ultralytics`? (`Y/N`)
2. Jika perlu, install `torch + torchvision`? (`Y/N`)
3. Folder gambar
4. Path model YOLO
5. Confidence

## Tools Tambahan

Semua tools tambahan ada di folder `tools\`:

- `tools\CHECK_PROGRESS.bat`: cek progres dan statistik labeling
- `tools\CONVERT_TO_YOLO.bat`: konversi JSON ke YOLO TXT
- `tools\EXTRACT_VIDEO_FRAMES.bat`: ekstrak video ke frame gambar (`images`)

### Mode Konversi JSON -> YOLO

`tools\CONVERT_TO_YOLO.bat` sekarang dikunci ke mode **Ripeness 4 kelas**:
- Kelas dibaca dari `_internal\configs\classes.txt`
- Fallback default: `B1,B2,B3,B4`

## Struktur Folder (Rapih)

```text
LabelingKit/
├── 1_INSTALL.bat
├── 2_START_LABELING.bat
├── AUTO_LABEL_ALL (opsional).bat
├── UNINSTALL (opsional).bat
├── README.md
├── Dataset/
├── models/
├── tools/
│   ├── CHECK_PROGRESS.bat
│   ├── CONVERT_TO_YOLO.bat
│   └── EXTRACT_VIDEO_FRAMES.bat
└── _internal/
    ├── requirements.txt
    ├── configs/
    └── scripts/
```

## Keyboard Shortcuts AnyLabeling

- `R`: Rectangle tool
- `D`: gambar selanjutnya
- `A`: gambar sebelumnya
- `Ctrl+S`: simpan
- `Del`: hapus label terpilih

## Troubleshooting

- Python tidak ditemukan: simpan `python-3.12.10-embed-amd64.zip` di root `LabelingKit`, lalu jalankan `1_INSTALL.bat`.
- Auto-label butuh `ultralytics`: pilih `Y` saat diminta install (mode ringan `--no-deps`).
- Jika setelah install ringan muncul warning dependency, itu normal. Lanjutkan sampai prompt folder gambar muncul.
- Jika auto-label tetap gagal import, install dependency ini:
  - `python\python.exe -m pip install torch torchvision`

## Uninstall Cepat

Jalankan `UNINSTALL (opsional).bat` untuk reset environment.

- Dihapus: `python/`, `venv/`, `output/`, cache
- Tidak dihapus: `Dataset/` dan file label user
