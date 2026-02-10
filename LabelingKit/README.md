# FFB Labeling Kit

Toolkit untuk anotasi Fresh Fruit Bunch (FFB) kelapa sawit.

## Workflow Cepat

1. Jalankan `INSTALL.bat` (sekali saja).
2. Jalankan `START.bat`.
3. Di AnyLabeling pilih folder dari `Dataset\...`.

## Auto Label (Opsional)

Jalankan `AUTO_LABEL_ALL.bat` untuk pre-label otomatis satu folder.

- Input folder contoh: `Dataset\Lonsum\Kelompok 5`
- Model default (jika ada): `models\ffb_autolabel_stage1_seed42_best.pt`
- Output: file `.json` di samping gambar

## Tools Tambahan

Semua tools tambahan ada di folder `tools\`:

- `tools\CHECK_PROGRESS.bat`: cek progres dan statistik labeling
- `tools\CONVERT_TO_YOLO.bat`: konversi JSON ke YOLO TXT

## Struktur Folder (Rapih)

```text
LabelingKit/
├── INSTALL.bat
├── START.bat
├── AUTO_LABEL_ALL.bat
├── UNINSTALL.bat
├── README.md
├── Dataset/
├── models/
├── tools/
│   ├── CHECK_PROGRESS.bat
│   └── CONVERT_TO_YOLO.bat
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

- Python tidak ditemukan: simpan `python-3.12.10-embed-amd64.zip` di root `LabelingKit`, lalu jalankan `INSTALL.bat`.
- Auto-label butuh `ultralytics`: pilih `Y` saat diminta install (mode ringan `--no-deps`).
- Jika auto-label gagal import setelah install ringan, jalankan:
  - `python -m pip install torch torchvision`

## Uninstall Cepat

Jalankan `UNINSTALL.bat` untuk reset environment.

- Dihapus: `python/`, `venv/`, `output/`, cache
- Tidak dihapus: `Dataset/` dan file label user
