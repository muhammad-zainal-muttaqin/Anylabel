# FFB Labeling Kit

Toolkit untuk anotasi Fresh Fruit Bunch (FFB) kelapa sawit.

## Workflow Cepat (Manual Labeling)

1. Jalankan `INSTALL.bat` (sekali saja).
2. Jalankan `START.bat`.
3. Di AnyLabeling pilih folder `images` dari kelompok yang ingin dilabel.
4. Kelas default otomatis muncul: `B1`, `B2`, `B3`, `B4`.

Contoh:
- `Dataset\Damimas\Kelompok 1\images`
- `Dataset\Lonsum\Kelompok 5\images`

## Auto Label (Opsional)

Jalankan `AUTO_LABEL_ALL.bat` untuk pre-label otomatis satu folder.

- Saat diminta `Folder gambar`, isi folder `images`, contoh:
  - `Dataset\Lonsum\Kelompok 5\images`
- Model default (jika ada): `models\ffb_autolabel_stage1_seed42_best.pt`
- Confidence: tekan `Enter` untuk default `0.25`
- Output: file `.json` dibuat di samping gambar (di folder `images`)

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
- Jika setelah install ringan muncul warning dependency, itu normal. Lanjutkan sampai prompt folder gambar muncul.
- Jika auto-label tetap gagal import, install dependency ini:
  - `python\python.exe -m pip install torch torchvision`

## Uninstall Cepat

Jalankan `UNINSTALL.bat` untuk reset environment.

- Dihapus: `python/`, `venv/`, `output/`, cache
- Tidak dihapus: `Dataset/` dan file label user
