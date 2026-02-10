# FFB Labeling Kit

Toolkit untuk anotasi Fresh Fruit Bunch (FFB) kelapa sawit.

---

## Cara Pakai

### 1. Install (Sekali Saja)
```
Klik 2x: INSTALL.bat
```
Tunggu sampai selesai (~5-10 menit).

### 2. Mulai Labeling
```
Klik 2x: START.bat
```

### 3. Pilih Folder Kelompok
Di AnyLabeling:
1. **File → Open Dir**
2. Pilih folder kelompok kamu, contoh:
   - `Dataset\Damimas\Kelompok 1`
   - `Dataset\Lonsum\Kelompok 5`
3. Mulai labeling!

---

## Cara Labeling

1. Tekan **R** untuk Rectangle tool
2. Buat kotak di setiap **buah sawit (FFB)**
3. Isi label: `fresh_fruit_bunch`
4. Tekan **Ctrl+S** untuk simpan
5. Tekan **D** untuk gambar selanjutnya

**Ulangi sampai semua gambar selesai!**

---

## Struktur Folder

```
LabelingKit/
├── INSTALL.bat            ← Jalankan pertama kali
├── START.bat              ← Jalankan untuk labeling
├── Dataset/
│   ├── Damimas/
│   │   ├── Kelompok 1/
│   │   ├── Kelompok 2/
│   │   ├── Kelompok 3/
│   │   ├── Kelompok 4/
│   │   ├── Kelompok 5/
│   │   └── Kelompok 6/
│   └── Lonsum/
│       ├── Kelompok 2/
│       ├── Kelompok 4/
│       └── Kelompok 5/
└── ...
```

---

## Keyboard Shortcuts

| Tombol | Fungsi |
|--------|--------|
| **R** | Rectangle tool |
| **D** | Gambar selanjutnya |
| **A** | Gambar sebelumnya |
| **Ctrl+S** | Simpan |
| **Del** | Hapus label terpilih |

---

## Aturan Labeling

1. Label **SEMUA** buah sawit yang terlihat
2. Buat kotak **pas** mengelilingi buah (tidak terlalu besar/kecil)
3. Jika buah **>50% terlihat**, tetap label
4. **Simpan (Ctrl+S)** setiap selesai 1 gambar

---

## Troubleshooting

**"Python tidak ditemukan"**
- Pastikan file `python-3.12.10-embed-amd64.zip` ada di folder
- Jalankan INSTALL.bat lagi

**"Belum diinstall"**
- Jalankan INSTALL.bat terlebih dahulu

---

*FFB Labeling Kit - Oil Palm Research*

---

## Uninstall Cepat

Jika ingin reset instalasi dengan cepat:
```
Klik 2x: UNINSTALL.bat
```

`UNINSTALL.bat` menghapus:
- `python/`
- `venv/`
- `output/`
- cache Python

`UNINSTALL.bat` **tidak** menghapus:
- `Dataset/`
- file label user (`.json`, `.txt`)
