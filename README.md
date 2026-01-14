# AnyLabeling Setup

Proyek ini menggunakan [AnyLabeling](https://github.com/vietanhdev/anylabeling) untuk anotasi gambar dengan dukungan AI.

## 📋 Prerequisites

- **Python 3.10+** (disarankan Python 3.12)
- **Windows 10/11**

## 🚀 Instalasi

### 1. Install Python (jika belum ada)

Menggunakan Python Install Manager:

```powershell
# Install Python Install Manager dari Microsoft Store atau python.org
# Kemudian install Python 3.12
pymanager install 3.12
```

### 2. Buat Virtual Environment

```powershell
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
.\venv\Scripts\Activate
```

### 3. Install AnyLabeling

```powershell
# Install AnyLabeling (versi CPU)
pip install anylabeling

# ATAU untuk versi GPU (jika punya NVIDIA GPU dengan CUDA)
pip install anylabeling-gpu
```

## ▶️ Menjalankan AnyLabeling

```powershell
# Pastikan virtual environment aktif
.\venv\Scripts\Activate

# Jalankan AnyLabeling
anylabeling
```

## 📁 Struktur Folder

```
Anylabel/
├── venv/               # Virtual environment (tidak di-commit)
├── images/             # Folder untuk gambar yang akan dilabeli
├── labels/             # Folder output label (YOLO, COCO, dll)
├── .gitignore
└── README.md
```

## 🎯 Fitur AnyLabeling

- ✅ Auto-labeling dengan AI models (YOLO, SAM, dll)
- ✅ Manual annotation tools
- ✅ Export ke berbagai format (YOLO, COCO, Pascal VOC)
- ✅ Support untuk object detection, segmentation, classification

## 🔧 Troubleshooting

### PowerShell Script Execution Error

Jika mendapat error saat aktivasi venv:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### pip tidak ditemukan

Gunakan:

```powershell
python -m pip install anylabeling
```

## 📚 Dokumentasi

- [AnyLabeling GitHub](https://github.com/vietanhdev/anylabeling)
- [AnyLabeling Documentation](https://anylabeling.nrl.ai/)

---

*Dibuat untuk keperluan Asisten Dosen*
