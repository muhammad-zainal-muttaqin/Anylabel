import os
import shutil
import random
import glob

# --- KONFIGURASI ---
# Folder sumber dataset (sesuaikan jika lokasi berubah)
SOURCE_DATASET_DIR = r"D:\Work\Assisten Dosen\Anylabel\Dataset\gohjinyu-oilpalm-ffb-dataset-d66eb99\ffb-localization"
SOURCE_IMAGES_DIR = os.path.join(SOURCE_DATASET_DIR, "rgb_images")
OUTPUT_BASE_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments\datasets\ffb_localization"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELING_DIR = os.path.join(PROJECT_DIR, "labeling", "ffb_localization")
LABELS_ALL_DIR = os.path.join(LABELING_DIR, "yolo_all")
SOURCE_LABELS_DIRS = [LABELS_ALL_DIR, os.path.join(SOURCE_DATASET_DIR, "labels_yolo")]

# Rasio pembagian
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Random Seed untuk konsistensi (sesuai instruksi)
SEED = 42

def setup_directories():
    """Membuat struktur folder datasets/images dan datasets/labels"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'labels', split), exist_ok=True)
    os.makedirs(LABELS_ALL_DIR, exist_ok=True)
    print(f"Folder struktur dibuat di: {OUTPUT_BASE_DIR}")

def split_dataset():
    """Membagi dataset menjadi train/val/test dan menyalin file"""
    
    # 1. Ambil list semua file gambar
    # Mencari .png dan .jpg
    image_files = glob.glob(os.path.join(SOURCE_IMAGES_DIR, "*.png")) + \
                  glob.glob(os.path.join(SOURCE_IMAGES_DIR, "*.jpg"))
    
    total_images = len(image_files)
    if total_images == 0:
        print(f"Error: Tidak ditemukan gambar di {SOURCE_IMAGES_DIR}")
        return

    print(f"Menemukan {total_images} gambar.")

    # 2. Acak urutan (Shuffle)
    random.seed(SEED)
    random.shuffle(image_files)

    # 3. Hitung jumlah data per split
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)
    # Sisanya untuk test
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count : train_count + val_count]
    test_files = image_files[train_count + val_count:]

    print(f"Pembagian: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # 4. Fungsi helper untuk copy
    def copy_files(file_list, split_name):
        print(f"Menyalin data {split_name}...")
        for img_path in file_list:
            # Info File
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            label_path = None
            for label_dir in SOURCE_LABELS_DIRS:
                candidate = os.path.join(label_dir, basename + ".txt")
                if os.path.exists(candidate):
                    label_path = candidate
                    break

            # Destination Paths
            dest_img_path = os.path.join(OUTPUT_BASE_DIR, 'images', split_name, filename)
            dest_label_path = os.path.join(OUTPUT_BASE_DIR, 'labels', split_name, basename + ".txt")

            # Copy Image
            shutil.copy2(img_path, dest_img_path)

            # Copy Label (Jika ada)
            if label_path and os.path.exists(label_path):
                shutil.copy2(label_path, dest_label_path)
            else:
                # Warning kalau label ga ada (normal jika belum anotasi semua)
                pass

    # 5. Eksekusi Copy
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print("\nSelesai! Dataset siap digunakan untuk training.")
    print(f"Lokasi: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    has_any_labels = False
    for label_dir in SOURCE_LABELS_DIRS:
        if os.path.exists(label_dir) and glob.glob(os.path.join(label_dir, "*.txt")):
            has_any_labels = True
            break
    if not has_any_labels:
        print("Peringatan: Folder label belum ada atau masih kosong.")
        print("Script ini tetap akan jalan mengcopy gambar, tapi tanpa label.")
        print("Pastikan Anda sudah melakukan anotasi (Tahap 1.1) sebelum training serius.")
    
    setup_directories()
    split_dataset()
