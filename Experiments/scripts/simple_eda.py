import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# --- KONFIGURASI ---
DATASET_DIR = r"D:\Work\Assisten Dosen\Anylabel\Dataset\gohjinyu-oilpalm-ffb-dataset-d66eb99"
LOCALIZATION_IMG_DIR = os.path.join(DATASET_DIR, "ffb-localization", "rgb_images")
LOCALIZATION_LABEL_DIR = os.path.join(DATASET_DIR, "ffb-localization", "labels_yolo") # Asumsi ada di sini nanti
RIPENESS_DIR = os.path.join(DATASET_DIR, "ffb-ripeness-classification")

def analyze_localization():
    print("--- ANALISIS DATASET LOKALISASI ---")
    
    # Cek Gambar
    images = glob.glob(os.path.join(LOCALIZATION_IMG_DIR, "*.png"))
    print(f"Jumlah Gambar: {len(images)}")
    
    # Cek Label
    if os.path.exists(LOCALIZATION_LABEL_DIR):
        labels = glob.glob(os.path.join(LOCALIZATION_LABEL_DIR, "*.txt"))
        print(f"Jumlah Label : {len(labels)}")
        print(f"Status Anotasi: {len(labels)}/{len(images)} ({(len(labels)/len(images))*100:.1f}%)")
    else:
        print("Folder Label belum ditemukan (Belum ada anotasi).")

def analyze_ripeness():
    print("\n--- ANALISIS DATASET KLASIFIKASI KEMATANGAN ---")
    
    # Cek Json COCO
    coco_path = os.path.join(RIPENESS_DIR, "_annotations.coco.json")
    if not os.path.exists(coco_path):
        print("Annotation file missing!")
        return

    with open(coco_path, 'r') as f:
        data = json.load(f)
    
    print(f"Jumlah Gambar (Metadata): {len(data['images'])}")
    print("Kategori:")
    for cat in data['categories']:
        print(f"  - ID {cat['id']}: {cat['name']} (Parent: {cat['supercategory']})")
        
    # Hitung Distribusi
    # category_id 1 = Ripe, 2 = Unripe (Cek ulang json)
    ripe_count = 0
    unripe_count = 0
    
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id == 1:
            ripe_count += 1
        elif cat_id == 2:
            unripe_count += 1
            
    print(f"Distribusi Anotasi:")
    print(f"  Ripe-FFB   : {ripe_count}")
    print(f"  Unripe-FFB : {unripe_count}")

if __name__ == "__main__":
    analyze_localization()
    analyze_ripeness()
    print("\nCheck Selesai.")
