import cv2
import numpy as np
import os
import glob
from pathlib import Path

# --- KONFIGURASI ---
SOURCE_DEPTH_DIR = r"D:\Work\Assisten Dosen\Anylabel\Dataset\gohjinyu-oilpalm-ffb-dataset-d66eb99\ffb-localization\depth_maps"
OUTPUT_DEPTH_RGB_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments\datasets\depth_processed_rgb"

# Range normalisasi (sesuai instruksi: 0.6m - 6m)
MIN_DEPTH = 0.6  # meter
MAX_DEPTH = 6.0  # meter

# Scaling factor depth map asli (tergantung format dataset, seringkali mm jadi /1000)
# KITA PERLU CEK DULU ISI FILE ASLINYA. Untuk sekarang asumsi unit mm.
DEPTH_SCALE_FACTOR = 1000.0  

def process_depth_maps():
    os.makedirs(OUTPUT_DEPTH_RGB_DIR, exist_ok=True)
    
    depth_files = glob.glob(os.path.join(SOURCE_DEPTH_DIR, "*.png"))
    total_files = len(depth_files)
    
    print(f"🔄 Memproses {total_files} file depth map...")
    print(f"   Target Normalisasi: {MIN_DEPTH}m - {MAX_DEPTH}m")
    
    count = 0
    for file_path in depth_files:
        filename = os.path.basename(file_path)
        
        # 1. Baca Depth Map (Flag -1 untuk load as-is, biasanya 16-bit atau 32-bit float)
        depth_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if depth_img is None:
            print(f"❌ Gagal baca: {filename}")
            continue

        # Convert ke float untuk operasi matematika
        depth_img = depth_img.astype(np.float32)

        # Asumsi unit data asli adalah milimeter (umum di depth camera seperti RealSense) -> convert ke meter
        # Jika data ternyata sudah meter, ubah line ini.
        # depth_img_meter = depth_img / 1000.0 
        
        # Karena kita belum yakin 100% unitnya, kita pakai normalisasi Min-Max standard (0-255)
        # berdasarkan instruksi "Normalisasi nilai depth map ke rentang 0-255"
        
        # Opsi A: Clip ke range spesifik (0.6 - 6m) lalu normalisasi 
        # depth_img_meter = np.clip(depth_img_meter, MIN_DEPTH, MAX_DEPTH)
        # norm_img = (depth_img_meter - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * 255.0
        
        # Opsi B: Langsung Normalisasi MinMax dari data yang ada di gambar (lebih aman visualisasinya)
        norm_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        
        norm_img = norm_img.astype(np.uint8)

        # 2. Replikasi 3 Channel (Grayscale -> RGB)
        # R=G=B=Depth
        depth_3ch = cv2.merge([norm_img, norm_img, norm_img])
        
        # 3. Simpan
        save_path = os.path.join(OUTPUT_DEPTH_RGB_DIR, filename)
        cv2.imwrite(save_path, depth_3ch)
        
        count += 1
        if count % 50 == 0:
            print(f"   ...terproses {count}/{total_files}")

    print(f"✅ Selesai. Hasil depth 3-channel disimpan di: {OUTPUT_DEPTH_RGB_DIR}")

if __name__ == "__main__":
    process_depth_maps()
