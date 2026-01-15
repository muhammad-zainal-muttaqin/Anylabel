"""
Eksperimen A.2: Depth Only
Train YOLOv8n pada dataset Depth (3-channel replicated)
"""

import subprocess
import yaml
import os
import shutil
import glob

# Configurations
CONFIG_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments"
DATASET_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments\datasets"
DEPTH_SOURCE = os.path.join(DATASET_DIR, "depth_processed_rgb")
DEPTH_TARGET = os.path.join(DATASET_DIR, "ffb_localization_depth")

BASE_CONFIG = {
    'task': 'detect',
    'mode': 'train',
    'model': 'yolov8n.pt',
    'data': 'ffb_localization_depth.yaml',
    'epochs': 50,
    'patience': 10,
    'batch': 16,
    'imgsz': 640,
    'save': True,
    'device': 0,
    'workers': 4,
    'project': 'runs/detect',
    'name': 'exp_a2_depth_only'
}

SEEDS = [42, 123]

def prepare_depth_dataset():
    """Copy depth images to replace RGB images in train/val/test folders"""
    
    print("📁 Menyiapkan dataset depth...")
    
    # Create depth dataset structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DEPTH_TARGET, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DEPTH_TARGET, 'labels', split), exist_ok=True)
    
    # Get all depth images
    depth_images = glob.glob(os.path.join(DEPTH_SOURCE, "*.png"))
    
    if not depth_images:
        print(f"❌ Tidak ada depth images di {DEPTH_SOURCE}")
        print("   Jalankan prepare_depth_data.py terlebih dahulu!")
        return False
    
    # Copy depth images to corresponding folders
    # Asumsi: File di depth_processed_rgb sudah siap
    # Perlu mapping filename ke split (train/val/test)
    
    # Cek mapping dari dataset asli
    original_splits = {
        'train': glob.glob(os.path.join(DATASET_DIR, "ffb_localization", "images", "train", "*.png")),
        'val': glob.glob(os.path.join(DATASET_DIR, "ffb_localization", "images", "val", "*.png")),
        'test': glob.glob(os.path.join(DATASET_DIR, "ffb_localization", "images", "test", "*.png"))
    }
    
    for split, files in original_splits.items():
        print(f"  Processing {split}: {len(files)} files")
        
        for file in files:
            filename = os.path.basename(file)
            depth_file = os.path.join(DEPTH_SOURCE, filename)
            
            if os.path.exists(depth_file):
                # Copy depth image
                dest = os.path.join(DEPTH_TARGET, 'images', split, filename)
                shutil.copy2(depth_file, dest)
                
                # Copy label (same as original)
                label_src = os.path.join(DATASET_DIR, "ffb_localization", "labels", split, 
                                        filename.replace('.png', '.txt'))
                label_dest = os.path.join(DEPTH_TARGET, 'labels', split, 
                                         filename.replace('.png', '.txt'))
                
                if os.path.exists(label_src):
                    shutil.copy2(label_src, label_dest)
            else:
                print(f"  ⚠️ Warning: Depth file not found for {filename}")
    
    print("✅ Dataset depth siap!")
    return True

def create_depth_config():
    """Create dataset config for depth"""
    
    config_content = f"""# Dataset Configuration for FFB Localization - DEPTH ONLY
# Path: Experiments/ffb_localization_depth.yaml

path: {DEPTH_TARGET.replace('\\', '/')}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['fresh_fruit_bunch']
"""
    
    config_path = os.path.join(CONFIG_DIR, "ffb_localization_depth.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Config depth created: {config_path}")
    return config_path

def train_with_seed(seed):
    """Train with specific seed"""
    
    config = BASE_CONFIG.copy()
    config['seed'] = seed
    config['name'] = f'exp_a2_depth_seed_{seed}'
    
    config_filename = f'config_a2_depth_seed_{seed}.yaml'
    config_path = os.path.join(CONFIG_DIR, config_filename)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING A.2 DEPTH ONLY - Seed {seed}")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Output: runs/detect/{config['name']}")
    print(f"{'='*60}\n")
    
    command = f"cd {CONFIG_DIR} && yolo detect train config={config_filename}"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ Training completed for seed {seed}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for seed {seed}: {e}")
        return False

def main():
    print("🔬 Eksperimen A.2: Depth Only")
    print("="*60)
    
    # Step 1: Prepare depth dataset
    if not prepare_depth_dataset():
        return
    
    # Step 2: Create config
    create_depth_config()
    
    # Step 3: Train with both seeds
    results = []
    for seed in SEEDS:
        success = train_with_seed(seed)
        results.append((seed, success))
    
    # Summary
    print("\n" + "="*60)
    print("📊 RINGKASAN EKSEKUSI")
    print("="*60)
    for seed, success in results:
        status = "✅ BERHASIL" if success else "❌ GAGAL"
        print(f"Seed {seed}: {status}")
    
    print("\n📁 Hasil training:")
    for seed in SEEDS:
        print(f"  - Seed {seed}: runs/detect/exp_a2_depth_seed_{seed}/")
    
    print("\n💡 Untuk evaluasi:")
    print("  yolo detect val model=runs/detect/exp_a2_depth_seed_42/weights/best.pt data=ffb_localization_depth.yaml split=test")

if __name__ == "__main__":
    main()
