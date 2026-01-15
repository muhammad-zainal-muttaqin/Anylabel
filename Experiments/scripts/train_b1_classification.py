"""
Eksperimen B.1: RGB Only (Classification)
Train YOLOv8n-cls untuk klasifikasi ripe vs unripe
"""

import subprocess
import yaml
import os

# Configurations
CONFIG_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments"
DATASET_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments\datasets"

BASE_CONFIG = {
    'task': 'classify',
    'mode': 'train',
    'model': 'yolov8n-cls.pt',
    'data': 'ffb_ripeness.yaml',
    'epochs': 50,
    'patience': 10,
    'batch': 32,  # Higher batch for classification
    'imgsz': 224,  # Classification typically uses 224
    'save': True,
    'device': 0,
    'workers': 4,
    'project': 'runs/classify',
    'name': 'exp_b1_rgb_classification'
}

SEEDS = [42, 123]

def check_classification_dataset():
    """Check if classification dataset exists"""
    
    dataset_path = os.path.join(DATASET_DIR, "ffb_ripeness")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset klasifikasi tidak ditemukan: {dataset_path}")
        print("\n📂 Struktur yang dibutuhkan:")
        print(f"   {dataset_path}/")
        print(f"   ├── images/train/ripe/")
        print(f"   ├── images/train/unripe/")
        print(f"   ├── images/val/ripe/")
        print(f"   ├── images/val/unripe/")
        print(f"   ├── images/test/ripe/")
        print(f"   └── images/test/unripe/")
        return False
    
    # Cek subfolder
    for split in ['train', 'val', 'test']:
        for class_name in ['ripe', 'unripe']:
            path = os.path.join(dataset_path, 'images', split, class_name)
            if not os.path.exists(path):
                print(f"⚠️ Folder tidak ada: {path}")
                return False
    
    print("✅ Dataset klasifikasi ditemukan!")
    return True

def create_classification_config():
    """Create dataset config for classification"""
    
    config_content = """# Dataset Configuration for FFB Ripeness Classification
# Path: Experiments/ffb_ripeness.yaml

path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_ripeness
train: images/train
val: images/val
test: images/test

nc: 2
names: ['ripe_ffb', 'unripe_ffb']
"""
    
    config_path = os.path.join(CONFIG_DIR, "ffb_ripeness.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Config klasifikasi created: {config_path}")
    return config_path

def train_with_seed(seed):
    """Train classification with specific seed"""
    
    config = BASE_CONFIG.copy()
    config['seed'] = seed
    config['name'] = f'exp_b1_cls_seed_{seed}'
    
    config_filename = f'config_b1_cls_seed_{seed}.yaml'
    config_path = os.path.join(CONFIG_DIR, config_filename)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING B.1 CLASSIFICATION - Seed {seed}")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Output: runs/classify/{config['name']}")
    print(f"Model: yolov8n-cls.pt")
    print(f"Image size: 224x224")
    print(f"{'='*60}\n")
    
    command = f"cd {CONFIG_DIR} && yolo classify train config={config_filename}"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ Training completed for seed {seed}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for seed {seed}: {e}")
        return False

def main():
    print("🔬 Eksperimen B.1: Classification (RGB Only)")
    print("="*60)
    
    # Check dataset
    if not check_classification_dataset():
        print("\n💡 Solusi:")
        print("   - Siapkan dataset ripeness di Experiments/datasets/ffb_ripeness/")
        print("   - Atau gunakan script lain untuk prepare data")
        return
    
    # Create config
    create_classification_config()
    
    # Train with both seeds
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
        print(f"  - Seed {seed}: runs/classify/exp_b1_cls_seed_{seed}/")
    
    print("\n💡 Untuk evaluasi:")
    print("  yolo classify val model=runs/classify/exp_b1_cls_seed_42/weights/best.pt data=ffb_ripeness.yaml split=test")

if __name__ == "__main__":
    main()
