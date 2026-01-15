"""
Eksperimen A.1: RGB Only (Baseline)
Train YOLOv8n pada dataset RGB dengan 2 random seeds
"""

import subprocess
import yaml
import os

# Configurations
CONFIG_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments"
BASE_CONFIG = {
    'task': 'detect',
    'mode': 'train',
    'model': 'yolov8n.pt',
    'data': 'ffb_localization.yaml',
    'epochs': 50,
    'patience': 10,
    'batch': 16,
    'imgsz': 640,
    'save': True,
    'device': 0,  # Change to 'cpu' if no GPU
    'workers': 4,
    'project': 'runs/detect',
    'name': 'exp_a1_rgb_baseline'
}

SEEDS = [42, 123]

def train_with_seed(seed):
    """Train with specific seed"""
    
    # Update config with seed
    config = BASE_CONFIG.copy()
    config['seed'] = seed
    config['name'] = f'exp_a1_rgb_seed_{seed}'
    
    # Save config file
    config_filename = f'config_a1_rgb_seed_{seed}.yaml'
    config_path = os.path.join(CONFIG_DIR, config_filename)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING A.1 RGB - Seed {seed}")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Output: runs/detect/{config['name']}")
    print(f"{'='*60}\n")
    
    # Run training
    command = f"cd {CONFIG_DIR} && yolo detect train config={config_filename}"
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ Training completed for seed {seed}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for seed {seed}: {e}")
        return False
    
    return True

def main():
    print("🔬 Eksperimen A.1: RGB Only (Baseline)")
    print("="*60)
    
    # Check if dataset exists
    dataset_path = os.path.join(CONFIG_DIR, "datasets", "ffb_localization")
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset tidak ditemukan: {dataset_path}")
        print("   Jalankan split_localization_data.py terlebih dahulu!")
        return
    
    # Check if config file exists
    config_file = os.path.join(CONFIG_DIR, "ffb_localization.yaml")
    if not os.path.exists(config_file):
        print(f"❌ Config dataset tidak ditemukan: {config_file}")
        print("   Pastikan ffb_localization.yaml sudah dibuat!")
        return
    
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
    
    # Show results location
    print("\n📁 Hasil training:")
    for seed in SEEDS:
        print(f"  - Seed {seed}: runs/detect/exp_a1_rgb_seed_{seed}/")
    
    print("\n💡 Untuk evaluasi, jalankan:")
    print("  cd Experiments")
    print("  yolo detect val model=runs/detect/exp_a1_rgb_seed_42/weights/best.pt data=ffb_localization.yaml split=test")

if __name__ == "__main__":
    main()
