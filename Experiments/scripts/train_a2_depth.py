"""
Eksperimen A.2: Real Depth Only - V2
Train YOLOv11n pada dataset Depth (3-channel replicated) dengan uniform geometric augmentation

Uniform Augmentation (geometric only):
    - translate=0.1
    - scale=0.5
    - fliplr=0.5
    - HSV disabled (hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    - erasing=0.0, mosaic=0.0, mixup=0.0

Special: Reset BatchNorm stats untuk adaptasi depth data

Seeds: 42, 123, 456, 789, 101 (5 seeds)
Output: kaggleoutput/a2_depth_results.txt

Author: Research Team
Date: 2026-01-28
"""

import os
import sys
import shutil
import glob
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

# Import reset BN utility
sys.path.insert(0, str(Path(__file__).parent))
from reset_bn import reset_bn_stats_simple


def get_config():
    """Get configuration based on environment (Kaggle vs Local)."""
    if os.path.exists('/kaggle/working'):
        # Kaggle environment
        return {
            'base_dir': Path('/kaggle/working'),
            'data_yaml': '/kaggle/working/dataset/ffb_localization_depth.yaml',
            'dataset_path': '/kaggle/working/dataset/ffb_localization_depth',
            'depth_source': '/kaggle/working/dataset/depth_processed_rgb',
            'output_dir': Path('/kaggle/working/kaggleoutput'),
            'is_kaggle': True
        }
    else:
        # Local environment
        base = Path(r'D:\Work\Assisten Dosen\Anylabel\Experiments')
        return {
            'base_dir': base,
            'data_yaml': str(base / 'configs' / 'ffb_localization_depth.yaml'),
            'dataset_path': str(base / 'datasets' / 'ffb_localization_depth'),
            'depth_source': str(base / 'datasets' / 'depth_processed_rgb'),
            'rgb_dataset': str(base / 'datasets' / 'ffb_localization'),
            'output_dir': base / 'kaggleoutput',
            'is_kaggle': False
        }


# Uniform geometric augmentation parameters (for ALL experiments)
AUGMENT_PARAMS = {
    'translate': 0.1,    # geometric - ACTIVE
    'scale': 0.5,        # geometric - ACTIVE
    'fliplr': 0.5,       # geometric - ACTIVE
    'hsv_h': 0.0,        # disabled
    'hsv_s': 0.0,        # disabled
    'hsv_v': 0.0,        # disabled
    'erasing': 0.0,      # disabled
    'mosaic': 0.0,       # disabled
    'mixup': 0.0,        # disabled
}

# Standard training configuration
BASE_CONFIG = {
    'model': 'yolo11n.pt',
    'epochs': 100,
    'patience': 30,
    'batch': 16,
    'imgsz': 640,
    'save': True,
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'workers': 4,
    'project': 'runs/detect',
    'name': 'exp_a2_depth',
}

# 5 seeds for reproducibility
SEEDS = [42, 123, 456, 789, 101]


def prepare_depth_dataset(config):
    """Prepare depth dataset by copying depth images to target structure."""
    print("\nPreparing depth dataset...")

    dataset_path = Path(config['dataset_path'])
    depth_source = Path(config['depth_source'])

    # Create directories
    for split in ['train', 'val', 'test']:
        (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Get depth images
    depth_images = list(depth_source.glob('*.png'))
    if not depth_images:
        print(f"ERROR: No depth images found in {depth_source}")
        return False

    print(f"Found {len(depth_images)} depth images")

    # Get splits from RGB dataset
    if config.get('is_kaggle'):
        # On Kaggle, assume dataset is already structured
        return True

    rgb_dataset = Path(config['rgb_dataset'])

    for split in ['train', 'val', 'test']:
        rgb_images = list((rgb_dataset / 'images' / split).glob('*.png'))
        print(f"  Processing {split}: {len(rgb_images)} files")

        for rgb_file in rgb_images:
            filename = rgb_file.name
            depth_file = depth_source / filename

            if depth_file.exists():
                # Copy depth image
                shutil.copy2(depth_file, dataset_path / 'images' / split / filename)

                # Copy label
                label_src = rgb_dataset / 'labels' / split / filename.replace('.png', '.txt')
                label_dest = dataset_path / 'labels' / split / filename.replace('.png', '.txt')

                if label_src.exists():
                    shutil.copy2(label_src, label_dest)
            else:
                print(f"    Warning: Depth file not found for {filename}")

    print("Depth dataset preparation complete!")
    return True


def create_depth_config(config):
    """Create YAML config for depth dataset."""
    yaml_content = f"""# Dataset Configuration for FFB Localization - DEPTH ONLY
path: {config['dataset_path'].replace('\\', '/')}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['fresh_fruit_bunch']
"""

    config_path = Path(config['data_yaml'])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml_content)
    print(f"Config created: {config_path}")
    return config_path


def verify_dataset(config):
    """Verify dataset exists and is properly structured."""
    dataset_path = Path(config['dataset_path'])

    required_dirs = [
        dataset_path / 'images' / 'train',
        dataset_path / 'images' / 'val',
        dataset_path / 'labels' / 'train',
        dataset_path / 'labels' / 'val',
    ]

    missing = [d for d in required_dirs if not d.exists()]
    if missing:
        print(f"ERROR: Missing directories:")
        for d in missing:
            print(f"  - {d}")
        return False

    # Count files
    train_images = list((dataset_path / 'images' / 'train').glob('*.png'))
    train_labels = list((dataset_path / 'labels' / 'train').glob('*.txt'))

    print(f"Dataset verified:")
    print(f"  - Train images: {len(train_images)}")
    print(f"  - Train labels: {len(train_labels)}")

    return len(train_images) > 0 and len(train_labels) > 0


def train_with_seed(seed, config, output_file):
    """
    Train YOLO model with specific seed and reset BN stats.

    Args:
        seed (int): Random seed for reproducibility
        config (dict): Environment configuration
        output_file: File handle for logging results

    Returns:
        dict: Training results
    """
    print(f"\n{'='*70}")
    print(f"TRAINING A.2 REAL DEPTH - Seed {seed}")
    print(f"{'='*70}")

    # Merge base config with augmentation params
    train_config = {**BASE_CONFIG, **AUGMENT_PARAMS}
    train_config['seed'] = seed
    train_config['name'] = f'exp_a2_depth_seed_{seed}'
    train_config['data'] = config['data_yaml']

    print(f"Configuration:")
    print(f"  Model: {train_config['model']}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Batch: {train_config['batch']}")
    print(f"  Image size: {train_config['imgsz']}")
    print(f"  Device: {train_config['device']}")
    print(f"  Augmentation (uniform geometric):")
    print(f"    - translate: {AUGMENT_PARAMS['translate']}")
    print(f"    - scale: {AUGMENT_PARAMS['scale']}")
    print(f"    - fliplr: {AUGMENT_PARAMS['fliplr']}")
    print(f"    - HSV: disabled")
    print(f"  Special: Reset BatchNorm stats enabled")

    start_time = datetime.now()

    try:
        # Load model
        model = YOLO(train_config['model'])

        # Reset BatchNorm stats for depth adaptation
        print("\nResetting BatchNorm statistics...")
        model.model = reset_bn_stats_simple(model.model)
        print("BatchNorm stats reset complete!")

        # Train
        results = model.train(
            data=train_config['data'],
            epochs=train_config['epochs'],
            patience=train_config['patience'],
            batch=train_config['batch'],
            imgsz=train_config['imgsz'],
            save=train_config['save'],
            device=train_config['device'],
            workers=train_config['workers'],
            project=train_config['project'],
            name=train_config['name'],
            seed=train_config['seed'],
            translate=train_config['translate'],
            scale=train_config['scale'],
            fliplr=train_config['fliplr'],
            hsv_h=train_config['hsv_h'],
            hsv_s=train_config['hsv_s'],
            hsv_v=train_config['hsv_v'],
            erasing=train_config['erasing'],
            mosaic=train_config['mosaic'],
            mixup=train_config['mixup'],
        )

        end_time = datetime.now()
        duration = end_time - start_time

        # Get best metrics
        best_map50 = results.results_dict.get('metrics/mAP50(B)', 0)
        best_map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)

        result_line = (f"Seed {seed:3d}: mAP50={best_map50:.4f}, "
                      f"mAP50-95={best_map50_95:.4f}, duration={duration}")

        print(f"\nResults: {result_line}")
        output_file.write(result_line + "\n")
        output_file.flush()

        return {
            'seed': seed,
            'success': True,
            'map50': best_map50,
            'map50_95': best_map50_95,
            'duration': duration,
            'error': None
        }

    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        error_msg = str(e)

        result_line = f"Seed {seed:3d}: FAILED - {error_msg}, duration={duration}"
        print(f"\nERROR: {result_line}")
        output_file.write(result_line + "\n")
        output_file.flush()

        return {
            'seed': seed,
            'success': False,
            'map50': 0,
            'map50_95': 0,
            'duration': duration,
            'error': error_msg
        }


def main():
    """Main execution function."""
    print("="*70)
    print("Eksperimen A.2: Real Depth Only - V2")
    print("Uniform Geometric Augmentation + Reset BN")
    print("="*70)

    # Get environment configuration
    config = get_config()
    print(f"\nEnvironment: {'Kaggle' if config['is_kaggle'] else 'Local'}")
    print(f"Base directory: {config['base_dir']}")
    print(f"Output directory: {config['output_dir']}")

    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # Prepare dataset (only for local)
    if not config['is_kaggle']:
        if not prepare_depth_dataset(config):
            print("ERROR: Dataset preparation failed!")
            sys.exit(1)
        create_depth_config(config)

    # Verify dataset
    print("\nVerifying dataset...")
    if not verify_dataset(config):
        print("ERROR: Dataset verification failed!")
        sys.exit(1)

    # Open results file
    results_path = config['output_dir'] / 'a2_depth_results.txt'
    print(f"\nResults will be saved to: {results_path}")

    with open(results_path, 'w') as output_file:
        # Write header
        output_file.write("="*70 + "\n")
        output_file.write("Eksperimen A.2: Real Depth Only - V2\n")
        output_file.write("Uniform Geometric Augmentation + Reset BN\n")
        output_file.write("="*70 + "\n")
        output_file.write(f"Start time: {datetime.now()}\n")
        output_file.write(f"Model: {BASE_CONFIG['model']}\n")
        output_file.write(f"Epochs: {BASE_CONFIG['epochs']}\n")
        output_file.write(f"Batch size: {BASE_CONFIG['batch']}\n")
        output_file.write(f"Image size: {BASE_CONFIG['imgsz']}\n")
        output_file.write(f"Seeds: {SEEDS}\n")
        output_file.write(f"Augmentation params: {AUGMENT_PARAMS}\n")
        output_file.write(f"Reset BN: Enabled\n")
        output_file.write("="*70 + "\n\n")
        output_file.flush()

        # Train with all seeds
        all_results = []
        for seed in SEEDS:
            result = train_with_seed(seed, config, output_file)
            all_results.append(result)

        # Summary
        output_file.write("\n" + "="*70 + "\n")
        output_file.write("SUMMARY\n")
        output_file.write("="*70 + "\n")

        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]

        if successful:
            avg_map50 = sum(r['map50'] for r in successful) / len(successful)
            avg_map50_95 = sum(r['map50_95'] for r in successful) / len(successful)

            output_file.write(f"Successful runs: {len(successful)}/{len(SEEDS)}\n")
            output_file.write(f"Average mAP50: {avg_map50:.4f}\n")
            output_file.write(f"Average mAP50-95: {avg_map50_95:.4f}\n")
            output_file.write(f"Std mAP50: {(sum((r['map50'] - avg_map50)**2 for r in successful)/len(successful))**0.5:.4f}\n")
            output_file.write(f"Std mAP50-95: {(sum((r['map50_95'] - avg_map50_95)**2 for r in successful)/len(successful))**0.5:.4f}\n")

        if failed:
            output_file.write(f"\nFailed runs: {len(failed)}\n")
            for r in failed:
                output_file.write(f"  Seed {r['seed']}: {r['error']}\n")

        output_file.write(f"\nEnd time: {datetime.now()}\n")
        output_file.write("="*70 + "\n")

    # Print summary to console
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {results_path}")

    if successful:
        print(f"\nSuccessful runs: {len(successful)}/{len(SEEDS)}")
        print(f"Average mAP50: {avg_map50:.4f}")
        print(f"Average mAP50-95: {avg_map50_95:.4f}")

    if failed:
        print(f"\nFailed runs: {len(failed)}")
        for r in failed:
            print(f"  Seed {r['seed']}: {r['error']}")

    print("\nModel locations:")
    for seed in SEEDS:
        print(f"  Seed {seed}: runs/detect/exp_a2_depth_seed_{seed}/weights/best.pt")


if __name__ == "__main__":
    main()
