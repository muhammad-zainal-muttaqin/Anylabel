"""
Eksperimen B.2 Stage 1: Train Ripeness Detector (2 Classes)

Train YOLOv11n detector to detect ripe and unripe FFBs.
This is Stage 1 of the two-stage pipeline.

Stage 1: Detect FFB locations with ripeness class (ripe/unripe)
Stage 2: Crop detections → Classify with specialized classifier

Usage:
    python train_b2_stage1_detector.py

Requirements:
    - ffb_ripeness_detect dataset (2-class detection)
    - ultralytics>=8.3.0

Author: Research Team
Date: 2026-01-21
"""

import os
import subprocess
from pathlib import Path

import yaml

CONFIG_DIR = Path(r"D:\Work\Assisten Dosen\Anylabel\Experiments")

BASE_CONFIG = {
    "task": "detect",
    "mode": "train",
    "model": "yolo11n.pt",
    "data": "configs/ffb_ripeness_detect.yaml",  # 2-class detection
    "epochs": 50,
    "patience": 10,
    "batch": 16,
    "imgsz": 640,
    "save": True,
    "device": 0,
    "workers": 4,
    "project": "runs/detect",
    "name": "exp_b2_stage1_detector",
}

SEEDS = [42, 123]


def create_config_yaml():
    """Create YAML config for ripeness detection dataset."""
    yaml_content = f"""# Dataset Configuration for FFB Ripeness Detection (2 Classes)
# Used for B.2 Stage 1 detector

path: D:/Work/Assisten Dosen/Anylabel/Experiments/datasets/ffb_ripeness_detect
train: images/train
val: images/val
test: images/test

nc: 2
names: ['ripe', 'unripe']

# Note: This dataset has 2 classes for ripeness detection
# Different from B.1 which uses full-image classification
"""

    yaml_path = CONFIG_DIR / "configs" / "ffb_ripeness_detect.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Config created: {yaml_path}")
    return yaml_path


def train_with_seed(seed):
    """
    Train with specific seed.

    Args:
        seed (int): Random seed for reproducibility

    Returns:
        bool: True if training successful
    """
    config = BASE_CONFIG.copy()
    config["seed"] = seed
    config["name"] = f"exp_b2_stage1_seed_{seed}"

    config_filename = f"config_b2_stage1_seed_{seed}.yaml"
    config_path = CONFIG_DIR / config_filename

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'=' * 60}")
    print(f"TRAINING B.2 STAGE 1 DETECTOR - Seed {seed}")
    print(f"{'=' * 60}")
    print(f"Config: {config_path}")
    print(f"Output: runs/detect/{config['name']}")
    print(f"{'=' * 60}\n")

    command = f"cd {CONFIG_DIR} && yolo detect train config={config_filename}"

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"\n✓ Training completed for seed {seed}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed for seed {seed}: {e}")
        return False


def main():
    print("=" * 60)
    print("Eksperimen B.2 Stage 1: Ripeness Detector Training")
    print("=" * 60)

    # Check if dataset exists
    dataset_path = CONFIG_DIR / "datasets" / "ffb_ripeness_detect"

    if not dataset_path.exists():
        print(f"\n✗ Ripeness detection dataset not found: {dataset_path}")
        print("\nPlease check:")
        print("  - Dataset should be at: Experiments/datasets/ffb_ripeness_detect/")
        print("  - Or in: Experiments/UploadKaggle/ffb_ripeness_detect/")
        print("\nIf dataset exists in UploadKaggle, copy to datasets:")
        print(
            "  robocopy UploadKaggle\\ffb_ripeness_detect datasets\\ffb_ripeness_detect /E"
        )
        return

    print(f"\n✓ Dataset found: {dataset_path}")

    # Create config
    print("\n" + "-" * 60)
    config_yaml = create_config_yaml()

    # Train with both seeds
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    results = []
    for seed in SEEDS:
        success = train_with_seed(seed)
        results.append((seed, success))

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for seed, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"Seed {seed}: {status}")

    print("\nResults location:")
    for seed in SEEDS:
        print(f"  - Seed {seed}: runs/detect/exp_b2_stage1_seed_{seed}/")

    print("\nNext steps:")
    print("  1. python extract_crops_b2.py  # Extract crops from detections")
    print("  2. python train_b2_stage2_classifier.py  # Train classifier on crops")
    print("  3. python inference_b2_twostage.py  # Run two-stage pipeline")


if __name__ == "__main__":
    main()
