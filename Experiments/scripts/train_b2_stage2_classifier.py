"""
Eksperimen B.2 Stage 2: Train Ripeness Classifier on Cropped FFBs

Train YOLOv11n-cls classifier on cropped FFB images from Stage 1 detector.
This is Stage 2 of the two-stage pipeline.

Stage 1: Detect FFB locations → Extract crops
Stage 2: Classify crops as ripe/unripe

Usage:
    python train_b2_stage2_classifier.py

Requirements:
    - Crops extracted (run extract_crops_b2.py first)
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
    "task": "classify",
    "mode": "train",
    "model": "yolo11n-cls.pt",
    "data": "datasets/ffb_ripeness_twostage_crops",  # Directory-based classification
    "epochs": 50,
    "patience": 10,
    "batch": 32,  # Higher batch for classification
    "imgsz": 224,  # Standard classification size
    "save": True,
    "device": 0,
    "workers": 4,
    "project": "runs/classify",
    "name": "exp_b2_stage2_classifier",
}

SEEDS = [42, 123]


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
    config["name"] = f"exp_b2_stage2_seed_{seed}"

    config_filename = f"config_b2_stage2_seed_{seed}.yaml"
    config_path = CONFIG_DIR / config_filename

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'=' * 60}")
    print(f"TRAINING B.2 STAGE 2 CLASSIFIER - Seed {seed}")
    print(f"{'=' * 60}")
    print(f"Config: {config_path}")
    print(f"Output: runs/classify/{config['name']}")
    print(f"{'=' * 60}\n")

    command = f"cd {CONFIG_DIR} && yolo classify train config={config_filename}"

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"\n✓ Training completed for seed {seed}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed for seed {seed}: {e}")
        return False


def main():
    print("=" * 60)
    print("Eksperimen B.2 Stage 2: Ripeness Classifier Training")
    print("=" * 60)

    # Check if crops exist
    crops_dir = CONFIG_DIR / "datasets" / "ffb_ripeness_twostage_crops"

    if not crops_dir.exists():
        print(f"\n✗ Crops directory not found: {crops_dir}")
        print("\nPlease run first:")
        print("  python extract_crops_b2.py")
        return

    # Verify directory structure
    required_dirs = [
        crops_dir / "train" / "ripe",
        crops_dir / "train" / "unripe",
        crops_dir / "val" / "ripe",
        crops_dir / "val" / "unripe",
        crops_dir / "test" / "ripe",
        crops_dir / "test" / "unripe",
    ]

    missing_dirs = [d for d in required_dirs if not d.exists()]

    if missing_dirs:
        print(f"\n✗ Missing directories:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("\nPlease run:")
        print("  python extract_crops_b2.py")
        return

    print(f"\n✓ Crops directory found: {crops_dir}")

    # Check crop counts
    print("\nCrop counts:")
    for split in ["train", "val", "test"]:
        ripe_count = len(list((crops_dir / split / "ripe").glob("*.jpg")))
        unripe_count = len(list((crops_dir / split / "unripe").glob("*.jpg")))
        total = ripe_count + unripe_count
        print(f"  {split}: {total} crops ({ripe_count} ripe, {unripe_count} unripe)")

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
        print(f"  - Seed {seed}: runs/classify/exp_b2_stage2_seed_{seed}/")

    print("\nEvaluation:")
    print("  cd Experiments")
    print(
        "  yolo classify val model=runs/classify/exp_b2_stage2_seed_42/weights/best.pt \\"
    )
    print("                    data=datasets/ffb_ripeness_twostage_crops split=test")

    print("\nNext step:")
    print("  python inference_b2_twostage.py  # Run two-stage pipeline end-to-end")


if __name__ == "__main__":
    main()
