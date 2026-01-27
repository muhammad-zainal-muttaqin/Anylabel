"""
Eksperimen A.3: RGB+Depth (4-channel) - FIXED
Train YOLOv11n dengan custom RGBD dataloader dan synced augmentation

Option A: HSV augmentation ON (default YOLO)
Option B: HSV augmentation OFF (untuk fair comparison)

Usage:
    python train_a3_rgbd.py

Requirements:
    - torch>=2.0.0
    - ultralytics>=8.3.0
    - Dataset: ffb_localization (RGB) + depth_processed_rgb (Depth)
    - Modified model will be saved to yolo11n_4ch.pt

Author: Research Team
Date: 2026-01-21
"""

import os
import subprocess
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

CONFIG_DIR = Path(r"D:\Work\Assisten Dosen\Anylabel\Experiments")

BASE_CONFIG = {
    "task": "detect",
    "mode": "train",
    "model": "yolo11n_4ch.pt",  # Modified 4-channel model
    "data": "configs/ffb_localization_rgbd.yaml",
    "epochs": 50,
    "patience": 10,
    "batch": 16,
    "imgsz": 640,
    "save": True,
    "device": 0,
    "workers": 4,
    "project": "runs/detect",
}

SEEDS = [42, 123]


def modify_model_for_4ch(base_model="yolo11n.pt"):
    """
    Modify YOLOv11n first conv layer from 3-channel to 4-channel input.

    Args:
        base_model (str): Path to base YOLO model

    Returns:
        Path: Path to modified 4-channel model
    """
    print("Modifying YOLOv11n for 4-channel input...")

    try:
        model = YOLO(base_model)
    except Exception as e:
        print(f"Error loading base model: {e}")
        print(f"Make sure {base_model} is available.")
        return None

    # Get first conv layer
    first_conv = model.model.model[0].conv

    print(f"Original conv layer: in_channels={first_conv.in_channels}")

    # Create new 4-channel conv
    new_conv = torch.nn.Conv2d(
        in_channels=4,  # RGB + Depth
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )

    # Initialize weights: copy RGB channels, initialize depth from R channel
    with torch.no_grad():
        # Copy RGB weights (channels 0, 1, 2)
        new_conv.weight[:, :3, :, :] = first_conv.weight
        # Initialize depth channel from R channel (or average of RGB)
        new_conv.weight[:, 3:, :, :] = first_conv.weight[:, :1, :, :]

        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    # Replace in model
    model.model.model[0].conv = new_conv

    print(f"Modified conv layer: in_channels={new_conv.in_channels}")

    # Save modified model
    modified_path = CONFIG_DIR / "yolo11n_4ch.pt"

    try:
        torch.save(model.ckpt, modified_path)
        print(f"Modified model saved: {modified_path}")
        return modified_path
    except Exception as e:
        print(f"Error saving modified model: {e}")
        return None


def train_with_seed(seed, use_hsv_aug=True):
    """
    Train RGBD model with specific seed and augmentation option.

    Args:
        seed (int): Random seed
        use_hsv_aug (bool): Whether to use HSV augmentation

    Returns:
        bool: True if training successful
    """
    config = BASE_CONFIG.copy()
    config["seed"] = seed

    aug_suffix = "optionA" if use_hsv_aug else "optionB"
    config["name"] = f"exp_a3_rgbd_{aug_suffix}_seed_{seed}"

    # Set HSV augmentation parameters
    if use_hsv_aug:
        config["hsv_h"] = 0.015
        config["hsv_s"] = 0.7
        config["hsv_v"] = 0.4
    else:
        config["hsv_h"] = 0.0
        config["hsv_s"] = 0.0
        config["hsv_v"] = 0.0

    config_filename = f"config_a3_rgbd_{aug_suffix}_seed_{seed}.yaml"
    config_path = CONFIG_DIR / config_filename

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'=' * 60}")
    print(f"TRAINING A.3 RGBD - {aug_suffix.upper()} - Seed {seed}")
    print(f"{'=' * 60}")
    print(f"Config: {config_path}")
    print(f"Output: runs/detect/{config['name']}")
    print(f"HSV Aug: {'Enabled' if use_hsv_aug else 'Disabled'}")
    print(f"{'=' * 60}\n")

    # Note: For local training with custom dataset, integrate custom_rgbd_dataset.py
    # For Kaggle training, use notebook with custom dataset implementation

    print("NOTE: For actual training:")
    print("  Option 1 (Local): Integrate custom_rgbd_dataset.py with YOLO trainer")
    print("  Option 2 (Kaggle): Upload dataset & use Kaggle notebook")
    print(f"  Config saved for reference: {config_path}")

    return True


def main():
    print("Eksperimen A.3: RGB+Depth (4-channel) - FIX")
    print("=" * 60)

    # Check if dataset exists
    dataset_rgb = CONFIG_DIR / "datasets" / "ffb_localization"
    dataset_depth = CONFIG_DIR / "datasets" / "depth_processed_rgb"

    if not dataset_rgb.exists():
        print(f"✗ RGB dataset not found: {dataset_rgb}")
        print("  Run: python split_localization_data.py")
        return

    if not dataset_depth.exists():
        print(f"✗ Depth dataset not found: {dataset_depth}")
        print("  Run: python prepare_depth_data.py")
        return

    print(f"✓ RGB dataset: {dataset_rgb}")
    print(f"✓ Depth dataset: {dataset_depth}")

    # Step 1: Modify model architecture
    print("\nStep 1: Modifying YOLOv11n for 4-channel input...")
    modified_model = modify_model_for_4ch()

    if modified_model is None:
        print("✗ Failed to modify model")
        return

    print(f"✓ Modified model saved: {modified_model}")

    # Step 2: Train both options
    print("\nStep 2: Training with both augmentation options...")

    for option_name, use_hsv in [
        ("Option A (with HSV aug)", True),
        ("Option B (no HSV aug)", False),
    ]:
        print(f"\n{'=' * 60}")
        print(f"{option_name}")
        print(f"{'=' * 60}")

        for seed in SEEDS:
            train_with_seed(seed, use_hsv_aug=use_hsv)

    print("\n" + "=" * 60)
    print("A.3 RGBD Training Setup Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. For local training: Integrate custom_rgbd_dataset.py")
    print("  2. For Kaggle training:")
    print("     - Upload dataset: python build_uploadkaggle_rgbd_pairs.py")
    print("     - Create Kaggle notebook with custom dataset")
    print("     - Use configs generated in Experiments/ directory")
    print("\nFiles created:")
    print("  - yolo11n_4ch.pt (modified 4-channel model)")
    print("  - config_a3_rgbd_option*_seed_*.yaml (training configs)")


if __name__ == "__main__":
    main()
