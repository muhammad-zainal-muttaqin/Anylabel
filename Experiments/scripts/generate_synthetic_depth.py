"""
Generate Synthetic Depth Maps using Depth-Anything-V2-Large

This script generates depth maps from RGB images using the state-of-the-art
Depth-Anything-V2-Large model from HuggingFace.

Output: Normalized depth (0-255, 3-channel) compatible dengan existing pipeline

Usage:
    python generate_synthetic_depth.py

Requirements:
    - transformers>=4.30.0
    - torch>=2.0.0
    - CUDA GPU recommended (CPU akan 10x lebih lambat)
    - ~1.3GB model download on first run

Estimated Time: ~400 images × 3-5 seconds = 20-30 minutes on GPU

Author: Research Team
Date: 2026-01-21
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

PROJECT_DIR = Path(__file__).resolve().parents[2]
RGB_DIR = PROJECT_DIR / "Experiments" / "datasets" / "ffb_localization" / "images"
OUTPUT_DIR = PROJECT_DIR / "Experiments" / "datasets" / "depth_synthetic_da2"

MODEL_NAME = "depth-anything/Depth-Anything-V2-Large"


def normalize_depth(depth_map):
    """
    Normalize Depth-Anything-V2 output to uint8 0-255, 3-channel.

    DA-V2 outputs relative depth (not metric). We use min-max normalization
    to scale to 0-255 range, compatible with existing depth processing pipeline.

    Args:
        depth_map (np.ndarray): Depth map from DA-V2

    Returns:
        np.ndarray: 3-channel uint8 depth (H, W, 3)
    """
    # Min-max normalization to 0-1
    depth_norm = (depth_map - depth_map.min()) / (
        depth_map.max() - depth_map.min() + 1e-8
    )

    # Scale to 0-255
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # Replicate to 3 channels (R=G=B) for YOLO compatibility
    depth_3ch = cv2.merge([depth_uint8, depth_uint8, depth_uint8])

    return depth_3ch


def generate_depth_for_split(split, depth_pipeline):
    """
    Generate depth maps for one split (train/val/test).

    Args:
        split (str): Split name (train/val/test)
        depth_pipeline: HuggingFace depth estimation pipeline
    """
    split_rgb_dir = RGB_DIR / split
    split_output_dir = OUTPUT_DIR / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # Get all RGB images
    rgb_files = sorted(split_rgb_dir.glob("*.png")) + sorted(
        split_rgb_dir.glob("*.jpg")
    )

    if not rgb_files:
        print(f"Warning: No images found in {split_rgb_dir}")
        return

    print(f"Processing {split}: {len(rgb_files)} images")

    success_count = 0
    error_count = 0

    for rgb_path in tqdm(rgb_files, desc=f"Generate {split}"):
        # Load RGB
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"Failed to load: {rgb_path}")
            error_count += 1
            continue

        # Convert BGR to RGB for transformers
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Generate depth using Depth-Anything-V2
        try:
            result = depth_pipeline(rgb_rgb)
            depth_map = np.array(result["depth"])

            # Normalize to 3-channel uint8
            depth_3ch = normalize_depth(depth_map)

            # Save with same filename
            output_path = split_output_dir / rgb_path.name
            cv2.imwrite(str(output_path), depth_3ch)
            success_count += 1

        except Exception as e:
            print(f"Error processing {rgb_path.name}: {e}")
            error_count += 1
            continue

    print(f"  Success: {success_count}, Errors: {error_count}")


def main():
    print("=" * 60)
    print("Generating Synthetic Depth using Depth-Anything-V2-Large")
    print("=" * 60)

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU. This will be VERY slow (~10x slower than GPU).")
        print("         Estimated time: 3-5 hours for 400 images.")
        print("         Consider using Google Colab or Kaggle for GPU access.")
        response = input("Continue with CPU? (y/n): ")
        if response.lower() != "y":
            print("Aborted. Please run on GPU for better performance.")
            return

    # Load Depth-Anything-V2 model
    print(f"\nLoading model: {MODEL_NAME}")
    print("This will download ~1.3GB model on first run...")

    try:
        depth_pipeline = pipeline(
            task="depth-estimation",
            model=MODEL_NAME,
            device=0 if device == "cuda" else -1,
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection (model needs to be downloaded)")
        print("  2. Ensure transformers>=4.30.0 is installed")
        print("  3. Ensure torch>=2.0.0 is installed")
        return

    # Process each split
    print("\n" + "=" * 60)
    print("Generating Depth Maps")
    print("=" * 60)

    total_time_estimate = (
        "20-30 minutes (GPU)" if device == "cuda" else "3-5 hours (CPU)"
    )
    print(f"Estimated total time: {total_time_estimate}\n")

    for split in ["train", "val", "test"]:
        if not (RGB_DIR / split).exists():
            print(f"Warning: {split} directory not found, skipping")
            continue

        generate_depth_for_split(split, depth_pipeline)

    print("\n" + "=" * 60)
    print("Synthetic Depth Generation Complete!")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split
        if split_dir.exists():
            count = len(list(split_dir.glob("*.png"))) + len(
                list(split_dir.glob("*.jpg"))
            )
            print(f"  {split}: {count} depth maps")

    print("\nNext steps:")
    print("  1. python prepare_synthetic_depth_data.py  # Organize dataset")
    print("  2. python train_a4a_synthetic_depth.py     # Train depth-only")
    print("  3. python train_a4b_rgbd_synthetic.py      # Train RGB+synthetic depth")


if __name__ == "__main__":
    main()
