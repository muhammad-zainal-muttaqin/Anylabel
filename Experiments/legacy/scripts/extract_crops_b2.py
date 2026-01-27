"""
Extract FFB Crops dari Stage 1 Detector untuk B.2 Stage 2

Extract bounding boxes from Stage 1 detector predictions and save as crops
organized by predicted class for Stage 2 classifier training.

Bounding boxes are expanded by 10% margin for better context.

Usage:
    python extract_crops_b2.py

Requirements:
    - Stage 1 detector trained (run train_b2_stage1_detector.py first)
    - ultralytics>=8.3.0

Output:
    - Experiments/datasets/ffb_ripeness_twostage_crops/
      ├── train/
      │   ├── ripe/
      │   └── unripe/
      ├── val/
      │   ├── ripe/
      │   └── unripe/
      └── test/
          ├── ripe/
          └── unripe/

Author: Research Team
Date: 2026-01-21
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[2]
DETECTOR_PATH = (
    PROJECT_DIR
    / "Experiments"
    / "runs"
    / "detect"
    / "exp_b2_stage1_seed_42"
    / "weights"
    / "best.pt"
)
RGB_DIR = PROJECT_DIR / "Experiments" / "datasets" / "ffb_localization" / "images"
OUTPUT_DIR = PROJECT_DIR / "Experiments" / "datasets" / "ffb_ripeness_twostage_crops"

BBOX_MARGIN = 0.10  # 10% margin around bbox
CONF_THRESHOLD = 0.25  # Confidence threshold for detection


def extract_crops_for_split(model, split):
    """
    Extract crops from one split using Stage 1 detector.

    Args:
        model: YOLO model
        split (str): Split name (train/val/test)

    Returns:
        dict: Crop counts per class
    """
    split_dir = RGB_DIR / split
    output_split = OUTPUT_DIR / split

    if not split_dir.exists():
        print(f"Warning: {split} directory not found: {split_dir}")
        return {"ripe": 0, "unripe": 0}

    # Create output directories
    for cls in ["ripe", "unripe"]:
        (output_split / cls).mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpg"))

    if not image_files:
        print(f"Warning: No images found in {split_dir}")
        return {"ripe": 0, "unripe": 0}

    crop_count = {"ripe": 0, "unripe": 0}

    for img_path in tqdm(image_files, desc=f"Extract {split}"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load: {img_path}")
            continue

        # Run detection
        try:
            results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print(f"Prediction error on {img_path.name}: {e}")
            continue

        h, w = img.shape[:2]

        # Extract each detected FFB
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()

            # Get class name
            cls_name = model.names[cls_id]  # 'ripe' or 'unripe'

            # Add margin
            box_w, box_h = x2 - x1, y2 - y1
            margin_w, margin_h = box_w * BBOX_MARGIN, box_h * BBOX_MARGIN

            x1 = max(0, int(x1 - margin_w))
            y1 = max(0, int(y1 - margin_h))
            x2 = min(w, int(x2 + margin_w))
            y2 = min(h, int(y2 + margin_h))

            # Extract crop
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Save crop
            crop_name = f"{img_path.stem}_crop{i:02d}_{cls_name}_conf{conf:.2f}.jpg"
            crop_path = output_split / cls_name / crop_name

            cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            crop_count[cls_name] += 1

    return crop_count


def main():
    print("=" * 60)
    print("Extracting FFB Crops for B.2 Stage 2")
    print("=" * 60)

    # Check if detector exists
    if not DETECTOR_PATH.exists():
        print(f"\n✗ Detector not found: {DETECTOR_PATH}")
        print("\nPlease run first:")
        print("  python train_b2_stage1_detector.py")
        return

    print(f"\n✓ Detector found: {DETECTOR_PATH}")

    # Load detector
    print("\nLoading Stage 1 detector...")
    try:
        model = YOLO(str(DETECTOR_PATH))
        print(f"✓ Model loaded: {model.names}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Extract crops for each split
    print("\n" + "=" * 60)
    print("Extracting Crops")
    print("=" * 60)
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"Bounding box margin: {BBOX_MARGIN * 100}%\n")

    total_crops = {"ripe": 0, "unripe": 0}

    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split}...")
        counts = extract_crops_for_split(model, split)

        print(f"  Ripe: {counts['ripe']}")
        print(f"  Unripe: {counts['unripe']}")
        print(f"  Total: {counts['ripe'] + counts['unripe']}")

        total_crops["ripe"] += counts["ripe"]
        total_crops["unripe"] += counts["unripe"]

    # Summary
    print("\n" + "=" * 60)
    print("Crop Extraction Complete!")
    print("=" * 60)
    print(f"\nTotal crops extracted:")
    print(f"  Ripe: {total_crops['ripe']}")
    print(f"  Unripe: {total_crops['unripe']}")
    print(f"  Total: {total_crops['ripe'] + total_crops['unripe']}")

    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Check class balance
    if total_crops["ripe"] + total_crops["unripe"] > 0:
        ripe_pct = (
            total_crops["ripe"] / (total_crops["ripe"] + total_crops["unripe"]) * 100
        )
        unripe_pct = (
            total_crops["unripe"] / (total_crops["ripe"] + total_crops["unripe"]) * 100
        )
        print(f"\nClass distribution:")
        print(f"  Ripe: {ripe_pct:.1f}%")
        print(f"  Unripe: {unripe_pct:.1f}%")

        if ripe_pct < 15 or ripe_pct > 85:
            print("\n⚠️  Warning: Significant class imbalance detected!")
            print("   Consider using class weights in Stage 2 training")

    print("\nNext step:")
    print("  python train_b2_stage2_classifier.py  # Train classifier on crops")


if __name__ == "__main__":
    main()
