"""
B.2 Two-Stage Inference Pipeline: Detect → Crop → Classify

End-to-end two-stage inference for ripeness classification:
1. Stage 1: Detect FFB locations with initial ripeness prediction
2. Stage 2: Crop detected FFBs and classify with specialized classifier

Usage:
    python inference_b2_twostage.py

Requirements:
    - Stage 1 detector trained (run train_b2_stage1_detector.py)
    - Stage 2 classifier trained (run train_b2_stage2_classifier.py)
    - ultralytics>=8.3.0

Output:
    - Experiments/runs/b2_twostage_inference/
      ├── results.json (all predictions)
      ├── vis_*.png (visualizations)
      └── summary.txt (statistics)

Author: Research Team
Date: 2026-01-21
"""

import json
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
CLASSIFIER_PATH = (
    PROJECT_DIR
    / "Experiments"
    / "runs"
    / "classify"
    / "exp_b2_stage2_seed_42"
    / "weights"
    / "best.pt"
)
TEST_DIR = (
    PROJECT_DIR / "Experiments" / "datasets" / "ffb_localization" / "images" / "test"
)
OUTPUT_DIR = PROJECT_DIR / "Experiments" / "runs" / "b2_twostage_inference"

BBOX_MARGIN = 0.10  # 10% margin for cropping
CONF_THRESHOLD = 0.25  # Detection confidence threshold


def run_twostage(detector, classifier, img_path):
    """
    Run two-stage inference on one image.

    Args:
        detector: YOLO detector model
        classifier: YOLO classifier model
        img_path: Path to input image

    Returns:
        list: List of detections with classification results
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    h, w = img.shape[:2]

    # Stage 1: Detect FFBs
    try:
        detect_results = detector.predict(img, conf=CONF_THRESHOLD, verbose=False)
    except Exception as e:
        print(f"Detection error on {img_path.name}: {e}")
        return []

    detections = []

    # Stage 2: Classify each detected FFB
    for box in detect_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detect_conf = box.conf[0].item()
        detect_cls_id = int(box.cls[0].item())
        detect_cls_name = detector.names[detect_cls_id]

        # Add margin & crop
        box_w, box_h = x2 - x1, y2 - y1
        margin_w, margin_h = box_w * BBOX_MARGIN, box_h * BBOX_MARGIN

        x1_crop = max(0, int(x1 - margin_w))
        y1_crop = max(0, int(y1 - margin_h))
        x2_crop = min(w, int(x2 + margin_w))
        y2_crop = min(h, int(y2 + margin_h))

        crop = img[y1_crop:y2_crop, x1_crop:x2_crop]

        if crop.size == 0:
            continue

        # Classify crop (Stage 2)
        try:
            classify_results = classifier.predict(crop, verbose=False)
            probs = classify_results[0].probs
            cls_id = probs.top1
            cls_name = classifier.names[cls_id]
            cls_conf = probs.top1conf.item()
        except Exception as e:
            print(f"Classification error: {e}")
            # Fallback to Stage 1 prediction
            cls_name = detect_cls_name
            cls_conf = detect_conf

        detections.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "stage1_class": detect_cls_name,
                "stage1_conf": float(detect_conf),
                "stage2_class": cls_name,
                "stage2_conf": float(cls_conf),
                "final_class": cls_name,  # Use Stage 2 as final
                "final_conf": float(cls_conf),
            }
        )

    return detections


def visualize_results(img_path, detections, output_path):
    """
    Draw detections with classification on image.

    Args:
        img_path: Path to input image
        detections: List of detection dicts
        output_path: Path to save visualization
    """
    img = cv2.imread(str(img_path))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["final_class"]
        cls_conf = det["final_conf"]
        stage1_cls = det["stage1_class"]

        # Color by final class
        color = (
            (0, 255, 0) if cls_name == "ripe" else (0, 0, 255)
        )  # Green for ripe, Red for unripe

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label with Stage 2 result
        label = f"{cls_name} {cls_conf:.2f}"

        # Show if Stage 1 disagreed
        if stage1_cls != cls_name:
            label += f" (S1:{stage1_cls})"

        # Text background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)

        # Text
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    cv2.imwrite(str(output_path), img)


def main():
    print("=" * 60)
    print("B.2 Two-Stage Inference Pipeline")
    print("=" * 60)

    # Check if models exist
    if not DETECTOR_PATH.exists():
        print(f"\n✗ Detector not found: {DETECTOR_PATH}")
        print("Run: python train_b2_stage1_detector.py")
        return

    if not CLASSIFIER_PATH.exists():
        print(f"\n✗ Classifier not found: {CLASSIFIER_PATH}")
        print("Run: python train_b2_stage2_classifier.py")
        return

    if not TEST_DIR.exists():
        print(f"\n✗ Test directory not found: {TEST_DIR}")
        return

    print(f"\n✓ Detector: {DETECTOR_PATH}")
    print(f"✓ Classifier: {CLASSIFIER_PATH}")
    print(f"✓ Test directory: {TEST_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\nLoading models...")
    try:
        detector = YOLO(str(DETECTOR_PATH))
        classifier = YOLO(str(CLASSIFIER_PATH))
        print(f"✓ Models loaded")
        print(f"  Detector classes: {detector.names}")
        print(f"  Classifier classes: {classifier.names}")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return

    # Run inference on test set
    print("\n" + "=" * 60)
    print("Running Two-Stage Inference")
    print("=" * 60)
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"Crop margin: {BBOX_MARGIN * 100}%\n")

    test_images = list(TEST_DIR.glob("*.png")) + list(TEST_DIR.glob("*.jpg"))

    if not test_images:
        print(f"No images found in {TEST_DIR}")
        return

    all_results = {}
    stats = {"ripe": 0, "unripe": 0, "total_detections": 0, "stage_disagreements": 0}

    for img_path in tqdm(test_images, desc="Inference"):
        detections = run_twostage(detector, classifier, img_path)

        all_results[img_path.name] = detections

        # Update stats
        for det in detections:
            stats["total_detections"] += 1
            stats[det["final_class"]] += 1
            if det["stage1_class"] != det["stage2_class"]:
                stats["stage_disagreements"] += 1

        # Visualize
        if detections:  # Only save if there are detections
            vis_path = OUTPUT_DIR / f"vis_{img_path.name}"
            visualize_results(img_path, detections, vis_path)

    # Save results JSON
    results_json = OUTPUT_DIR / "results.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save summary
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("B.2 Two-Stage Inference Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test images: {len(test_images)}\n")
        f.write(f"Total detections: {stats['total_detections']}\n")
        f.write(f"Ripe: {stats['ripe']}\n")
        f.write(f"Unripe: {stats['unripe']}\n")
        f.write(f"Stage 1-2 disagreements: {stats['stage_disagreements']}\n")
        if stats["total_detections"] > 0:
            f.write(
                f"Disagreement rate: {stats['stage_disagreements'] / stats['total_detections'] * 100:.1f}%\n"
            )

    # Print summary
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"\nResults: {OUTPUT_DIR}")
    print(f"  - JSON: {results_json}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Visualizations: vis_*.png")

    print(f"\nStatistics:")
    print(f"  Test images: {len(test_images)}")
    print(f"  Total detections: {stats['total_detections']}")
    print(
        f"  Ripe: {stats['ripe']} ({stats['ripe'] / max(stats['total_detections'], 1) * 100:.1f}%)"
    )
    print(
        f"  Unripe: {stats['unripe']} ({stats['unripe'] / max(stats['total_detections'], 1) * 100:.1f}%)"
    )
    print(
        f"  Stage disagreements: {stats['stage_disagreements']} ({stats['stage_disagreements'] / max(stats['total_detections'], 1) * 100:.1f}%)"
    )

    print("\nNext step:")
    print("  Compare B.2 (two-stage) with B.1 (end-to-end) performance")


if __name__ == "__main__":
    main()
