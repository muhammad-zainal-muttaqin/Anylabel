"""
Batch auto-label images using a YOLO model and save LabelMe JSON files.

Output format is compatible with AnyLabeling (one .json per image).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch auto-label folder with YOLO model")
    parser.add_argument("--images_dir", type=str, required=True, help="Image folder path")
    parser.add_argument("--model", type=str, required=True, help="YOLO model path (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--recursive", action="store_true", help="Scan images recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON labels")
    return parser.parse_args()


def collect_images(images_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def to_labelme_json(image_path: Path, result) -> dict:
    h, w = result.orig_shape
    names = result.names
    shapes = []

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().tolist()
        cls_ids = result.boxes.cls.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        for box, cls_id, conf in zip(xyxy, cls_ids, confs):
            x1, y1, x2, y2 = box
            label = names[int(cls_id)] if isinstance(names, dict) else str(int(cls_id))
            shapes.append(
                {
                    "label": label,
                    "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                    "group_id": None,
                    "description": f"auto:{conf:.4f}",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                }
            )

    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    model_path = Path(args.model).resolve()

    if not images_dir.exists():
        print(f"[ERROR] images_dir not found: {images_dir}")
        return 1
    if not model_path.exists():
        print(f"[ERROR] model file not found: {model_path}")
        return 1

    try:
        from ultralytics import YOLO
    except Exception:
        print("[ERROR] ultralytics is not installed.")
        print("Install with:")
        print("  python -m pip install ultralytics")
        return 1

    image_files = collect_images(images_dir, args.recursive)
    if not image_files:
        print(f"[ERROR] no images found in: {images_dir}")
        return 1

    print("=" * 70)
    print("Batch Auto-Label")
    print("=" * 70)
    print(f"Images dir : {images_dir}")
    print(f"Model      : {model_path}")
    print(f"Images     : {len(image_files)}")
    print(f"Conf / IoU : {args.conf} / {args.iou}")
    print(f"Recursive  : {args.recursive}")
    print(f"Overwrite  : {args.overwrite}")
    print()

    model = YOLO(str(model_path))

    processed = 0
    skipped = 0
    failed = 0
    total_boxes = 0

    for idx, image_path in enumerate(image_files, start=1):
        json_path = image_path.with_suffix(".json")
        if json_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            result = model.predict(
                source=str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            data = to_labelme_json(image_path, result)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            processed += 1
            total_boxes += len(data["shapes"])

            if idx % 20 == 0:
                print(f"[INFO] processed {idx}/{len(image_files)} images...")

        except Exception as exc:
            failed += 1
            print(f"[ERROR] {image_path.name}: {exc}")

    print()
    print("=" * 70)
    print("Done")
    print("=" * 70)
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Total box : {total_boxes}")
    if processed > 0:
        print(f"Avg box   : {total_boxes / processed:.2f} / image")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
