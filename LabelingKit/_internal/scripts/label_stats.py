"""
Show labeling statistics from JSON label files.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_json_label(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    labels = [s.get("label", "unknown") for s in shapes]
    shape_types = [s.get("shape_type", "unknown") for s in shapes]

    return {
        "num_objects": len(shapes),
        "labels": labels,
        "shape_types": shape_types,
        "width": data.get("imageWidth", 0),
        "height": data.get("imageHeight", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Show labeling statistics")
    parser.add_argument("--labels", "-l", type=str, default="Dataset", help="Directory containing JSON label files")
    parser.add_argument("--recursive", action="store_true", help="Scan labels recursively")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    labels_dir = (root_dir / args.labels).resolve()

    print("=" * 60)
    print("  Labeling Statistics")
    print("=" * 60)
    print(f"Labels directory: {labels_dir}")
    print(f"Recursive: {args.recursive}")
    print()

    if not labels_dir.exists():
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return 1

    json_files = list(labels_dir.rglob("*.json")) if args.recursive else list(labels_dir.glob("*.json"))
    if not json_files:
        print("[WARNING] No JSON label files found")
        return 0

    total_objects = 0
    all_labels: list[str] = []
    all_shape_types: list[str] = []
    objects_per_image: list[int] = []
    image_sizes = Counter()

    for json_file in json_files:
        try:
            stats = analyze_json_label(json_file)
            total_objects += stats["num_objects"]
            all_labels.extend(stats["labels"])
            all_shape_types.extend(stats["shape_types"])
            objects_per_image.append(stats["num_objects"])
            size_key = f"{stats['width']}x{stats['height']}"
            image_sizes[size_key] += 1
        except Exception as exc:
            print(f"[WARNING] Error reading {json_file.name}: {exc}")

    print(f"Total label files: {len(json_files)}")
    print(f"Total objects:     {total_objects}")
    print()

    label_counts = Counter(all_labels)
    print("Class Distribution:")
    print("-" * 40)
    for label, count in label_counts.most_common():
        pct = count / total_objects * 100 if total_objects > 0 else 0
        print(f"  {label:25s} {count:5d} ({pct:5.1f}%)")
    print()

    shape_counts = Counter(all_shape_types)
    print("Shape Types:")
    print("-" * 40)
    for shape, count in shape_counts.most_common():
        print(f"  {shape:25s} {count:5d}")
    print()

    if objects_per_image:
        avg_objects = sum(objects_per_image) / len(objects_per_image)
        print("Objects per Image:")
        print("-" * 40)
        print(f"  Average: {avg_objects:.2f}")
        print(f"  Min:     {min(objects_per_image)}")
        print(f"  Max:     {max(objects_per_image)}")
        print(f"  Empty:   {objects_per_image.count(0)} images")
        print()

    print("Image Sizes:")
    print("-" * 40)
    for size, count in image_sizes.most_common():
        print(f"  {size:15s} {count:5d} images")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
