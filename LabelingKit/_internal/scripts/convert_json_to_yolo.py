"""
Convert AnyLabeling JSON (LabelMe format) to YOLO format (.txt).
"""

import argparse
import json
from pathlib import Path


DEFAULT_RIPENESS_CLASSES = ["B1", "B2", "B3", "B4"]


def load_classes_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    classes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label = line.strip()
        if not label or label.startswith("#"):
            continue
        classes.append(label)
    return classes


def build_class_mapping(root_dir: Path) -> dict[str, int]:
    default_file = (root_dir / "_internal" / "configs" / "classes.txt").resolve()
    class_names = load_classes_from_file(default_file)
    if not class_names:
        class_names = list(DEFAULT_RIPENESS_CLASSES)

    unique_names: list[str] = []
    seen: set[str] = set()
    for name in class_names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)

    return {name: idx for idx, name in enumerate(unique_names)}


def convert_rectangle_to_yolo(points, img_width, img_height):
    x1, y1 = points[0]
    x2, y2 = points[1]

    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    x_center = ((x_min + x_max) / 2.0) / img_width
    y_center = ((y_min + y_max) / 2.0) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def convert_json_to_yolo(json_path: Path, output_txt_path: Path, class_mapping: dict[str, int]) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_width = data.get("imageWidth", 1280)
    img_height = data.get("imageHeight", 720)
    shapes = data.get("shapes", [])

    yolo_lines: list[str] = []
    for shape in shapes:
        if shape.get("shape_type") != "rectangle":
            continue

        label = shape.get("label", "")
        if label not in class_mapping:
            print(f"  Warning: Unknown label '{label}' in {json_path.name}, skipping")
            continue

        points = shape.get("points", [])
        if len(points) != 2:
            print(f"  Warning: Invalid points in {json_path.name}, skipping")
            continue

        class_id = class_mapping[label]
        x_center, y_center, width, height = convert_rectangle_to_yolo(points, img_width, img_height)
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.writelines(yolo_lines)

    return len(yolo_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AnyLabeling JSON to YOLO format")
    parser.add_argument("--input", "-i", type=str, default="Dataset", help="Input directory containing JSON files")
    parser.add_argument("--output", "-o", type=str, default="output/yolo", help="Output directory for YOLO txt files")
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively")
    parser.add_argument("--preserve_structure", action="store_true", help="Keep relative folder structure in output")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    labels_dir = (root_dir / args.input).resolve()
    output_dir = (root_dir / args.output).resolve()
    class_mapping = build_class_mapping(root_dir=root_dir)

    print("=" * 60)
    print("  JSON to YOLO Converter")
    print("=" * 60)
    print(f"Input:  {labels_dir}")
    print(f"Output: {output_dir}")
    print("Mode: ripeness (fixed 4 classes)")
    print(f"Classes: {class_mapping}")
    print(f"Recursive: {args.recursive}")
    print(f"Preserve structure: {args.preserve_structure}")
    print()

    if not labels_dir.exists():
        print(f"[ERROR] Input directory not found: {labels_dir}")
        return 1

    json_files = list(labels_dir.rglob("*.json")) if args.recursive else list(labels_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON files found in {labels_dir}")
        return 1

    converted = 0
    total_objects = 0
    empty_files = 0

    for json_file in json_files:
        try:
            rel = json_file.relative_to(labels_dir)
            if args.preserve_structure:
                out_txt = output_dir / rel.with_suffix(".txt")
            else:
                out_txt = output_dir / f"{json_file.stem}.txt"

            num_objects = convert_json_to_yolo(json_file, out_txt, class_mapping)
            converted += 1
            total_objects += num_objects
            if num_objects == 0:
                empty_files += 1
        except Exception as exc:
            print(f"  [ERROR] {json_file.name}: {exc}")

    print()
    print("=" * 60)
    print("  Conversion Summary")
    print("=" * 60)
    print(f"  Total JSON files:       {len(json_files)}")
    print(f"  Successfully converted: {converted}")
    print(f"  Empty files:            {empty_files}")
    print(f"  Total objects:          {total_objects}")
    if converted > 0:
        print(f"  Avg objects/image:      {total_objects / converted:.2f}")
    print()
    print(f"YOLO files saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
