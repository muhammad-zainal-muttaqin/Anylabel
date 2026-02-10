"""
Verify labeling consistency by comparing image files and label files.

Supports recursive scan and relative-path matching to avoid filename collisions.
"""

import argparse
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
LABEL_EXTS = {".json", ".txt"}


def collect_stems(base_dir: Path, exts: set[str], recursive: bool) -> set[str]:
    paths = base_dir.rglob("*") if recursive else base_dir.glob("*")
    stems: set[str] = set()
    for path in paths:
        if path.is_file() and path.suffix.lower() in exts:
            rel = path.relative_to(base_dir).with_suffix("")
            stems.add(rel.as_posix())
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify images and labels consistency")
    parser.add_argument("--images", "-i", type=str, default="Dataset", help="Directory containing images")
    parser.add_argument("--labels", "-l", type=str, default="Dataset", help="Directory containing labels")
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    images_dir = (root_dir / args.images).resolve()
    labels_dir = (root_dir / args.labels).resolve()

    print("=" * 60)
    print("  Label Verification Tool")
    print("=" * 60)
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Recursive: {args.recursive}")
    print()

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return 1
    if not labels_dir.exists():
        print(f"[WARNING] Labels directory not found: {labels_dir}")
        print("          No labels to verify yet.")
        return 0

    image_stems = collect_stems(images_dir, IMAGE_EXTS, args.recursive)
    json_stems = collect_stems(labels_dir, {".json"}, args.recursive)
    txt_stems = collect_stems(labels_dir, {".txt"}, args.recursive)
    label_stems = json_stems | txt_stems

    print(f"Images found: {len(image_stems)}")
    print(f"Labels found: {len(label_stems)} (JSON: {len(json_stems)}, TXT: {len(txt_stems)})")
    print()

    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems
    matched = image_stems & label_stems

    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Matched pairs:          {len(matched)}")
    print(f"  Images without labels:  {len(images_without_labels)}")
    print(f"  Labels without images:  {len(labels_without_images)}")
    print()

    if images_without_labels:
        print("Images without labels (need labeling):")
        for name in sorted(images_without_labels)[:10]:
            print(f"  - {name}")
        if len(images_without_labels) > 10:
            print(f"  ... and {len(images_without_labels) - 10} more")
        print()

    if labels_without_images:
        print("Labels without images (orphaned):")
        for name in sorted(labels_without_images)[:10]:
            print(f"  - {name}")
        if len(labels_without_images) > 10:
            print(f"  ... and {len(labels_without_images) - 10} more")
        print()

    if image_stems:
        progress = len(matched) / len(image_stems) * 100
        print(f"Labeling progress: {progress:.1f}% ({len(matched)}/{len(image_stems)})")
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"[{bar}]")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
