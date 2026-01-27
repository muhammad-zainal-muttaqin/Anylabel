"""
Build a Kaggle-uploadable depth-only YOLO detection dataset using the existing RGB split.

Key idea:
- Reuse the existing YOLO labels from RGB split (same boxes/classes).
- Replace RGB images with depth images converted to 3-channel PNG.
- Output is written to: Experiments/UploadKaggle/ffb_localization_depth/

Expected inputs:
- Experiments/UploadKaggle/ffb_localization/images/{train,val,test}/*.png
- Experiments/UploadKaggle/ffb_localization/labels/{train,val,test}/*.txt
- Depth source directory (auto-detected):
  - Experiments/datasets/depth_processed_rgb/*.png (preferred), OR
  - Dataset/.../ffb-localization/depth_maps/*.png

The output images are saved using the RGB filenames (e.g., rgb_0000.png) so labels match.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
UPLOADKAGGLE_DIR = PROJECT_DIR / "Experiments" / "UploadKaggle"

RGB_DATASET_DIR = UPLOADKAGGLE_DIR / "ffb_localization"
OUT_DATASET_DIR = UPLOADKAGGLE_DIR / "ffb_localization_depth"

# Candidate depth sources (auto pick first that exists)
DEPTH_SOURCES = [
    PROJECT_DIR / "Experiments" / "datasets" / "depth_processed_rgb",
    PROJECT_DIR
    / "Dataset"
    / "gohjinyu-oilpalm-ffb-dataset-d66eb99"
    / "ffb-localization"
    / "depth_maps",
]

SPLITS = ("train", "val", "test")


def extract_id(stem: str) -> str | None:
    """
    Extract numeric id from a filename stem.
    Examples:
    - rgb_0000 -> 0000
    - depth_0000 -> 0000
    """
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else None


def normalize_to_uint8(depth_img: np.ndarray) -> np.ndarray:
    """
    Convert depth image (any dtype) to uint8 (0-255) for visualization/training.

    We use min-max normalization to avoid wrong unit assumptions (mm vs meters).
    """
    depth_f = depth_img.astype(np.float32)
    norm = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def ensure_dirs(out_root: Path) -> None:
    for split in SPLITS:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml(out_root: Path) -> None:
    yaml_path = out_root / "ffb_localization_depth.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names: ['fresh_fruit_bunch']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def pick_depth_source() -> Path:
    for p in DEPTH_SOURCES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No depth source directory found. Checked:\n"
        + "\n".join(str(p) for p in DEPTH_SOURCES)
    )


def build_depth_index(depth_dir: Path) -> dict[str, Path]:
    """
    Map numeric id -> depth file path.
    If duplicates exist, keep the first.
    """
    idx: dict[str, Path] = {}
    for p in depth_dir.glob("*.png"):
        file_id = extract_id(p.stem)
        if not file_id:
            continue
        idx.setdefault(file_id, p)
    return idx


def copy_labels(rgb_labels_dir: Path, out_labels_dir: Path) -> int:
    count = 0
    for label_path in rgb_labels_dir.glob("*.txt"):
        shutil.copy2(label_path, out_labels_dir / label_path.name)
        count += 1
    return count


def build_split(
    *,
    split: str,
    rgb_images_dir: Path,
    rgb_labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    depth_index: dict[str, Path],
) -> tuple[int, int, int]:
    """
    Returns (images_written, labels_copied, missing_depth).
    """
    labels_copied = copy_labels(rgb_labels_dir, out_labels_dir)

    images_written = 0
    missing = 0

    for rgb_img in rgb_images_dir.glob("*.png"):
        img_id = extract_id(rgb_img.stem)
        if not img_id or img_id not in depth_index:
            missing += 1
            continue

        depth_src = depth_index[img_id]
        depth_img = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            missing += 1
            continue

        # Normalize + replicate to 3 channels
        if depth_img.ndim == 2:
            norm_u8 = normalize_to_uint8(depth_img)
            depth_3ch = cv2.merge([norm_u8, norm_u8, norm_u8])
        else:
            # If already 3-channel, keep as-is (ensure uint8)
            if depth_img.dtype != np.uint8:
                depth_img = normalize_to_uint8(depth_img)
                depth_3ch = cv2.merge([depth_img, depth_img, depth_img])
            else:
                depth_3ch = depth_img

        out_path = out_images_dir / rgb_img.name  # keep RGB filename for label matching
        ok = cv2.imwrite(str(out_path), depth_3ch)
        if not ok:
            missing += 1
            continue
        images_written += 1

    return images_written, labels_copied, missing


def main() -> None:
    if not RGB_DATASET_DIR.exists():
        raise FileNotFoundError(f"RGB Kaggle dataset not found at: {RGB_DATASET_DIR}")

    depth_dir = pick_depth_source()
    depth_index = build_depth_index(depth_dir)

    if not depth_index:
        raise RuntimeError(f"No depth PNG files indexed in: {depth_dir}")

    ensure_dirs(OUT_DATASET_DIR)
    write_yaml(OUT_DATASET_DIR)

    print(f"Depth source: {depth_dir}")
    print(f"Depth index size: {len(depth_index)}")
    print(f"Output: {OUT_DATASET_DIR}")

    total_images = 0
    total_labels = 0
    total_missing = 0

    for split in SPLITS:
        rgb_images_dir = RGB_DATASET_DIR / "images" / split
        rgb_labels_dir = RGB_DATASET_DIR / "labels" / split
        out_images_dir = OUT_DATASET_DIR / "images" / split
        out_labels_dir = OUT_DATASET_DIR / "labels" / split

        if not rgb_images_dir.exists():
            raise FileNotFoundError(f"Missing RGB images split dir: {rgb_images_dir}")
        if not rgb_labels_dir.exists():
            raise FileNotFoundError(f"Missing RGB labels split dir: {rgb_labels_dir}")

        images_written, labels_copied, missing = build_split(
            split=split,
            rgb_images_dir=rgb_images_dir,
            rgb_labels_dir=rgb_labels_dir,
            out_images_dir=out_images_dir,
            out_labels_dir=out_labels_dir,
            depth_index=depth_index,
        )

        total_images += images_written
        total_labels += labels_copied
        total_missing += missing

        print(
            f"[{split}] images_written={images_written}, labels_copied={labels_copied}, missing_depth_or_write={missing}"
        )

    print("Done.")
    print(f"Total images written: {total_images}")
    print(f"Total labels copied: {total_labels}")
    if total_missing:
        print(
            f"WARNING: {total_missing} images missing depth match or failed to write. "
            "Check filename mapping between rgb_#### and depth_####."
        )


if __name__ == "__main__":
    main()

