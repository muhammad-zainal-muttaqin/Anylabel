"""
Build a Kaggle-uploadable RGB+Depth *paired* YOLO detection dataset (for A.3).

Why "paired" instead of a single 4-channel image?
- Ultralytics default dataloader expects 3-channel images.
- True RGB+Depth early-fusion (4-ch) needs a custom dataloader + model first-layer change.
- This script prepares the dataset cleanly so Kaggle training code can load RGB + Depth and concatenate.

Output directory:
  Experiments/UploadKaggle/ffb_localization_rgbd/
    rgb/{train,val,test}/*.png            (copied from the RGB split)
    depth/{train,val,test}/*.png          (single-channel uint8, normalized)
    labels/{train,val,test}/*.txt         (copied from the RGB split)
    ffb_localization_rgbd.yaml            (custom metadata for pairing)
    ffb_localization_rgbd.zip             (optional, created at the end)

Expected inputs:
- Experiments/UploadKaggle/ffb_localization/images/{train,val,test}/*.png
- Experiments/UploadKaggle/ffb_localization/labels/{train,val,test}/*.txt
- Depth source directory (auto-detected):
  - Experiments/datasets/depth_processed_rgb/*.png (preferred), OR
  - Dataset/.../ffb-localization/depth_maps/*.png

Note:
- Depth is stored as grayscale uint8 PNG to save space.
- Filenames are kept identical (e.g., rgb_0000.png) so pairing is trivial.
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
OUT_DATASET_DIR = UPLOADKAGGLE_DIR / "ffb_localization_rgbd"

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
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else None


def normalize_to_uint8(depth_img: np.ndarray) -> np.ndarray:
    """
    Convert depth image (any dtype) to uint8 (0-255) for training.
    Uses min-max normalization to avoid hard dependency on depth units.
    """
    depth_f = depth_img.astype(np.float32)
    norm = cv2.normalize(depth_f, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def ensure_dirs(out_root: Path) -> None:
    for split in SPLITS:
        (out_root / "rgb" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "depth" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml(out_root: Path) -> None:
    """
    This YAML is *not* a standard Ultralytics dataset YAML.
    It's metadata for a custom Kaggle notebook that loads RGB+Depth pairs.
    """
    yaml_path = out_root / "ffb_localization_rgbd.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "rgb: rgb",
                "depth: depth",
                "labels: labels",
                "train: train",
                "val: val",
                "test: test",
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
    idx: dict[str, Path] = {}
    for p in depth_dir.glob("*.png"):
        file_id = extract_id(p.stem)
        if not file_id:
            continue
        idx.setdefault(file_id, p)
    return idx


def build_split(
    *,
    split: str,
    rgb_images_dir: Path,
    rgb_labels_dir: Path,
    out_rgb_dir: Path,
    out_depth_dir: Path,
    out_labels_dir: Path,
    depth_index: dict[str, Path],
) -> tuple[int, int, int]:
    """
    Returns (pairs_written, labels_copied, missing_depth_or_write).
    """
    labels_copied = 0
    pairs_written = 0
    missing = 0

    for rgb_img in rgb_images_dir.glob("*.png"):
        img_id = extract_id(rgb_img.stem)
        if not img_id or img_id not in depth_index:
            missing += 1
            continue

        label_src = rgb_labels_dir / f"{rgb_img.stem}.txt"
        if not label_src.exists():
            missing += 1
            continue

        # Copy RGB image and label
        shutil.copy2(rgb_img, out_rgb_dir / rgb_img.name)
        shutil.copy2(label_src, out_labels_dir / label_src.name)
        labels_copied += 1

        # Read + normalize depth and save as grayscale uint8 PNG
        depth_src = depth_index[img_id]
        depth_img = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            missing += 1
            continue

        if depth_img.ndim == 3:
            # If depth is already 3-ch (e.g., depth_processed_rgb), take one channel
            depth_img = depth_img[:, :, 0]

        depth_u8 = depth_img if depth_img.dtype == np.uint8 else normalize_to_uint8(depth_img)
        ok = cv2.imwrite(str(out_depth_dir / rgb_img.name), depth_u8)
        if not ok:
            missing += 1
            continue

        pairs_written += 1

    return pairs_written, labels_copied, missing


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

    total_pairs = 0
    total_labels = 0
    total_missing = 0

    for split in SPLITS:
        rgb_images_dir = RGB_DATASET_DIR / "images" / split
        rgb_labels_dir = RGB_DATASET_DIR / "labels" / split

        if not rgb_images_dir.exists():
            raise FileNotFoundError(f"Missing RGB images split dir: {rgb_images_dir}")
        if not rgb_labels_dir.exists():
            raise FileNotFoundError(f"Missing RGB labels split dir: {rgb_labels_dir}")

        out_rgb_dir = OUT_DATASET_DIR / "rgb" / split
        out_depth_dir = OUT_DATASET_DIR / "depth" / split
        out_labels_dir = OUT_DATASET_DIR / "labels" / split

        pairs_written, labels_copied, missing = build_split(
            split=split,
            rgb_images_dir=rgb_images_dir,
            rgb_labels_dir=rgb_labels_dir,
            out_rgb_dir=out_rgb_dir,
            out_depth_dir=out_depth_dir,
            out_labels_dir=out_labels_dir,
            depth_index=depth_index,
        )

        total_pairs += pairs_written
        total_labels += labels_copied
        total_missing += missing
        print(
            f"[{split}] pairs_written={pairs_written}, labels_copied={labels_copied}, missing_depth_or_label_or_write={missing}"
        )

    # Create a zip archive for Kaggle upload convenience
    zip_base = str(OUT_DATASET_DIR / "ffb_localization_rgbd")
    archive_path = shutil.make_archive(zip_base, "zip", root_dir=OUT_DATASET_DIR)

    print("Done.")
    print(f"Total pairs written: {total_pairs}")
    print(f"Total labels copied: {total_labels}")
    print(f"Zip created: {archive_path}")
    if total_missing:
        print(
            f"WARNING: {total_missing} samples missing depth/label match or failed to write. "
            "Check filename mapping between rgb_#### and depth_####."
        )


if __name__ == "__main__":
    main()

