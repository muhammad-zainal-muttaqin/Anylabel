"""
Build a Kaggle-uploadable YOLO *detection* dataset for ripeness (2 classes: ripe/unripe)
from the Goh 2025 COCO annotations.

This matches the "AnyLabeling style" workflow (bounding boxes with 2 classes),
without needing manual re-labeling.

Source (expected):
  Dataset/gohjinyu-oilpalm-ffb-dataset-d66eb99/ffb-ripeness-classification/
    - *.jpg
    - _annotations.coco.json

Output:
  Experiments/UploadKaggle/ffb_ripeness_detect/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt   (YOLO bbox labels)
    ffb_ripeness_detect.yaml

Class mapping:
  - "Ripe-FFB"   -> 0 (ripe_ffb)
  - "Unripe-FFB" -> 1 (unripe_ffb)
  - other categories are ignored

Split:
  Deterministic random split by image (70/20/10) with SPLIT_SEED.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = (
    PROJECT_DIR
    / "Dataset"
    / "gohjinyu-oilpalm-ffb-dataset-d66eb99"
    / "ffb-ripeness-classification"
)
COCO_PATH = SRC_DIR / "_annotations.coco.json"

OUT_DIR = PROJECT_DIR / "Experiments" / "UploadKaggle" / "ffb_ripeness_detect"
SPLITS = ("train", "val", "test")

SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

CLASS_NAMES = ["ripe_ffb", "unripe_ffb"]
CAT_TO_CLASS = {"Ripe-FFB": 0, "Unripe-FFB": 1}


@dataclass(frozen=True)
class ImageInfo:
    id: int
    file_name: str
    width: int
    height: int


def ensure_dirs() -> None:
    for split in SPLITS:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml() -> None:
    (OUT_DIR / "ffb_ripeness_detect.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "nc: 2",
                f"names: {CLASS_NAMES}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_coco() -> dict:
    if not COCO_PATH.exists():
        raise FileNotFoundError(f"COCO annotation not found: {COCO_PATH}")
    return json.loads(COCO_PATH.read_text(encoding="utf-8"))


def build_category_id_to_name(coco: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for c in coco.get("categories", []):
        out[int(c["id"])] = str(c["name"])
    return out


def build_images(coco: dict) -> dict[int, ImageInfo]:
    out: dict[int, ImageInfo] = {}
    for img in coco.get("images", []):
        image_id = int(img["id"])
        out[image_id] = ImageInfo(
            id=image_id,
            file_name=str(img["file_name"]),
            width=int(img["width"]),
            height=int(img["height"]),
        )
    return out


def yolo_line(cls_id: int, x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> str:
    # COCO bbox: top-left x,y + width,height in pixels
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h

    # clamp for safety
    def clamp01(v: float) -> float:
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    x_c = clamp01(x_c)
    y_c = clamp01(y_c)
    w_n = clamp01(w_n)
    h_n = clamp01(h_n)
    return f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"


def split_images(image_ids: list[int]) -> dict[str, list[int]]:
    rng = random.Random(SPLIT_SEED)
    ids = list(image_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * SPLIT_RATIOS["train"]))
    n_val = int(round(n * SPLIT_RATIOS["val"]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val : n_train + n_val + n_test],
    }


def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source dataset dir not found: {SRC_DIR}")

    ensure_dirs()
    write_yaml()

    coco = load_coco()
    cat_id_to_name = build_category_id_to_name(coco)
    images = build_images(coco)

    # collect labels per image
    labels_by_image: dict[int, list[str]] = defaultdict(list)
    kept_anns = 0
    ignored_anns = 0

    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        cat_name = cat_id_to_name.get(cat_id, "")
        if cat_name not in CAT_TO_CLASS:
            ignored_anns += 1
            continue

        img = images.get(image_id)
        if not img:
            ignored_anns += 1
            continue

        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            ignored_anns += 1
            continue

        x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if w <= 1 or h <= 1:
            ignored_anns += 1
            continue

        cls_id = CAT_TO_CLASS[cat_name]
        labels_by_image[image_id].append(yolo_line(cls_id, x, y, w, h, img.width, img.height))
        kept_anns += 1

    image_ids = sorted(images.keys())
    splits = split_images(image_ids)

    # copy images + write labels
    for split, ids in splits.items():
        for image_id in ids:
            img = images[image_id]
            src_img = SRC_DIR / img.file_name
            if not src_img.exists():
                continue
            dst_img = OUT_DIR / "images" / split / src_img.name
            shutil.copy2(src_img, dst_img)

            # YOLO label file must match image stem
            dst_label = OUT_DIR / "labels" / split / f"{src_img.stem}.txt"
            lines = labels_by_image.get(image_id, [])
            dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # report
    print(f"Images total: {len(image_ids)}")
    print(f"Annotations kept (ripe/unripe): {kept_anns}")
    print(f"Annotations ignored: {ignored_anns}")
    for split in SPLITS:
        n_imgs = len(list((OUT_DIR / 'images' / split).glob('*.jpg')))
        n_lbl = len(list((OUT_DIR / 'labels' / split).glob('*.txt')))
        print(f"- {split}: images={n_imgs}, labels={n_lbl}")
    print(f"Output ready at: {OUT_DIR}")


if __name__ == "__main__":
    main()

