"""
Build a Kaggle-uploadable 2-class ripeness *classification* dataset by cropping
FFB bounding boxes from the COCO annotations.

Why crops?
The source "ffb-ripeness-classification" export contains images where both
Ripe-FFB and Unripe-FFB objects may appear in the same image. Image-level
labeling becomes ambiguous. Cropping per-object fixes this and increases samples.

Source (expected):
  Dataset/gohjinyu-oilpalm-ffb-dataset-d66eb99/ffb-ripeness-classification/
    - *.jpg
    - _annotations.coco.json  (COCO bbox annotations)

Output:
  Experiments/UploadKaggle/ffb_ripeness_crops/
    images/{train,val,test}/{ripe,unripe}/*.jpg
    ffb_ripeness_crops.yaml

Split:
  Stratified split on crops, deterministic with SPLIT_SEED.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = (
    PROJECT_DIR
    / "Dataset"
    / "gohjinyu-oilpalm-ffb-dataset-d66eb99"
    / "ffb-ripeness-classification"
)
COCO_PATH = SRC_DIR / "_annotations.coco.json"

OUT_DIR = PROJECT_DIR / "Experiments" / "UploadKaggle" / "ffb_ripeness_crops"
SPLITS = ("train", "val", "test")
CLASSES = ("ripe", "unripe")

SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

# Add a small context margin around bbox (10%)
BBOX_MARGIN_FRAC = 0.10


@dataclass(frozen=True)
class CropRec:
    src_file: str
    label: str  # 'ripe' | 'unripe'
    ann_id: int
    bbox_xywh: tuple[float, float, float, float]


def ensure_dirs() -> None:
    for split in SPLITS:
        for cls in CLASSES:
            (OUT_DIR / "images" / split / cls).mkdir(parents=True, exist_ok=True)


def write_yaml() -> None:
    (OUT_DIR / "ffb_ripeness_crops.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 2",
                "names: ['ripe', 'unripe']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_coco() -> dict:
    if not COCO_PATH.exists():
        raise FileNotFoundError(f"COCO annotation not found: {COCO_PATH}")
    return json.loads(COCO_PATH.read_text(encoding="utf-8"))


def build_category_id_map(coco: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for c in coco.get("categories", []):
        out[int(c["id"])] = str(c["name"])
    return out


def build_image_map(coco: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for img in coco.get("images", []):
        out[int(img["id"])] = str(img["file_name"])
    return out


def parse_crops(coco: dict) -> list[CropRec]:
    cat_id_to_name = build_category_id_map(coco)
    img_id_to_file = build_image_map(coco)

    crops: list[CropRec] = []
    skipped = 0

    for ann in coco.get("annotations", []):
        ann_id = int(ann.get("id", -1))
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            skipped += 1
            continue

        cat_name = cat_id_to_name.get(cat_id, "")
        if cat_name == "Ripe-FFB":
            label = "ripe"
        elif cat_name == "Unripe-FFB":
            label = "unripe"
        else:
            # Ignore "Fresh-Fruit-Bunch" base or any other categories
            continue

        file_name = img_id_to_file.get(img_id)
        if not file_name:
            skipped += 1
            continue

        x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if w <= 1 or h <= 1:
            skipped += 1
            continue

        crops.append(CropRec(src_file=file_name, label=label, ann_id=ann_id, bbox_xywh=(x, y, w, h)))

    if skipped:
        print(f"Skipped malformed annotations: {skipped}")
    return crops


def stratified_split(items: list[CropRec]) -> dict[str, list[CropRec]]:
    by_label: dict[str, list[CropRec]] = {c: [] for c in CLASSES}
    for it in items:
        by_label[it.label].append(it)

    rng = random.Random(SPLIT_SEED)
    for cls in CLASSES:
        rng.shuffle(by_label[cls])

    out: dict[str, list[CropRec]] = {s: [] for s in SPLITS}
    for cls, arr in by_label.items():
        n = len(arr)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        out["train"].extend(arr[:n_train])
        out["val"].extend(arr[n_train : n_train + n_val])
        out["test"].extend(arr[n_train + n_val : n_train + n_val + n_test])

    for s in SPLITS:
        rng.shuffle(out[s])
    return out


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def crop_and_save(rec: CropRec, dst_dir: Path) -> bool:
    src_path = SRC_DIR / rec.src_file
    if not src_path.exists():
        return False

    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    h_img, w_img = img.shape[:2]
    x, y, w, h = rec.bbox_xywh

    # margin
    mx = w * BBOX_MARGIN_FRAC
    my = h * BBOX_MARGIN_FRAC
    x1 = int(x - mx)
    y1 = int(y - my)
    x2 = int(x + w + mx)
    y2 = int(y + h + my)

    x1 = clamp(x1, 0, w_img - 1)
    y1 = clamp(y1, 0, h_img - 1)
    x2 = clamp(x2, 1, w_img)
    y2 = clamp(y2, 1, h_img)

    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return False

    crop = img[y1:y2, x1:x2]
    stem = Path(rec.src_file).stem
    out_name = f"{stem}__ann{rec.ann_id}.jpg"
    out_path = dst_dir / out_name

    # write JPEG
    ok = cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return bool(ok)


def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source dataset dir not found: {SRC_DIR}")

    ensure_dirs()
    write_yaml()

    coco = load_coco()
    crops = parse_crops(coco)
    print(f"Total crop candidates: {len(crops)}")

    label_counts = Counter(c.label for c in crops)
    print(f"Crop label counts: {dict(label_counts)}")

    splits = stratified_split(crops)

    written_counts = Counter()
    failed = 0
    for split in SPLITS:
        for rec in splits[split]:
            dst_dir = OUT_DIR / "images" / split / rec.label
            ok = crop_and_save(rec, dst_dir)
            if ok:
                written_counts[(split, rec.label)] += 1
            else:
                failed += 1

    print("\nWritten crops (by split, label):")
    for split in SPLITS:
        for cls in CLASSES:
            print(f"- {split}/{cls}: {written_counts[(split, cls)]}")

    if failed:
        print(f"\nWARNING: failed to write {failed} crops (missing file/read error/invalid bbox).")

    print(f"\nOutput ready at: {OUT_DIR}")
    print("Upload this folder (zip) to Kaggle Dataset (private).")


if __name__ == "__main__":
    main()

