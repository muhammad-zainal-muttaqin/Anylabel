"""
Build a Kaggle-uploadable 2-class ripeness classification dataset (ripe/unripe)
from the Goh 2025 dataset COCO annotations.

Source (expected):
  Dataset/gohjinyu-oilpalm-ffb-dataset-d66eb99/ffb-ripeness-classification/
    - *.jpg
    - _annotations.coco.json

Output:
  Experiments/UploadKaggle/ffb_ripeness/
    images/{train,val,test}/{ripe,unripe}/*.jpg
    ffb_ripeness.yaml

Labeling rule:
  - If an image has only 'Ripe-FFB' annotations => label 'ripe'
  - If an image has only 'Unripe-FFB' annotations => label 'unripe'
  - If both present or none present => skip (counted as ambiguous/unknown)
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
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

OUT_DIR = PROJECT_DIR / "Experiments" / "UploadKaggle" / "ffb_ripeness"
SPLITS = ("train", "val", "test")
CLASSES = ("ripe", "unripe")

# Deterministic split for reproducibility
SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


@dataclass(frozen=True)
class Record:
    file_name: str
    label: str  # 'ripe' | 'unripe'


def ensure_dirs() -> None:
    for split in SPLITS:
        for cls in CLASSES:
            (OUT_DIR / "images" / split / cls).mkdir(parents=True, exist_ok=True)


def write_yaml() -> None:
    (OUT_DIR / "ffb_ripeness.yaml").write_text(
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


def build_category_maps(coco: dict) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    for c in coco.get("categories", []):
        cid = int(c["id"])
        name = str(c["name"])
        id_to_name[cid] = name
        name_to_id[name] = cid
    return id_to_name, name_to_id


def derive_image_labels(coco: dict) -> list[Record]:
    id_to_name, _ = build_category_maps(coco)

    image_id_to_file: dict[int, str] = {}
    for img in coco.get("images", []):
        image_id_to_file[int(img["id"])] = str(img["file_name"])

    image_id_to_categories: dict[int, set[str]] = defaultdict(set)
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        cat_name = id_to_name.get(cat_id, f"unknown_{cat_id}")
        image_id_to_categories[image_id].add(cat_name)

    records: list[Record] = []
    skipped_ambiguous = 0
    skipped_unknown = 0

    for image_id, file_name in image_id_to_file.items():
        cats = image_id_to_categories.get(image_id, set())
        has_ripe = "Ripe-FFB" in cats
        has_unripe = "Unripe-FFB" in cats

        if has_ripe and not has_unripe:
            records.append(Record(file_name=file_name, label="ripe"))
        elif has_unripe and not has_ripe:
            records.append(Record(file_name=file_name, label="unripe"))
        elif has_ripe and has_unripe:
            skipped_ambiguous += 1
        else:
            skipped_unknown += 1

    print(f"Total images in COCO: {len(image_id_to_file)}")
    print(f"Labeled images: {len(records)}")
    print(f"Skipped (ambiguous ripe+unripe): {skipped_ambiguous}")
    print(f"Skipped (no ripe/unripe annotations): {skipped_unknown}")
    return records


def stratified_split(records: list[Record]) -> dict[str, list[Record]]:
    by_label: dict[str, list[Record]] = {c: [] for c in CLASSES}
    for r in records:
        if r.label in by_label:
            by_label[r.label].append(r)

    rng = random.Random(SPLIT_SEED)
    for cls in CLASSES:
        rng.shuffle(by_label[cls])

    split_out: dict[str, list[Record]] = {s: [] for s in SPLITS}
    for cls, items in by_label.items():
        n = len(items)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))
        # Ensure totals match
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        split_out["train"].extend(items[:n_train])
        split_out["val"].extend(items[n_train : n_train + n_val])
        split_out["test"].extend(items[n_train + n_val : n_train + n_val + n_test])

    # Shuffle within each split for nicer ordering
    for s in SPLITS:
        rng.shuffle(split_out[s])

    return split_out


def copy_split(split_name: str, records: list[Record]) -> None:
    for r in records:
        src = SRC_DIR / r.file_name
        if not src.exists():
            # Some COCO exports may reference files not present; skip safely.
            continue
        dst = OUT_DIR / "images" / split_name / r.label / src.name
        shutil.copy2(src, dst)


def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source dataset dir not found: {SRC_DIR}")

    ensure_dirs()
    write_yaml()

    coco = load_coco()
    records = derive_image_labels(coco)
    splits = stratified_split(records)

    for s in SPLITS:
        copy_split(s, splits[s])

    counts = {
        s: Counter(r.label for r in splits[s])
        for s in SPLITS
    }
    print("\nSplit summary (by label):")
    for s in SPLITS:
        print(f"- {s}: {dict(counts[s])} (total={sum(counts[s].values())})")

    print(f"\nOutput ready at: {OUT_DIR}")
    print("Upload this folder (zip) to Kaggle Dataset (private).")


if __name__ == "__main__":
    main()

