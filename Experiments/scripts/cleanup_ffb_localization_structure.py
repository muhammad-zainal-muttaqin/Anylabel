"""
Clean and simplify the FFB localization folder structure.

Target (after cleanup):

Experiments/datasets/ffb_localization/
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt

Experiments/labeling/ffb_localization/
  json/*.json              # AnyLabeling (LabelMe) archive
  yolo_all/*.txt           # One .txt per image stem (source for splitting)

Notes:
- This script MOVES files into the new structure (no destructive overwrite).
- If destination exists and content differs, the source is moved with a suffix
  into the same destination folder for manual review.
- Training splits are never modified by this script.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_move_file(src: Path, dst: Path) -> str:
    """
    Move src to dst safely.
    Returns: "moved", "skipped_same", or "conflict_renamed".
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return "moved"

    try:
        if sha256(src) == sha256(dst):
            # Same content -> remove source
            src.unlink()
            return "skipped_same"
    except Exception:
        pass

    # Conflict: keep both (rename source)
    conflict = dst.parent / f"{src.stem}__conflict{src.suffix}"
    i = 1
    while conflict.exists():
        conflict = dst.parent / f"{src.stem}__conflict_{i}{src.suffix}"
        i += 1
    shutil.move(str(src), str(conflict))
    return "conflict_renamed"


def copy_split_labels_to_yolo_all(labels_split_dir: Path, yolo_all_dir: Path) -> None:
    yolo_all_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_dir = labels_split_dir / split
        if not split_dir.is_dir():
            continue
        for src in split_dir.glob("*.txt"):
            dst = yolo_all_dir / src.name
            # Copy (not move) to keep training splits intact
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            else:
                # If exists but differs, keep both
                try:
                    if sha256(src) == sha256(dst):
                        continue
                except Exception:
                    pass
                conflict = yolo_all_dir / f"{src.stem}__from_split_{split}{src.suffix}"
                i = 1
                while conflict.exists():
                    conflict = yolo_all_dir / f"{src.stem}__from_split_{split}_{i}{src.suffix}"
                    i += 1
                shutil.copy2(str(src), str(conflict))


def maybe_rmtree_if_empty(dir_path: Path) -> bool:
    if not dir_path.exists():
        return False
    # Remove only if empty (including subdirs)
    entries = list(dir_path.rglob("*"))
    if any(p.is_file() for p in entries):
        return False
    shutil.rmtree(str(dir_path))
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = repo_root / "Experiments" / "datasets" / "ffb_localization"
    labels_split_dir = dataset_dir / "labels"

    labeling_root = repo_root / "Experiments" / "labeling" / "ffb_localization"
    json_dir = labeling_root / "json"
    yolo_all_dir = labeling_root / "yolo_all"

    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}")
        return 2
    if not labels_split_dir.is_dir():
        print(f"ERROR: split labels dir not found: {labels_split_dir}")
        return 2

    json_dir.mkdir(parents=True, exist_ok=True)
    yolo_all_dir.mkdir(parents=True, exist_ok=True)

    # 1) Always ensure yolo_all is complete by copying from split labels
    copy_split_labels_to_yolo_all(labels_split_dir, yolo_all_dir)

    moved_json = skipped_json = conflict_json = 0
    moved_txt = skipped_txt = conflict_txt = 0

    # 2) Move legacy json_archive/**/*.json -> labeling/json
    legacy_json_archive = dataset_dir / "json_archive"
    if legacy_json_archive.exists():
        for src in legacy_json_archive.rglob("*.json"):
            result = safe_move_file(src, json_dir / src.name)
            if result == "moved":
                moved_json += 1
            elif result == "skipped_same":
                skipped_json += 1
            else:
                conflict_json += 1

    # 3) Move legacy labels_all/*.txt -> labeling/yolo_all
    legacy_labels_all = dataset_dir / "labels_all"
    if legacy_labels_all.exists():
        for src in legacy_labels_all.glob("*.txt"):
            result = safe_move_file(src, yolo_all_dir / src.name)
            if result == "moved":
                moved_txt += 1
            elif result == "skipped_same":
                skipped_txt += 1
            else:
                conflict_txt += 1

    # 4) Remove redundant dirs if empty
    removed_legacy_labels_all = maybe_rmtree_if_empty(legacy_labels_all)
    removed_legacy_json_archive = maybe_rmtree_if_empty(legacy_json_archive)

    # 5) Summary
    total_yolo_all = len(list(yolo_all_dir.glob("*.txt")))
    total_json = len(list(json_dir.glob("*.json")))
    print("OK: Cleanup completed.")
    print(f"- dataset (training): {dataset_dir}")
    print(f"- labeling archive:   {labeling_root}")
    print("")
    print("Moved from legacy folders:")
    print(f"- json moved/skipped_same/conflict: {moved_json}/{skipped_json}/{conflict_json}")
    print(f"- txt moved/skipped_same/conflict:  {moved_txt}/{skipped_txt}/{conflict_txt}")
    print("")
    print("Final counts:")
    print(f"- labeling/json:      {total_json} *.json")
    print(f"- labeling/yolo_all:  {total_yolo_all} *.txt")
    print("")
    print("Removed legacy folders (if empty):")
    print(f"- removed datasets/.../labels_all:  {int(removed_legacy_labels_all)}")
    print(f"- removed datasets/.../json_archive:{int(removed_legacy_json_archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

