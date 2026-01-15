import argparse
import csv
import glob
import importlib
import json
import math
import os
import statistics
import struct

try:
    cv2 = importlib.import_module("cv2")
    np = importlib.import_module("numpy")
except Exception:
    cv2 = None
    np = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_DATASET_DIR = os.path.join(PROJECT_DIR, "Dataset", "gohjinyu-oilpalm-ffb-dataset-d66eb99")
RAW_LOCALIZATION_DIR = os.path.join(RAW_DATASET_DIR, "ffb-localization")
RAW_RGB_DIR = os.path.join(RAW_LOCALIZATION_DIR, "rgb_images")
RAW_DEPTH_DIR = os.path.join(RAW_LOCALIZATION_DIR, "depth_maps")
RAW_PC_DIR = os.path.join(RAW_LOCALIZATION_DIR, "point_clouds")
RAW_METADATA_FILE = os.path.join(RAW_LOCALIZATION_DIR, "metadata", "metadata.csv")
RAW_LABELS_DIR = os.path.join(RAW_LOCALIZATION_DIR, "labels_yolo")

EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, "Experiments")
PROCESSED_DATASETS_DIR = os.path.join(EXPERIMENTS_DIR, "datasets")
PROCESSED_LOCALIZATION_DIR = os.path.join(PROCESSED_DATASETS_DIR, "ffb_localization")
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_LOCALIZATION_DIR, "images")
PROCESSED_LABELS_DIR = os.path.join(PROCESSED_LOCALIZATION_DIR, "labels")

RIPENESS_DIR = os.path.join(RAW_DATASET_DIR, "ffb-ripeness-classification")

DEFAULT_OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "eda_output")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_COLOR_TYPE_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
PNG_COLOR_TYPE_NAME = {0: "grayscale", 2: "rgb", 3: "indexed", 4: "grayscale_alpha", 6: "rgba"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows, fieldnames=None) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows and not fieldnames:
        fieldnames = []
    if fieldnames is None:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def is_finite_number(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def parse_numeric_id(filename: str):
    digits = "".join([c for c in filename if c.isdigit()])
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def read_png_info(path: str):
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
            if sig != PNG_SIGNATURE:
                return None
            length_bytes = f.read(4)
            if len(length_bytes) != 4:
                return None
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = f.read(4)
            if chunk_type != b"IHDR":
                return None
            data = f.read(length)
            if len(data) != length or len(data) < 13:
                return None
            width, height = struct.unpack(">II", data[:8])
            bit_depth = int(data[8])
            color_type = int(data[9])
            channels = PNG_COLOR_TYPE_CHANNELS.get(color_type)
            return {
                "width": int(width),
                "height": int(height),
                "bit_depth": bit_depth,
                "color_type": color_type,
                "color_type_name": PNG_COLOR_TYPE_NAME.get(color_type, "unknown"),
                "channels": channels if channels is not None else "",
            }
    except Exception:
        return None


def summarize_numbers(values):
    clean = [v for v in values if is_finite_number(v)]
    if not clean:
        return {"count": 0}
    clean_sorted = sorted(clean)
    n = len(clean_sorted)
    def pct(p: float) -> float:
        if n == 1:
            return clean_sorted[0]
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return clean_sorted[lo]
        frac = idx - lo
        return clean_sorted[lo] * (1 - frac) + clean_sorted[hi] * frac
    return {
        "count": n,
        "min": clean_sorted[0],
        "p10": pct(0.10),
        "median": pct(0.50),
        "p90": pct(0.90),
        "max": clean_sorted[-1],
        "mean": float(statistics.fmean(clean_sorted)) if n else 0.0,
    }


def analyze_png_files(file_paths, out_csv_path: str, max_files=None):
    rows = []
    widths = []
    heights = []
    bit_depths = {}
    color_types = {}

    for p in file_paths[: max_files or None]:
        info = read_png_info(p)
        size_bytes = os.path.getsize(p) if os.path.exists(p) else 0
        row = {
            "filename": os.path.basename(p),
            "path": p,
            "file_size_bytes": size_bytes,
        }
        if info:
            row.update(info)
            widths.append(float(info["width"]))
            heights.append(float(info["height"]))
            bd = int(info["bit_depth"])
            bit_depths[bd] = bit_depths.get(bd, 0) + 1
            ct = str(info.get("color_type_name") or "unknown")
            color_types[ct] = color_types.get(ct, 0) + 1
        else:
            row.update({"width": "", "height": "", "bit_depth": "", "color_type": "", "color_type_name": "", "channels": ""})
        rows.append(row)

    write_csv(out_csv_path, rows)

    return {
        "files": len(file_paths),
        "analyzed": len(rows),
        "width": summarize_numbers(widths),
        "height": summarize_numbers(heights),
        "bit_depth_counts": bit_depths,
        "color_type_counts": color_types,
    }


def read_metadata_csv(path: str):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def analyze_localization_integrity(out_dir: str):
    ensure_dir(out_dir)

    rgb_files = sorted(glob.glob(os.path.join(RAW_RGB_DIR, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(RAW_DEPTH_DIR, "*.png")))
    pc_files = sorted(glob.glob(os.path.join(RAW_PC_DIR, "*.ply")))
    metadata_rows = read_metadata_csv(RAW_METADATA_FILE)

    rgb_by_id = {}
    depth_by_id = {}
    pc_by_id = {}
    md_by_id = {}

    for p in rgb_files:
        i = parse_numeric_id(os.path.basename(p))
        if i is not None:
            rgb_by_id[i] = p
    for p in depth_files:
        i = parse_numeric_id(os.path.basename(p))
        if i is not None:
            depth_by_id[i] = p
    for p in pc_files:
        i = parse_numeric_id(os.path.basename(p))
        if i is not None:
            pc_by_id[i] = p

    for r in metadata_rows:
        try:
            i = int(str(r.get("image_count", "")).strip())
            md_by_id[i] = r
        except Exception:
            continue

    all_ids = sorted(set(rgb_by_id.keys()) | set(depth_by_id.keys()) | set(pc_by_id.keys()) | set(md_by_id.keys()))

    rows = []
    missing_rgb = []
    missing_depth = []
    missing_pc = []
    missing_metadata = []

    for i in all_ids:
        has_rgb = i in rgb_by_id
        has_depth = i in depth_by_id
        has_pc = i in pc_by_id
        has_md = i in md_by_id
        if not has_rgb:
            missing_rgb.append(i)
        if not has_depth:
            missing_depth.append(i)
        if not has_pc:
            missing_pc.append(i)
        if not has_md:
            missing_metadata.append(i)
        rows.append(
            {
                "id": i,
                "rgb": os.path.basename(rgb_by_id[i]) if has_rgb else "",
                "depth": os.path.basename(depth_by_id[i]) if has_depth else "",
                "point_cloud": os.path.basename(pc_by_id[i]) if has_pc else "",
                "metadata_rgb": (md_by_id[i].get("rgb_image") if has_md else "") or "",
                "metadata_depth": (md_by_id[i].get("depth_map") if has_md else "") or "",
                "metadata_point_cloud": (md_by_id[i].get("point_cloud") if has_md else "") or "",
                "metadata_timestamp": (md_by_id[i].get("timestamp") if has_md else "") or "",
                "has_rgb": int(has_rgb),
                "has_depth": int(has_depth),
                "has_point_cloud": int(has_pc),
                "has_metadata": int(has_md),
            }
        )

    write_csv(os.path.join(out_dir, "localization_integrity_raw.csv"), rows)

    return {
        "rgb_files": len(rgb_files),
        "depth_files": len(depth_files),
        "point_cloud_files": len(pc_files),
        "metadata_rows": len(metadata_rows),
        "unique_ids": len(all_ids),
        "missing_rgb": len(missing_rgb),
        "missing_depth": len(missing_depth),
        "missing_point_cloud": len(missing_pc),
        "missing_metadata": len(missing_metadata),
    }


def try_depth_stats(depth_paths, max_files=100):
    if cv2 is None or np is None:
        return None
    sample = depth_paths[: max_files or None]
    mins = []
    maxs = []
    means = []
    dtypes = {}
    for p in sample:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        dtypes[str(img.dtype)] = dtypes.get(str(img.dtype), 0) + 1
        arr = img.astype(np.float64)
        mins.append(float(np.min(arr)))
        maxs.append(float(np.max(arr)))
        means.append(float(np.mean(arr)))
    return {
        "sampled": len(sample),
        "dtype_counts": dtypes,
        "min": summarize_numbers(mins),
        "max": summarize_numbers(maxs),
        "mean": summarize_numbers(means),
    }


def find_label_roots():
    candidates = []
    if os.path.exists(RAW_LABELS_DIR):
        candidates.append(("raw_labels_yolo", RAW_LABELS_DIR))
    if os.path.exists(PROCESSED_LABELS_DIR):
        candidates.append(("processed_labels", PROCESSED_LABELS_DIR))
    flat_exp_labels = os.path.join(PROCESSED_LOCALIZATION_DIR, "labels")
    if os.path.exists(flat_exp_labels) and ("processed_labels", flat_exp_labels) not in candidates:
        candidates.append(("processed_labels", flat_exp_labels))
    return candidates


def find_image_for_stem(images_dir: str, stem: str):
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    matches = glob.glob(os.path.join(images_dir, stem + ".*"))
    return matches[0] if matches else None


def parse_yolo_label_file(path: str):
    boxes = []
    invalid_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                invalid_lines += 1
                continue
            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                if not all(is_finite_number(v) for v in [x, y, w, h]):
                    invalid_lines += 1
                    continue
                boxes.append(
                    {
                        "class_id": class_id,
                        "x_center": x,
                        "y_center": y,
                        "width": w,
                        "height": h,
                        "area": w * h,
                        "aspect_ratio": (w / h) if h > 0 else "",
                        "out_of_bounds": int(x < 0 or x > 1 or y < 0 or y > 1 or w <= 0 or w > 1 or h <= 0 or h > 1),
                    }
                )
            except Exception:
                invalid_lines += 1
                continue
    return boxes, invalid_lines


def analyze_yolo_labels(images_root: str, labels_root: str, out_dir: str, tag: str):
    ensure_dir(out_dir)

    splits = [s for s in ["train", "val", "test"] if os.path.isdir(os.path.join(labels_root, s))]
    if splits:
        label_sets = [(s, os.path.join(labels_root, s), os.path.join(images_root, s)) for s in splits]
    else:
        label_sets = [("all", labels_root, images_root)]

    per_image_rows = []
    per_box_rows = []

    total_labels = 0
    total_boxes = 0
    total_invalid_lines = 0
    total_out_of_bounds = 0
    missing_images = 0

    for split_name, lbl_dir, img_dir in label_sets:
        label_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
        for lp in label_files:
            stem = os.path.splitext(os.path.basename(lp))[0]
            img_path = find_image_for_stem(img_dir, stem)
            total_labels += 1

            img_w = ""
            img_h = ""
            if img_path and img_path.lower().endswith(".png"):
                info = read_png_info(img_path)
                if info:
                    img_w = info.get("width", "")
                    img_h = info.get("height", "")

            if not img_path:
                missing_images += 1

            boxes, invalid_lines = parse_yolo_label_file(lp)
            total_invalid_lines += invalid_lines
            total_boxes += len(boxes)
            oob = sum(int(b.get("out_of_bounds", 0)) for b in boxes)
            total_out_of_bounds += oob

            per_image_rows.append(
                {
                    "split": split_name,
                    "label_file": os.path.basename(lp),
                    "image_file": os.path.basename(img_path) if img_path else "",
                    "image_width": img_w,
                    "image_height": img_h,
                    "boxes": len(boxes),
                    "invalid_lines": invalid_lines,
                    "out_of_bounds_boxes": oob,
                }
            )

            for b in boxes:
                per_box_rows.append(
                    {
                        "split": split_name,
                        "image_stem": stem,
                        "class_id": b["class_id"],
                        "x_center": b["x_center"],
                        "y_center": b["y_center"],
                        "width": b["width"],
                        "height": b["height"],
                        "area": b["area"],
                        "aspect_ratio": b["aspect_ratio"],
                        "out_of_bounds": b["out_of_bounds"],
                        "image_width": img_w,
                        "image_height": img_h,
                    }
                )

    write_csv(os.path.join(out_dir, f"yolo_labels_{tag}_per_image.csv"), per_image_rows)
    write_csv(os.path.join(out_dir, f"yolo_labels_{tag}_per_box.csv"), per_box_rows)

    return {
        "labels_root": labels_root,
        "images_root": images_root,
        "label_files": total_labels,
        "boxes": total_boxes,
        "invalid_lines": total_invalid_lines,
        "out_of_bounds_boxes": total_out_of_bounds,
        "missing_images": missing_images,
    }


def analyze_localization(out_dir: str, max_files=None) -> None:
    print("--- ANALISIS DATASET LOKALISASI (RAW) ---")
    print(f"Raw Dataset Directory: {RAW_DATASET_DIR}")
    print(f"RGB Images Directory: {RAW_RGB_DIR}")

    if not os.path.exists(RAW_LOCALIZATION_DIR):
        print("Folder ffb-localization tidak ditemukan!")
        return

    ensure_dir(out_dir)

    integrity = analyze_localization_integrity(out_dir)
    print(f"Jumlah Gambar RGB: {integrity['rgb_files']}")
    print(f"Jumlah Depth Maps: {integrity['depth_files']}")
    print(f"Jumlah Point Clouds: {integrity['point_cloud_files']}")
    print(f"Jumlah Baris Metadata: {integrity['metadata_rows']}")
    print(f"Missing (RGB/Depth/PC/Metadata): {integrity['missing_rgb']}/{integrity['missing_depth']}/{integrity['missing_point_cloud']}/{integrity['missing_metadata']}")

    rgb_files = sorted(glob.glob(os.path.join(RAW_RGB_DIR, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(RAW_DEPTH_DIR, "*.png")))

    rgb_png_stats = analyze_png_files(rgb_files, os.path.join(out_dir, "localization_rgb_png_info.csv"), max_files=max_files)
    depth_png_stats = analyze_png_files(depth_files, os.path.join(out_dir, "localization_depth_png_info.csv"), max_files=max_files)

    print(f"RGB PNG info: analyzed {rgb_png_stats['analyzed']}/{rgb_png_stats['files']}")
    print(f"Depth PNG info: analyzed {depth_png_stats['analyzed']}/{depth_png_stats['files']}")

    depth_stats = try_depth_stats(depth_files, max_files=100)
    if depth_stats:
        print(f"Depth value stats (sample): dtype={depth_stats['dtype_counts']} min={depth_stats['min'].get('median','')} max={depth_stats['max'].get('median','')}")

    print("\n--- ANALISIS LABEL YOLO (JIKA ADA) ---")
    label_roots = find_label_roots()
    if not label_roots:
        print("Belum ada folder label YOLO yang terdeteksi.")
    else:
        for name, lbl_root in label_roots:
            if name == "processed_labels" and os.path.isdir(PROCESSED_IMAGES_DIR):
                images_root = PROCESSED_IMAGES_DIR
            else:
                images_root = RAW_RGB_DIR
            try:
                summary = analyze_yolo_labels(images_root, lbl_root, out_dir, name)
                if rgb_files:
                    print(
                        f"{name}: labels={summary['label_files']} boxes={summary['boxes']} invalid_lines={summary['invalid_lines']} oob_boxes={summary['out_of_bounds_boxes']} missing_images={summary['missing_images']}"
                    )
            except Exception as e:
                print(f"Gagal analisis label di {lbl_root}: {e}")


def analyze_ripeness(out_dir: str) -> None:
    print("\n--- ANALISIS DATASET KLASIFIKASI KEMATANGAN ---")
    print(f"Ripeness Directory: {RIPENESS_DIR}")

    coco_path = os.path.join(RIPENESS_DIR, "_annotations.coco.json")
    if not os.path.exists(coco_path):
        print("Annotation file _annotations.coco.json missing!")
        return

    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", []) or []
    categories = data.get("categories", []) or []
    annotations = data.get("annotations", []) or []

    cat_name_by_id = {int(c["id"]): str(c.get("name", "")) for c in categories if "id" in c}

    ann_counts = {}
    images_per_cat = {}
    for ann in annotations:
        try:
            cat_id = int(ann.get("category_id"))
            img_id = int(ann.get("image_id"))
        except Exception:
            continue
        ann_counts[cat_id] = ann_counts.get(cat_id, 0) + 1
        if cat_id not in images_per_cat:
            images_per_cat[cat_id] = set()
        images_per_cat[cat_id].add(img_id)

    print(f"Jumlah Gambar (Metadata): {len(images)}")
    print("Kategori:")
    for c in categories:
        try:
            cid = int(c["id"])
            print(f"  - ID {cid}: {c.get('name','')} (Parent: {c.get('supercategory','')})")
        except Exception:
            continue

    rows = []
    for cid, name in sorted(cat_name_by_id.items(), key=lambda x: x[0]):
        rows.append(
            {
                "category_id": cid,
                "category_name": name,
                "annotations": ann_counts.get(cid, 0),
                "images_with_annotations": len(images_per_cat.get(cid, set())),
            }
        )
    write_csv(os.path.join(out_dir, "ripeness_coco_category_summary.csv"), rows, fieldnames=["category_id", "category_name", "annotations", "images_with_annotations"])

    total_ann = sum(ann_counts.values())
    print("Distribusi Anotasi (COCO):")
    for r in rows:
        print(f"  {r['category_name'] or r['category_id']}: {r['annotations']}")
    print(f"  Total: {total_ann}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--skip-ripeness", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    print(f"Script Directory: {SCRIPT_DIR}")
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Raw Dataset Directory: {RAW_DATASET_DIR}")
    print(f"Raw Dataset Exists: {os.path.exists(RAW_DATASET_DIR)}")
    print(f"Output Directory: {out_dir}")
    print("-" * 60)

    max_files = args.max_files if args.max_files and args.max_files > 0 else None
    analyze_localization(out_dir, max_files=max_files)
    if not args.skip_ripeness:
        analyze_ripeness(out_dir)

    print("\nCheck Selesai.")


if __name__ == "__main__":
    main()
