import re
import glob
import os

def parse_yolo_log(filepath):
    best_map50_95 = -1.0
    best_map50 = -1.0
    best_line = ""
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().startswith('all'):
                parts = line.split()
                # Expected: all, images, instances, P, R, mAP50, mAP50-95
                if len(parts) >= 7:
                    try:
                        map50 = float(parts[5])
                        map50_95 = float(parts[6])
                        if map50_95 > best_map50_95:
                            best_map50_95 = map50_95
                            best_map50 = map50
                            best_line = line.strip()
                    except ValueError:
                        continue
    return best_map50, best_map50_95, best_line

files = {
    "Gap 1 (Small+SGD+50)": "test_gap1.txt",
    "Gap 2 (Nano+AdamW+50)": "test_gap2.txt",
    "Gap 3 (Nano+SGD+300)": "test_gap3.txt",
    "Gap 4 (Small+SGD+300)": "test_gap4.txt",
    "Old Best (Small+AdamW+300)": "test_Small_300_AdamW.txt",
    "Baseline (Nano+SGD+50)": "test.txt"
}

print(f"{'Experiment':<25} | {'mAP50':<10} | {'mAP50-95':<10}")
print("-" * 50)

base_path = r"d:\Work\Assisten Dosen\Anylabel\Reports\FFB_Ultimate_Report\artifacts\kaggleoutput"

for name, filename in files.items():
    full_path = os.path.join(base_path, filename)
    if os.path.exists(full_path):
        m50, m5095, line = parse_yolo_log(full_path)
        print(f"{name:<25} | {m50:<10.3f} | {m5095:<10.3f}")
    else:
        print(f"{name:<25} | {'N/A':<10} | {'N/A':<10}")
