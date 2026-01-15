"""
Analisis Kegagalan - Identifikasi False Positive, False Negative
Visualisasi hasil prediksi vs ground truth
"""

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
EXPERIMENTS_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments"
DATASET_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments\datasets\ffb_localization"
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "failure_analysis")
MODEL_PATH = 'runs/detect/exp_a1_rgb_seed_42/weights/best.pt'

def create_failure_analysis_dirs():
    """Create directories for saving analysis results"""
    
    dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "false_positives"),
        os.path.join(OUTPUT_DIR, "false_negatives"),
        os.path.join(OUTPUT_DIR, "correct_detections"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print(f"Output directory: {OUTPUT_DIR}")

def load_model():
    """Load trained model"""
    
    full_model_path = os.path.join(EXPERIMENTS_DIR, MODEL_PATH)
    
    if not os.path.exists(full_model_path):
        print(f"Model tidak ditemukan: {full_model_path}")
        print("   Pastikan training sudah selesai!")
        return None
    
    print(f"Loading model: {MODEL_PATH}")
    return YOLO(full_model_path)

def get_test_images():
    """Get test images"""
    
    test_images_dir = os.path.join(DATASET_DIR, "images", "test")
    test_images = list(Path(test_images_dir).glob("*.png"))
    
    print(f"Found {len(test_images)} test images")
    return test_images

def draw_boxes(image, detections, ground_truth, color, conf_threshold=0.25):
    """Draw detections and ground truth on image"""
    
    img = image.copy()
    
    # Draw predictions
    for det in detections:
        conf = float(det.confidence)
        if conf < conf_threshold:
            continue
        
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw ground truth
    for gt in ground_truth:
        x1, y1, x2, y2 = map(int, gt[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, "GT", (x1, y2+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return img

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def read_ground_truth(label_path):
    """Read ground truth from YOLO label file"""
    
    if not os.path.exists(label_path):
        return []
    
    gts = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                
                # Convert to xyxy
                x1 = int((x_c - w/2) * 1000)  # Scale for comparison
                y1 = int((y_c - h/2) * 1000)
                x2 = int((x_c + w/2) * 1000)
                y2 = int((y_c + h/2) * 1000)
                
                gts.append([x1, y1, x2, y2, class_id])
    
    return gts

def analyze_failures(model, test_images):
    """Analyze failures"""
    
    results_summary = {
        'correct': 0,
        'false_positive': 0,
        'false_negative': 0,
        'total_detections': 0,
        'total_ground_truth': 0
    }
    
    failure_cases = []
    
    for img_path in test_images[:50]:  # Limit for speed
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Get ground truth
        label_path = os.path.join(DATASET_DIR, "labels", "test", 
                                 img_path.stem + ".txt")
        gts = read_ground_truth(label_path)
        
        if not gts:
            continue
        
        results_summary['total_ground_truth'] += len(gts)
        
        # Predict
        results = model(img)
        detections = results[0].boxes
        
        # Get detections with confidence > 0.25
        dets = []
        for det in detections:
            if float(det.confidence) >= 0.25:
                dets.append(det)
        
        results_summary['total_detections'] += len(dets)
        
        # Match detections to ground truth
        matched_gt = set()
        matched_det = set()
        
        for i, det in enumerate(dets):
            det_box = det.xyxy[0].cpu().numpy()
            det_box_scaled = [int(x) for x in det_box]
            
            for j, gt in enumerate(gts):
                iou = calculate_iou(det_box_scaled[:4], gt[:4])
                
                if iou >= 0.5:  # IoU threshold
                    matched_gt.add(j)
                    matched_det.add(i)
                    results_summary['correct'] += 1
        
        # False Positives (detections not matched to any GT)
        fp_count = len(dets) - len(matched_det)
        results_summary['false_positive'] += fp_count
        
        # False Negatives (GT not matched to any detection)
        fn_count = len(gts) - len(matched_gt)
        results_summary['false_negative'] += fn_count
        
        # Save failure cases
        if fp_count > 0 or fn_count > 0:
            case = {
                'image': img_path.name,
                'false_positives': fp_count,
                'false_negatives': fn_count,
                'detections': len(dets),
                'ground_truth': len(gts),
                'img_shape': img.shape
            }
            failure_cases.append(case)
            
            # Visualize
            vis_img = draw_boxes(img, dets, gts, (0, 0, 255))
            
            if fp_count > 0 and fn_count > 0:
                filename = f"FP_{fp_count}_FN_{fn_count}_{img_path.name}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "false_positives", filename), vis_img)
            elif fp_count > 0:
                filename = f"FP_{fp_count}_{img_path.name}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "false_positives", filename), vis_img)
            elif fn_count > 0:
                filename = f"FN_{fn_count}_{img_path.name}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "false_negatives", filename), vis_img)
            else:
                filename = f"CORRECT_{img_path.name}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "correct_detections", filename), vis_img)
    
    return results_summary, failure_cases

def generate_failure_report(summary, failure_cases):
    """Generate failure analysis report"""
    
    report = "# ANALISIS KEGAGALAN DETEKSI TBS\n\n"
    
    report += "## Ringkasan\n\n"
    report += f"- **Total Ground Truth**: {summary['total_ground_truth']}\n"
    report += f"- **Total Detections**: {summary['total_detections']}\n"
    report += f"- **Correct Detections**: {summary['correct']}\n"
    report += f"- **False Positives**: {summary['false_positive']}\n"
    report += f"- **False Negatives**: {summary['false_negative']}\n"
    
    if summary['total_ground_truth'] > 0:
        fp_rate = (summary['false_positive'] / summary['total_detections']) * 100 if summary['total_detections'] > 0 else 0
        fn_rate = (summary['false_negative'] / summary['total_ground_truth']) * 100
        
        report += f"- **False Positive Rate**: {fp_rate:.2f}%\n"
        report += f"- **False Negative Rate**: {fn_rate:.2f}%\n"
    
    report += "\n## Kasus Kegagalan\n\n"
    
    if failure_cases:
        report += "| Gambar | FP | FN | Deteksi | GT |\n"
        report += "|--------|----|----|---------|----|\n"
        
        for case in failure_cases[:20]:  # Limit to 20 cases
            report += f"| {case['image']} | {case['false_positives']} | {case['false_negatives']} | {case['detections']} | {case['ground_truth']} |\n"
    else:
        report += "Tidak ada kasus kegagalan ditemukan!\n"
    
    report += "\n## Visualisasi\n\n"
    report += "Hasil visualisasi disimpan di:\n"
    report += f"- `{OUTPUT_DIR}/false_positives/` - Deteksi salah\n"
    report += f"- `{OUTPUT_DIR}/false_negatives/` - TBS tidak terdeteksi\n"
    report += f"- `{OUTPUT_DIR}/correct_detections/` - Deteksi benar\n"
    
    report += "\n## Analisis Kegagalan Umum\n\n"
    report += "### Penyebab False Positives:\n"
    report += "- Bayangan daun menyerupai TBS\n"
    report += "- Pecahan cahaya/metalik\n"
    report += "- Objek dengan bentuk mirip\n"
    report += "- Background yang kompleks\n\n"
    
    report += "### Penyebab False Negatives:\n"
    report += "- TBS terhalang daun\n"
    report += "- TBS terlalu kecil (jauh)\n"
    report += "- Pencahayaan buruk\n"
    report += "- TBS pada sudut sulit\n"
    report += "- TBS terlalu terang/overexposed\n"
    
    return report

def main():
    print("Analisis Kegagalan Deteksi")
    print("="*60)
    
    # Create output directories
    create_failure_analysis_dirs()
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Get test images
    test_images = get_test_images()
    if not test_images:
        print("Tidak ada gambar test ditemukan!")
        return
    
    # Analyze
    print("\nMulai analisis...")
    summary, failure_cases = analyze_failures(model, test_images)
    
    # Generate report
    report = generate_failure_report(summary, failure_cases)
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "FAILURE_ANALYSIS.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save summary as CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "failure_summary.csv"), index=False)
    
    # Save failure cases as CSV
    if failure_cases:
        cases_df = pd.DataFrame(failure_cases)
        cases_df.to_csv(os.path.join(OUTPUT_DIR, "failure_cases.csv"), index=False)
    
    print("\n" + "="*60)
    print("ANALISIS SELESAI")
    print("="*60)
    print(f"Hasil: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print(f"Summary: failure_summary.csv")
    print(f"Cases: failure_cases.csv")
    print(f"Images: false_positives/, false_negatives/, correct_detections/")

if __name__ == "__main__":
    main()
