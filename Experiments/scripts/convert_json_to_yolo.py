"""
Convert AnyLabeling JSON (LabelMe format) to YOLO format (.txt)
Converts all .json files in labels directory to .txt format
"""

import json
import os
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).resolve().parents[2]
LABELING_DIR = PROJECT_DIR / "Experiments" / "labeling" / "ffb_localization"
JSON_DIR = LABELING_DIR / "json"
YOLO_OUT_DIR = LABELING_DIR / "yolo_all"
CLASS_MAPPING = {"fresh_fruit_bunch": 0}  # Single class, class_id = 0

def convert_rectangle_to_yolo(points, img_width, img_height):
    """
    Convert rectangle points [x1, y1], [x2, y2] to YOLO format
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    """
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    # Ensure x1 < x2 and y1 < y2
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    
    # Calculate center and dimensions (normalized)
    x_center = ((x_min + x_max) / 2.0) / img_width
    y_center = ((y_min + y_max) / 2.0) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Clamp values to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height

def convert_json_to_yolo(json_path, output_dir):
    """Convert single JSON file to YOLO format"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_width = data.get('imageWidth', 1280)
    img_height = data.get('imageHeight', 720)
    shapes = data.get('shapes', [])
    
    # Generate YOLO format lines
    yolo_lines = []
    for shape in shapes:
        if shape.get('shape_type') != 'rectangle':
            continue  # Skip non-rectangle shapes
        
        label = shape.get('label', '')
        if label not in CLASS_MAPPING:
            print(f"Warning: Unknown label '{label}' in {json_path}, skipping")
            continue
        
        class_id = CLASS_MAPPING[label]
        points = shape.get('points', [])
        
        if len(points) != 2:
            print(f"Warning: Invalid points in {json_path}, skipping")
            continue
        
        x_center, y_center, width, height = convert_rectangle_to_yolo(
            points, img_width, img_height
        )
        
        # YOLO format: class_id x_center y_center width height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Write to .txt file
    json_filename = os.path.basename(json_path)
    txt_filename = json_filename.replace('.json', '.txt')
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(yolo_lines)
    
    return len(yolo_lines)

def main():
    """Convert all JSON files to YOLO format"""
    labels_dir = Path(JSON_DIR)
    output_dir = Path(YOLO_OUT_DIR)
    
    if not labels_dir.exists():
        print(f"Error: JSON labels directory not found: {labels_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(labels_dir.glob('*.json'))
    total_files = len(json_files)
    
    if total_files == 0:
        print(f"No JSON files found in {labels_dir}")
        return
    
    print(f"Found {total_files} JSON files")
    print(f"Converting to YOLO format...")
    print(f"Output directory: {output_dir}\n")
    
    converted = 0
    total_objects = 0
    empty_files = 0
    
    for json_file in json_files:
        try:
            num_objects = convert_json_to_yolo(str(json_file), str(output_dir))
            converted += 1
            total_objects += num_objects
            
            if num_objects == 0:
                empty_files += 1
                print(f"  {json_file.name}: No objects (empty or skipped)")
            else:
                print(f"  {json_file.name}: {num_objects} objects")
                
        except Exception as e:
            print(f"  Error converting {json_file.name}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Conversion Summary:")
    print(f"  Total JSON files: {total_files}")
    print(f"  Successfully converted: {converted}")
    print(f"  Empty files (no objects): {empty_files}")
    print(f"  Total objects: {total_objects}")
    print(f"  Average objects per image: {total_objects/converted:.2f}" if converted > 0 else "")
    print(f"{'='*50}")
    print(f"\nYOLO format files (.txt) saved to: {output_dir}")

if __name__ == "__main__":
    main()
