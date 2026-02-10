"""
Convert AnyLabeling JSON (LabelMe format) to YOLO format (.txt)
Portable version for LabelingKit

Usage:
    python scripts/convert_json_to_yolo.py
    python scripts/convert_json_to_yolo.py --input workspace/labels --output output/yolo
"""

import json
import os
import argparse
from pathlib import Path


# Default class mapping - modify as needed
CLASS_MAPPING = {
    "fresh_fruit_bunch": 0,
    # Add more classes here if needed:
    # "ripe": 0,
    # "unripe": 1,
}


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


def convert_json_to_yolo(json_path, output_dir, class_mapping):
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
        if label not in class_mapping:
            print(f"  Warning: Unknown label '{label}' in {json_path.name}, skipping")
            continue
        
        class_id = class_mapping[label]
        points = shape.get('points', [])
        
        if len(points) != 2:
            print(f"  Warning: Invalid points in {json_path.name}, skipping")
            continue
        
        x_center, y_center, width, height = convert_rectangle_to_yolo(
            points, img_width, img_height
        )
        
        # YOLO format: class_id x_center y_center width height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Write to .txt file
    txt_filename = json_path.stem + '.txt'
    txt_path = output_dir / txt_filename
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(yolo_lines)
    
    return len(yolo_lines)


def main():
    parser = argparse.ArgumentParser(description='Convert AnyLabeling JSON to YOLO format')
    parser.add_argument('--input', '-i', type=str, default='workspace/labels',
                        help='Input directory containing JSON files')
    parser.add_argument('--output', '-o', type=str, default='output/yolo',
                        help='Output directory for YOLO txt files')
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent.parent
    labels_dir = script_dir / args.input
    output_dir = script_dir / args.output
    
    print("=" * 60)
    print("  JSON to YOLO Converter")
    print("=" * 60)
    print(f"Input:  {labels_dir}")
    print(f"Output: {output_dir}")
    print(f"Classes: {CLASS_MAPPING}")
    print()
    
    if not labels_dir.exists():
        print(f"[ERROR] Input directory not found: {labels_dir}")
        print("        Place your JSON label files in workspace/labels/")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(labels_dir.glob('*.json'))
    total_files = len(json_files)
    
    if total_files == 0:
        print(f"[ERROR] No JSON files found in {labels_dir}")
        return
    
    print(f"Found {total_files} JSON files")
    print("Converting...")
    print()
    
    converted = 0
    total_objects = 0
    empty_files = 0
    
    for json_file in json_files:
        try:
            num_objects = convert_json_to_yolo(json_file, output_dir, CLASS_MAPPING)
            converted += 1
            total_objects += num_objects
            
            if num_objects == 0:
                empty_files += 1
                
        except Exception as e:
            print(f"  [ERROR] {json_file.name}: {e}")
    
    print()
    print("=" * 60)
    print("  Conversion Summary")
    print("=" * 60)
    print(f"  Total JSON files:      {total_files}")
    print(f"  Successfully converted: {converted}")
    print(f"  Empty files:           {empty_files}")
    print(f"  Total objects:         {total_objects}")
    if converted > 0:
        print(f"  Avg objects/image:     {total_objects/converted:.2f}")
    print()
    print(f"YOLO files saved to: {output_dir}")


if __name__ == "__main__":
    main()
