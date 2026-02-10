"""
Verify labeling consistency: check images vs labels count
Portable version for LabelingKit

Usage:
    python scripts/verify_labels.py
    python scripts/verify_labels.py --images workspace/images --labels workspace/labels
"""

import argparse
from pathlib import Path
from collections import Counter


def get_image_extensions():
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def main():
    parser = argparse.ArgumentParser(description='Verify images and labels consistency')
    parser.add_argument('--images', '-i', type=str, default='workspace/images',
                        help='Directory containing images')
    parser.add_argument('--labels', '-l', type=str, default='workspace/labels',
                        help='Directory containing label files (JSON or TXT)')
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent.parent
    images_dir = script_dir / args.images
    labels_dir = script_dir / args.labels
    
    print("=" * 60)
    print("  Label Verification Tool")
    print("=" * 60)
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print()
    
    # Check directories exist
    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"[WARNING] Labels directory not found: {labels_dir}")
        print("          No labels to verify yet.")
        return
    
    # Get image files
    image_extensions = get_image_extensions()
    image_files = set()
    for ext in image_extensions:
        image_files.update(f.stem for f in images_dir.glob(f'*{ext}'))
        image_files.update(f.stem for f in images_dir.glob(f'*{ext.upper()}'))
    
    # Get label files (both JSON and TXT)
    json_labels = set(f.stem for f in labels_dir.glob('*.json'))
    txt_labels = set(f.stem for f in labels_dir.glob('*.txt'))
    label_files = json_labels | txt_labels
    
    print(f"Images found: {len(image_files)}")
    print(f"Labels found: {len(label_files)} (JSON: {len(json_labels)}, TXT: {len(txt_labels)})")
    print()
    
    # Find mismatches
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files
    matched = image_files & label_files
    
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Matched pairs:          {len(matched)}")
    print(f"  Images without labels:  {len(images_without_labels)}")
    print(f"  Labels without images:  {len(labels_without_images)}")
    print()
    
    # Show details if there are mismatches
    if images_without_labels:
        print("Images without labels (need labeling):")
        for name in sorted(list(images_without_labels)[:10]):
            print(f"  - {name}")
        if len(images_without_labels) > 10:
            print(f"  ... and {len(images_without_labels) - 10} more")
        print()
    
    if labels_without_images:
        print("Labels without images (orphaned):")
        for name in sorted(list(labels_without_images)[:10]):
            print(f"  - {name}")
        if len(labels_without_images) > 10:
            print(f"  ... and {len(labels_without_images) - 10} more")
        print()
    
    # Calculate progress
    if image_files:
        progress = len(matched) / len(image_files) * 100
        print(f"Labeling progress: {progress:.1f}% ({len(matched)}/{len(image_files)})")
        
        # Progress bar
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"[{bar}]")
    
    print()


if __name__ == "__main__":
    main()
