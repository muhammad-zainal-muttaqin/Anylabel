"""
Custom RGBD Dataset dengan Synchronized Augmentation

RGB: HSV + geometric transforms
Depth: Geometric transforms only (synced dengan RGB)

This custom dataset class extends Ultralytics YOLODataset to handle 4-channel
RGBD input with proper synchronized augmentation between RGB and Depth modalities.

Author: Research Team
Date: 2026-01-21
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset


class RGBDYOLODataset(YOLODataset):
    """
    Custom YOLO dataset for RGBD (4-channel) input with synced augmentation.

    Args:
        *args: Arguments passed to parent YOLODataset
        depth_dir (str/Path): Directory containing depth maps
        use_hsv_aug (bool): Whether to apply HSV augmentation to RGB channel
        **kwargs: Keyword arguments passed to parent YOLODataset
    """

    def __init__(self, *args, depth_dir=None, use_hsv_aug=True, **kwargs):
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.use_hsv_aug = use_hsv_aug
        super().__init__(*args, **kwargs)

        # Define synced geometric augmentation
        # These transforms apply to both RGB and Depth identically
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.Affine(translate_percent=0.1, scale=(0.9, 1.1), p=0.5),
            ],
            additional_targets={"depth": "image"},
        )

        # HSV augmentation only for RGB
        if self.use_hsv_aug:
            self.hsv_transform = A.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015, p=0.5
            )

    def load_image(self, i):
        """
        Load RGB and Depth images, apply synced augmentation, and stack to 4-channel.

        Args:
            i (int): Index of image to load

        Returns:
            np.ndarray: 4-channel RGBD image (H, W, 4)
        """
        # Load RGB from parent class
        rgb = super().load_image(i)

        if self.depth_dir is None:
            # Fallback to RGB-only if depth_dir not specified
            return rgb

        # Load corresponding depth image
        img_file = Path(self.im_files[i])
        depth_file = self.depth_dir / img_file.name

        if not depth_file.exists():
            raise FileNotFoundError(
                f"Depth image not found: {depth_file}\n"
                f"RGB image: {img_file}\n"
                f"Ensure depth filename matches RGB filename."
            )

        depth = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)

        if depth is None:
            raise ValueError(f"Failed to load depth image: {depth_file}")

        # Resize depth to match RGB dimensions
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

        # Apply synced geometric transforms
        if self.augment:
            transformed = self.transform(image=rgb, depth=depth)
            rgb = transformed["image"]
            depth = transformed["depth"]

            # Apply HSV augmentation only to RGB
            if self.use_hsv_aug:
                rgb = self.hsv_transform(image=rgb)["image"]

        # Normalize depth to 0-1 range
        depth_norm = depth.astype(np.float32) / 255.0

        # Stack RGB and Depth to create 4-channel image [R, G, B, D]
        # Keep as uint8 for compatibility with YOLO pipeline
        rgbd = np.dstack([rgb, (depth_norm * 255).astype(np.uint8)])

        return rgbd

    def __getitem__(self, index):
        """
        Get RGBD image and labels at index.

        Args:
            index (int): Index

        Returns:
            tuple: (rgbd_image, labels)
        """
        return self.get_image_and_label(index)


def test_rgbd_dataset():
    """
    Test function to verify RGBD dataset loading and augmentation.
    Run this to validate the custom dataset before training.
    """
    from pathlib import Path

    PROJECT_DIR = Path(__file__).resolve().parents[2]

    # Test paths
    rgb_dir = (
        PROJECT_DIR
        / "Experiments"
        / "datasets"
        / "ffb_localization"
        / "images"
        / "train"
    )
    depth_dir = PROJECT_DIR / "Experiments" / "datasets" / "depth_processed_rgb"
    labels_dir = (
        PROJECT_DIR
        / "Experiments"
        / "datasets"
        / "ffb_localization"
        / "labels"
        / "train"
    )

    if not rgb_dir.exists():
        print(f"RGB directory not found: {rgb_dir}")
        return False

    if not depth_dir.exists():
        print(f"Depth directory not found: {depth_dir}")
        return False

    print("Testing RGBD Dataset...")
    print(f"RGB dir: {rgb_dir}")
    print(f"Depth dir: {depth_dir}")

    try:
        # Create dataset instance
        dataset = RGBDYOLODataset(
            img_path=str(rgb_dir), depth_dir=str(depth_dir), use_hsv_aug=True
        )

        # Load first image
        rgbd = dataset.load_image(0)

        print(f"\nRGBD shape: {rgbd.shape}")
        print(f"RGBD dtype: {rgbd.dtype}")
        print(f"RGBD value range: [{rgbd.min()}, {rgbd.max()}]")

        # Verify 4-channel
        if rgbd.shape[2] == 4:
            print("✓ 4-channel RGBD image loaded successfully")
            return True
        else:
            print(f"✗ Expected 4 channels, got {rgbd.shape[2]}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    # Run test
    success = test_rgbd_dataset()
    if success:
        print("\n✓ RGBD Dataset test passed!")
    else:
        print("\n✗ RGBD Dataset test failed!")
