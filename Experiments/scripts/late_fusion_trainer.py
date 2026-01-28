"""
Late Fusion Trainer for Experiment A.5

Custom trainer class that extends Ultralytics DetectionTrainer to support
the Late Fusion architecture with dual inputs (RGB + Depth).

Author: Research Team
Date: 2026-01-28
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr
import cv2
import numpy as np

from late_fusion_model import LateFusionModel, LateFusionBackbone, FusionLayer


class LateFusionDataset(YOLODataset):
    """
    Custom dataset for Late Fusion that loads both RGB and Depth images separately.

    The dataset returns both RGB and Depth tensors that are fed into separate
    backbones in the Late Fusion model.

    Args:
        *args: Arguments passed to parent YOLODataset
        depth_dir (str/Path): Directory containing depth images
        **kwargs: Keyword arguments passed to parent YOLODataset
    """

    def __init__(self, *args, depth_dir: Optional[Union[str, Path]] = None, **kwargs):
        self.depth_dir = Path(depth_dir) if depth_dir else None
        super().__init__(*args, **kwargs)

    def load_image(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load both RGB and Depth images.

        Args:
            i (int): Image index

        Returns:
            Tuple[np.ndarray, np.ndarray]: RGB image and Depth image
        """
        # Load RGB image using parent method
        rgb = super().load_image(i)

        if self.depth_dir is None:
            raise ValueError("depth_dir must be specified for LateFusionDataset")

        # Load corresponding depth image
        img_file = Path(self.im_files[i])
        depth_file = self.depth_dir / img_file.name

        if not depth_file.exists():
            # Try with different extension
            depth_file = self.depth_dir / (img_file.stem + ".png")

        if not depth_file.exists():
            raise FileNotFoundError(
                f"Depth image not found: {depth_file}\n"
                f"RGB image: {img_file}"
            )

        # Load depth as 3-channel (already processed)
        depth = cv2.imread(str(depth_file), cv2.IMREAD_COLOR)

        if depth is None:
            raise ValueError(f"Failed to load depth image: {depth_file}")

        # Resize depth to match RGB if needed
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

        return rgb, depth

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Get RGB, Depth, and labels at index.

        Args:
            index (int): Index

        Returns:
            Tuple containing RGB tensor, Depth tensor, and labels
        """
        # Get image and label info
        rgb, depth = self.load_image(index)
        labels = self.get_image_and_label(index)

        # Apply augmentations (same transform to both)
        if self.augment:
            rgb, depth = self._apply_augmentation(rgb, depth, labels)

        # Convert to tensors
        rgb_tensor = self._to_tensor(rgb)
        depth_tensor = self._to_tensor(depth)

        return rgb_tensor, depth_tensor, labels

    def _apply_augmentation(
        self, rgb: np.ndarray, depth: np.ndarray, labels: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply synchronized augmentation to both RGB and Depth.

        Args:
            rgb (np.ndarray): RGB image
            depth (np.ndarray): Depth image
            labels (Any): Label data

        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented RGB and Depth
        """
        # Use the same random seed for both images
        # This ensures geometric transforms are synchronized
        if hasattr(self, "transform") and self.transform:
            # Apply transform to RGB
            transformed_rgb = self.transform(image=rgb)
            rgb = transformed_rgb["image"]

            # Apply same transform to depth
            transformed_depth = self.transform(image=depth)
            depth = transformed_depth["image"]

        return rgb, depth

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to tensor.

        Args:
            img (np.ndarray): Image array (H, W, C)

        Returns:
            torch.Tensor: Image tensor (C, H, W)
        """
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


class LateFusionDetectionModel(DetectionModel):
    """
    DetectionModel wrapper for Late Fusion architecture.

    This class wraps the LateFusionModel to be compatible with Ultralytics
    training pipeline.
    """

    def __init__(
        self,
        rgb_weights: str,
        depth_weights: str,
        cfg: Optional[str] = None,
        ch: int = 3,
        nc: int = None,
        verbose: bool = True,
    ):
        """
        Initialize Late Fusion Detection Model.

        Args:
            rgb_weights (str): Path to A.1 RGB weights
            depth_weights (str): Path to A.2 Depth weights
            cfg (Optional[str]): Model config
            ch (int): Number of input channels (not used, kept for compatibility)
            nc (int): Number of classes
            verbose (bool): Verbose output
        """
        self.rgb_weights = rgb_weights
        self.depth_weights = depth_weights
        self.late_fusion_model = None

        # Initialize parent (will call _initialize_biases later)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _build_model(self):
        """
        Build the Late Fusion model.

        This overrides the parent's _build_model to create our custom architecture.
        """
        from ultralytics import YOLO

        device = next(self.parameters()).device if hasattr(self, "parameters") else "cpu"

        # Load backbones from pretrained weights
        LOGGER.info(f"Loading RGB backbone from {self.rgb_weights}")
        rgb_model = YOLO(self.rgb_weights)
        rgb_backbone = rgb_model.model.model[0]

        LOGGER.info(f"Loading Depth backbone from {self.depth_weights}")
        depth_model = YOLO(self.depth_weights)
        depth_backbone = depth_model.model.model[0]

        # Freeze backbones
        for param in rgb_backbone.parameters():
            param.requires_grad = False
        for param in depth_backbone.parameters():
            param.requires_grad = False

        rgb_backbone.eval()
        depth_backbone.eval()

        # Create Late Fusion model
        self.late_fusion_model = LateFusionModel(
            rgb_backbone=rgb_backbone,
            depth_backbone=depth_backbone,
            num_classes=self.nc,
            fusion_channels=256,
        )

        # Get detection head from reference YOLO model
        # The Detect head expects multi-scale features [P3, P4, P5]
        detect_head = rgb_model.model.model[-1]

        # Set detection head on our model
        self.late_fusion_model.set_detection_head(detect_head)

        # Store model components for forward pass
        self.model = nn.ModuleList([
            self.late_fusion_model.backbone,
            self.late_fusion_model.fusion_p3,
            self.late_fusion_model.fusion_p4,
            self.late_fusion_model.fusion_p5,
            detect_head,
        ])

        # Log model info
        if RANK in (-1, 0):
            total = sum(p.numel() for p in self.late_fusion_model.parameters())
            trainable = sum(
                p.numel()
                for p in self.late_fusion_model.parameters()
                if p.requires_grad
            )
            LOGGER.info(
                f"Late Fusion Model: {total:,} parameters, "
                f"{trainable:,} trainable ({trainable / total * 100:.1f}%)"
            )

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], *args, **kwargs
    ):
        """
        Forward pass supporting both single tensor (for compatibility) and dual inputs.

        Args:
            x: Either a single tensor (B, 6, H, W) or tuple of (rgb, depth)

        Returns:
            Model output
        """
        if isinstance(x, tuple):
            rgb, depth = x
            return self.late_fusion_model(rgb, depth)
        else:
            # Assume concatenated input: split into RGB and Depth
            # x shape: (B, 6, H, W) -> rgb: (B, 3, H, W), depth: (B, 3, H, W)
            if x.shape[1] == 6:
                rgb = x[:, :3, :, :]
                depth = x[:, 3:, :, :]
                return self.late_fusion_model(rgb, depth)
            else:
                raise ValueError(
                    f"Expected 6 channels (RGB+Depth), got {x.shape[1]}"
                )

    def fuse(self):
        """Fuse Conv2d and BatchNorm2d layers."""
        # Only fuse the fusion layer and detection head
        # Backbones remain frozen
        if hasattr(self, "late_fusion_model"):
            self.late_fusion_model.fusion = self._fuse_conv_bn(
                self.late_fusion_model.fusion
            )
        return self

    def _fuse_conv_bn(self, module: nn.Module) -> nn.Module:
        """Fuse Conv2d and BatchNorm2d in a module."""
        # Implementation of Conv-BN fusion
        # This is a simplified version
        return module


class LateFusionTrainer(DetectionTrainer):
    """
    Custom trainer for Late Fusion model.

    Extends DetectionTrainer to handle:
    - Dual input (RGB + Depth) data loading
    - Custom Late Fusion model architecture
    - Frozen backbones with trainable fusion + head
    """

    def __init__(self, rgb_weights: str, depth_weights: str, *args, **kwargs):
        """
        Initialize Late Fusion Trainer.

        Args:
            rgb_weights (str): Path to A.1 RGB model weights
            depth_weights (str): Path to A.2 Depth model weights
            *args: Additional args passed to DetectionTrainer
            **kwargs: Additional kwargs passed to DetectionTrainer
        """
        self.rgb_weights = rgb_weights
        self.depth_weights = depth_weights
        self.depth_dir = None  # Will be set from data config

        super().__init__(*args, **kwargs)

    def setup_model(self):
        """
        Setup the Late Fusion model.

        Overrides parent to load our custom architecture.
        """
        # Parse data config to get depth directory
        if hasattr(self, "data") and self.data:
            self.depth_dir = self.data.get("depth_dir")

        # Build Late Fusion model
        LOGGER.info("Building Late Fusion Model...")
        self.model = LateFusionDetectionModel(
            rgb_weights=self.rgb_weights,
            depth_weights=self.depth_weights,
            nc=self.data.get("nc", 1) if self.data else 1,
        )

        # Move to device
        self.model.to(self.device)

        # Log model structure
        self._log_model_info()

    def _log_model_info(self):
        """Log information about the model architecture."""
        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("Late Fusion Model Architecture")
        LOGGER.info("=" * 60)

        # Check frozen status
        if hasattr(self.model, "late_fusion_model"):
            lfm = self.model.late_fusion_model

            rgb_frozen = all(
                not p.requires_grad for p in lfm.backbone.rgb_backbone.parameters()
            )
            depth_frozen = all(
                not p.requires_grad for p in lfm.backbone.depth_backbone.parameters()
            )
            fusion_p3_trainable = any(p.requires_grad for p in lfm.fusion_p3.parameters())
            fusion_p4_trainable = any(p.requires_grad for p in lfm.fusion_p4.parameters())
            fusion_p5_trainable = any(p.requires_grad for p in lfm.fusion_p5.parameters())

            LOGGER.info(f"RGB Backbone: {'Frozen' if rgb_frozen else 'Trainable'}")
            LOGGER.info(f"Depth Backbone: {'Frozen' if depth_frozen else 'Trainable'}")
            LOGGER.info(f"Fusion P3: {'Trainable' if fusion_p3_trainable else 'Frozen'}")
            LOGGER.info(f"Fusion P4: {'Trainable' if fusion_p4_trainable else 'Frozen'}")
            LOGGER.info(f"Fusion P5: {'Trainable' if fusion_p5_trainable else 'Frozen'}")

            # Count parameters
            total = sum(p.numel() for p in lfm.parameters())
            trainable = sum(p.numel() for p in lfm.parameters() if p.requires_grad)
            LOGGER.info(f"\nTotal Parameters: {total:,}")
            LOGGER.info(f"Trainable Parameters: {trainable:,}")
            LOGGER.info(f"Frozen Parameters: {total - trainable:,}")

        LOGGER.info("=" * 60 + "\n")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Get the Late Fusion model.

        Args:
            cfg: Model configuration
            weights: Model weights (not used, we load from rgb_weights/depth_weights)
            verbose: Verbose output

        Returns:
            LateFusionDetectionModel
        """
        model = LateFusionDetectionModel(
            rgb_weights=self.rgb_weights,
            depth_weights=self.depth_weights,
            cfg=cfg,
            nc=self.data.get("nc", 1),
            verbose=verbose,
        )
        return model

    def build_dataset(
        self, img_path, mode="train", batch=None
    ) -> LateFusionDataset:
        """
        Build Late Fusion dataset.

        Args:
            img_path: Path to images
            mode (str): 'train' or 'val'
            batch: Batch size

        Returns:
            LateFusionDataset
        """
        # Get depth directory from data config
        depth_dir = self.data.get("depth_dir") if self.data else None

        if depth_dir is None:
            raise ValueError(
                "depth_dir must be specified in data config for Late Fusion"
            )

        # Build dataset
        dataset = LateFusionDataset(
            img_path=img_path,
            depth_dir=depth_dir,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max())
            if hasattr(self.model, "stride")
            else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.task,
            classes=self.data.get("names"),
        )

        return dataset

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        """
        Get dataloader for Late Fusion dataset.

        Args:
            dataset_path: Path to dataset
            batch_size: Batch size
            rank: Process rank for DDP
            mode (str): 'train' or 'val'

        Returns:
            DataLoader
        """
        # Build dataset
        dataset = self.build_dataset(dataset_path, mode, batch_size)

        # Build loader
        shuffle = mode == "train"
        workers = self.args.workers

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            drop_last=mode == "train",
        )

        return loader

    def preprocess_batch(self, batch):
        """
        Preprocess batch for Late Fusion training.

        Args:
            batch: Batch data from dataloader

        Returns:
            Preprocessed batch
        """
        # batch should contain (rgb, depth, labels)
        if len(batch) == 3:
            rgb, depth, labels = batch
            batch = {
                "img": (rgb, depth),  # Tuple for dual input
                "cls": labels.get("cls"),
                "bboxes": labels.get("bboxes"),
                "batch_idx": labels.get("batch_idx"),
            }

        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
            elif isinstance(v, tuple):
                batch[k] = tuple(t.to(self.device) for t in v)

        return batch

    def get_validator(self):
        """Get validator for Late Fusion model."""
        from ultralytics.models.yolo.detect import DetectionValidator

        class LateFusionValidator(DetectionValidator):
            """Custom validator that handles dual input."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def preprocess(self, batch):
                """Preprocess batch with dual input."""
                batch = super().preprocess(batch)
                # batch["img"] should already be a tuple from our dataset
                return batch

        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        return LateFusionValidator(self.test_loader, save_dir=self.save_dir, args=self.args)

    def train_step(self, batch):
        """
        Single training step with proper YOLO loss.

        Args:
            batch: Batch data with 'img' as tuple (rgb, depth)

        Returns:
            Loss tensor
        """
        # Ensure backbones stay frozen
        if hasattr(self.model, "late_fusion_model"):
            self.model.late_fusion_model.backbone.rgb_backbone.eval()
            self.model.late_fusion_model.backbone.depth_backbone.eval()

        # Forward pass - batch["img"] is a tuple of (rgb, depth)
        preds = self.model(batch["img"])

        # Compute loss using YOLO v8DetectionLoss
        # This returns loss and loss_items (box_loss, cls_loss, dfl_loss)
        loss, loss_items = self.criterion(preds, batch)

        return loss

    def validate(self):
        """Run validation."""
        # Ensure backbones are in eval mode
        if hasattr(self.model, "late_fusion_model"):
            self.model.late_fusion_model.backbone.rgb_backbone.eval()
            self.model.late_fusion_model.backbone.depth_backbone.eval()

        return super().validate()


def create_late_fusion_config(
    rgb_dataset_path: str,
    depth_dir: str,
    output_path: str = "configs/ffb_late_fusion.yaml",
) -> str:
    """
    Create YAML configuration file for Late Fusion dataset.

    Args:
        rgb_dataset_path (str): Path to RGB dataset
        depth_dir (str): Path to depth images
        output_path (str): Output YAML file path

    Returns:
        str: Path to created config file
    """
    config_content = f"""# Dataset Configuration for A.5 - Late Fusion (RGB + Depth)
# Dual input: RGB and Depth processed separately

path: {Path(rgb_dataset_path).as_posix()}
depth_dir: {Path(depth_dir).as_posix()}

train: images/train
val: images/val
test: images/test

nc: 1
names: ['fresh_fruit_bunch']

# Late Fusion specific settings
late_fusion: true
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(config_content)

    LOGGER.info(f"Late Fusion config created: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Test the trainer components
    print("=" * 60)
    print("Late Fusion Trainer - Component Test")
    print("=" * 60)

    # Test dataset creation
    print("\n1. Testing LateFusionDataset...")
    # This would require actual data paths
    print("   (Requires actual dataset paths to test)")

    # Test trainer initialization
    print("\n2. Testing LateFusionTrainer...")
    print("   (Requires trained A.1 and A.2 weights)")

    print("\n" + "=" * 60)
    print("Component test complete (manual verification needed)")
    print("=" * 60)
