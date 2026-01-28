"""
Late Fusion Model for Experiment A.5

Architecture:
    Input RGB (3ch)          Input Depth (3ch)
         |                         |
         ▼                         ▼
    ┌─────────────┐           ┌─────────────┐
    │  RGB Branch │           │ Depth Branch│
    │  (Frozen)   │           │  (Frozen)   │
    │  From A.1   │           │  From A.2   │
    │  Weights    │           │  Weights    │
    │  Output:    │           │  Output:    │
    │  P3 feature │           │  P3 feature │
    │  (256 ch)   │           │  (256 ch)   │
    └──────┬──────┘           └──────┬──────┘
           │                         │
           └───────────┬─────────────┘
                       ▼
              ┌─────────────────┐
              │   Concatenate   │  512 channel
              │  (RGB || Depth) │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  1x1 Conv       │  256 channel
              │  BatchNorm      │
              │  SiLU           │
              │  (trainable)    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Detection Head │
              │  (trainable)    │
              │  from YOLOv11n  │
              └─────────────────┘

Author: Research Team
Date: 2026-01-28
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Detect
from ultralytics.nn.tasks import DetectionModel


class LateFusionBackbone(nn.Module):
    """
    Dual backbone for Late Fusion architecture.
    Combines frozen RGB and frozen Depth backbones, extracting multi-scale features (P3, P4, P5).

    Args:
        rgb_backbone (nn.Module): Frozen RGB backbone from A.1
        depth_backbone (nn.Module): Frozen Depth backbone from A.2
        extract_layers (list): Layer indices to extract features from (default: [4, 6, 8] for P3, P4, P5)
    """

    def __init__(
        self,
        rgb_backbone: nn.Module,
        depth_backbone: nn.Module,
        extract_layers: list = None,
    ):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        # Default: extract P3 (layer 4), P4 (layer 6), P5 (layer 8)
        self.extract_layers = extract_layers or [4, 6, 8]

        # Freeze both backbones
        self._freeze_backbone(self.rgb_backbone)
        self._freeze_backbone(self.depth_backbone)

        # Get output channels from one backbone (both should be same)
        self.out_channels = self._get_out_channels(rgb_backbone, self.extract_layers[-1])

    def _freeze_backbone(self, backbone: nn.Module):
        """Freeze all parameters in the backbone."""
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

    def _get_out_channels(self, backbone: nn.Module, layer_idx: int) -> int:
        """Infer output channels from backbone architecture."""
        # For YOLOv11n, P3 features have 256 channels
        # This is typically at layer index 4 (after first C3k2 block)
        return 256

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through both backbones and concatenate multi-scale features.

        Args:
            rgb (torch.Tensor): RGB input tensor (B, 3, H, W)
            depth (torch.Tensor): Depth input tensor (B, 3, H, W)

        Returns:
            List[torch.Tensor]: List of concatenated features at P3, P4, P5
                Each tensor has shape (B, 512, H/s, W/s) where s is stride
        """
        # Extract multi-scale features from RGB backbone
        rgb_features = self._extract_multi_scale_features(self.rgb_backbone, rgb)

        # Extract multi-scale features from Depth backbone
        depth_features = self._extract_multi_scale_features(self.depth_backbone, depth)

        # Concatenate each scale
        fused_features = []
        for rgb_feat, depth_feat in zip(rgb_features, depth_features):
            concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)  # (B, 512, H/s, W/s)
            fused_features.append(concat_feat)

        return fused_features  # [P3_concat, P4_concat, P5_concat]

    def _extract_multi_scale_features(
        self, backbone: nn.Module, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Extract multi-scale features (P3, P4, P5) from backbone.

        For YOLOv11n:
        - P3: stride 8, 128 channels (layer 4)
        - P4: stride 16, 256 channels (layer 6)
        - P5: stride 32, 256 channels (layer 8)

        Args:
            backbone (nn.Module): Backbone network
            x (torch.Tensor): Input tensor

        Returns:
            List[torch.Tensor]: List of features at P3, P4, P5
        """
        features = []
        with torch.no_grad():
            for i, module in enumerate(backbone):
                x = module(x)
                if i in self.extract_layers:
                    features.append(x)
                    if len(features) == len(self.extract_layers):
                        break

        return features  # [P3, P4, P5]


class FusionLayer(nn.Module):
    """
    Fusion layer that reduces concatenated features from 512 to 256 channels.

    Architecture:
        1x1 Conv -> BatchNorm -> SiLU

    Args:
        in_channels (int): Input channels (512 for concatenated RGB+Depth)
        out_channels (int): Output channels (256)
    """

    def __init__(self, in_channels: int = 512, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Swish activation used in YOLO

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion layer.

        Args:
            x (torch.Tensor): Concatenated features (B, 512, H, W)

        Returns:
            torch.Tensor: Fused features (B, 256, H, W)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LateFusionModel(nn.Module):
    """
    Complete Late Fusion Model for FFB Detection.

    Combines frozen RGB and Depth backbones with trainable fusion layers
    and detection head. Uses multi-scale features (P3, P4, P5) for proper
    YOLO detection head compatibility.

    Args:
        rgb_backbone (nn.Module): RGB backbone from A.1
        depth_backbone (nn.Module): Depth backbone from A.2
        num_classes (int): Number of detection classes (default: 1)
        fusion_channels (int): Output channels from fusion layer (default: 256)
    """

    def __init__(
        self,
        rgb_backbone: nn.Module,
        depth_backbone: nn.Module,
        num_classes: int = 1,
        fusion_channels: int = 256,
    ):
        super().__init__()

        # Dual frozen backbones
        self.backbone = LateFusionBackbone(rgb_backbone, depth_backbone)

        # Fusion layers for each scale (P3, P4, P5)
        # P3: 128 (RGB) + 128 (Depth) = 256 -> 128
        # P4: 256 (RGB) + 256 (Depth) = 512 -> 256
        # P5: 256 (RGB) + 256 (Depth) = 512 -> 256
        self.fusion_p3 = FusionLayer(in_channels=256, out_channels=128)
        self.fusion_p4 = FusionLayer(in_channels=512, out_channels=256)
        self.fusion_p5 = FusionLayer(in_channels=512, out_channels=256)

        self.num_classes = num_classes

        # Detection head (trainable) - will be set externally
        self.detect = None

    def set_detection_head(self, detect_head: nn.Module):
        """
        Set the detection head (called after model initialization).

        Args:
            detect_head (nn.Module): Detection head from YOLO
        """
        self.detect = detect_head

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the complete model.

        Args:
            rgb (torch.Tensor): RGB input (B, 3, H, W)
            depth (torch.Tensor): Depth input (B, 3, H, W)

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Detection outputs
        """
        # Extract multi-scale features and concatenate
        # Returns [P3_concat(512ch), P4_concat(512ch), P5_concat(512ch)]
        concat_features = self.backbone(rgb, depth)

        # Fuse each scale with appropriate fusion layer
        # P3: 256ch -> 128ch, P4: 512ch -> 256ch, P5: 512ch -> 256ch
        fused_p3 = self.fusion_p3(concat_features[0])  # (B, 128, H/8, W/8)
        fused_p4 = self.fusion_p4(concat_features[1])  # (B, 256, H/16, W/16)
        fused_p5 = self.fusion_p5(concat_features[2])  # (B, 256, H/32, W/32)

        fused_features = [fused_p3, fused_p4, fused_p5]

        # Detection head
        if self.detect is not None:
            output = self.detect(fused_features)
            return output

        return fused_features

    def train(self, mode: bool = True):
        """
        Set training mode, ensuring backbones stay in eval mode.

        Args:
            mode (bool): Training mode flag
        """
        super().train(mode)
        # Always keep backbones in eval mode (frozen)
        if hasattr(self, 'backbone'):
            self.backbone.rgb_backbone.eval()
            self.backbone.depth_backbone.eval()
        return self


def load_frozen_backbone(weights_path: str, device: torch.device) -> nn.Module:
    """
    Load a YOLO model and extract its frozen backbone.

    Args:
        weights_path (str): Path to YOLO weights (best.pt)
        device (torch.device): Device to load model on

    Returns:
        nn.Module: Frozen backbone (model.model[0])
    """
    model = YOLO(weights_path)
    model.to(device)

    # Extract backbone (model.model[0] is the backbone in YOLO)
    backbone = model.model.model[0]

    # Freeze all parameters
    for param in backbone.parameters():
        param.requires_grad = False

    backbone.eval()

    return backbone


def build_late_fusion_model(
    rgb_weights: str,
    depth_weights: str,
    device: Union[str, torch.device] = "cuda",
    num_classes: int = 1,
) -> LateFusionModel:
    """
    Build a complete Late Fusion model from A.1 and A.2 weights.

    Args:
        rgb_weights (str): Path to A.1 RGB model weights
        depth_weights (str): Path to A.2 Depth model weights
        device (Union[str, torch.device]): Device to load model on
        num_classes (int): Number of detection classes

    Returns:
        LateFusionModel: Complete late fusion model
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Building Late Fusion Model on {device}")
    print(f"  RGB weights: {rgb_weights}")
    print(f"  Depth weights: {depth_weights}")

    # Load frozen backbones
    print("\nLoading RGB backbone (frozen)...")
    rgb_backbone = load_frozen_backbone(rgb_weights, device)

    print("Loading Depth backbone (frozen)...")
    depth_backbone = load_frozen_backbone(depth_weights, device)

    # Create late fusion model
    model = LateFusionModel(
        rgb_backbone=rgb_backbone,
        depth_backbone=depth_backbone,
        num_classes=num_classes,
        fusion_channels=256,
    )

    model.to(device)

    # Load a reference YOLO model to get the detection head
    print("\nInitializing detection head...")
    ref_model = YOLO("yolo11n.pt")
    ref_model.to(device)

    # Extract detection head
    # In YOLO, the detection head is model.model[-1]
    detect_head = ref_model.model.model[-1]

    # Update detection head for our fused features
    # The detect head expects features at multiple scales
    # We need to adapt it for our single-scale fused features

    model.set_detection_head(detect_head)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")

    return model


def verify_model_architecture(model: LateFusionModel) -> Dict[str, any]:
    """
    Verify the model architecture and return statistics.

    Args:
        model (LateFusionModel): Model to verify

    Returns:
        Dict[str, any]: Architecture statistics
    """
    stats = {
        "rgb_backbone_frozen": all(
            not p.requires_grad for p in model.backbone.rgb_backbone.parameters()
        ),
        "depth_backbone_frozen": all(
            not p.requires_grad for p in model.backbone.depth_backbone.parameters()
        ),
        "fusion_p3_trainable": any(
            p.requires_grad for p in model.fusion_p3.parameters()
        ),
        "fusion_p4_trainable": any(
            p.requires_grad for p in model.fusion_p4.parameters()
        ),
        "fusion_p5_trainable": any(
            p.requires_grad for p in model.fusion_p5.parameters()
        ),
        "fusion_p3_channels": f"{model.fusion_p3.conv.in_channels}->{model.fusion_p3.conv.out_channels}",
        "fusion_p4_channels": f"{model.fusion_p4.conv.in_channels}->{model.fusion_p4.conv.out_channels}",
        "fusion_p5_channels": f"{model.fusion_p5.conv.in_channels}->{model.fusion_p5.conv.out_channels}",
    }

    if model.detect is not None:
        stats["detect_head_present"] = True
        stats["detect_trainable"] = any(
            p.requires_grad for p in model.detect.parameters()
        )
    else:
        stats["detect_head_present"] = False

    return stats


if __name__ == "__main__":
    # Test the model architecture
    print("=" * 60)
    print("Late Fusion Model - Architecture Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create dummy backbones for testing
    print("\nCreating test backbones...")

    # Load YOLO to get backbone structure
    try:
        yolo = YOLO("yolo11n.pt")
        backbone_rgb = yolo.model.model[0]
        backbone_depth = yolo.model.model[0]

        # Create model
        model = LateFusionModel(
            rgb_backbone=backbone_rgb,
            depth_backbone=backbone_depth,
            num_classes=1,
        )

        model.to(device)

        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        rgb = torch.randn(batch_size, 3, 640, 640).to(device)
        depth = torch.randn(batch_size, 3, 640, 640).to(device)

        with torch.no_grad():
            output = model(rgb, depth)

        print(f"Input RGB shape: {rgb.shape}")
        print(f"Input Depth shape: {depth.shape}")
        print(f"Output shape: {output.shape}")

        # Verify architecture
        print("\nVerifying architecture...")
        stats = verify_model_architecture(model)
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Check trainable parameters
        print("\nTrainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.numel():,}")

        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
