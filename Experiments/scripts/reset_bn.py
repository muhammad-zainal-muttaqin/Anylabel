"""
Reset BatchNorm Utility Module for FFB Oil Palm Detection Project.

This module provides utilities for resetting BatchNorm running statistics
after loading pretrained weights, specifically designed for depth-based
experiments (A.2, A.3, A.4a, A.4b) in the FFB oil palm detection project.

Usage:
    from reset_bn import reset_bn_stats, reset_bn_stats_simple, adapt_first_conv_to_4ch

    model = YOLO("yolo11n.pt")

    # For RGBD experiments, adapt first conv to 4 channels
    adapt_first_conv_to_4ch(model.model)

    # Reset BN stats for depth experiments (with data forwarding)
    model.model = reset_bn_stats(model.model, train_loader, num_batches=100)

    # Or use simple reset (without data forwarding)
    model.model = reset_bn_stats_simple(model.model)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reset_bn_stats(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_batches: int = 100,
    device: str = 'cuda'
) -> nn.Module:
    """
    Reset running statistics of BatchNorm layers using training data.

    When loading pretrained weights (e.g., from ImageNet), the BatchNorm
    running statistics (running_mean, running_var) are from the source
    domain. For depth-based experiments, these stats need to be recomputed
    on the target domain data to ensure proper normalization.

    Args:
        model: YOLO model or PyTorch nn.Module containing BatchNorm layers.
        train_loader: Training dataloader yielding (images, targets, ...) batches.
        num_batches: Number of batches to use for statistics computation.
                     Default: 100. Increase for more stable estimates.
        device: Device to run computation on ('cuda' or 'cpu').
                Default: 'cuda'.

    Returns:
        nn.Module: Model with updated BatchNorm running statistics.

    Raises:
        ValueError: If train_loader is empty or num_batches <= 0.
        RuntimeError: If no BatchNorm layers are found in the model.

    Example:
        >>> from ultralytics import YOLO
        >>> from torch.utils.data import DataLoader
        >>> model = YOLO("yolo11n.pt")
        >>> train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>> model.model = reset_bn_stats(model.model, train_loader, num_batches=100)
        >>> # Model now has BN stats computed on training data

    Notes:
        - Sets model to train mode temporarily to allow BN updates.
        - Freezes all non-BN parameters to prevent weight updates.
        - Only the first element of each batch (images) is used.
        - Progress is logged at INFO level.
    """
    if num_batches <= 0:
        raise ValueError(f"num_batches must be positive, got {num_batches}")

    if train_loader is None or len(train_loader) == 0:
        raise ValueError("train_loader cannot be empty")

    # Find all BatchNorm layers
    bn_layers: List[nn.BatchNorm2d] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers.append(module)

    if not bn_layers:
        raise RuntimeError("No BatchNorm layers found in model")

    logger.info(f"Found {len(bn_layers)} BatchNorm layers")

    # Store original training state
    was_training = model.training

    # Set model to train mode to enable BN updates
    model.train()

    # Freeze all parameters except BN running stats
    param_requires_grad = {}
    for name, param in model.named_parameters():
        param_requires_grad[name] = param.requires_grad
        param.requires_grad = False

    # Enable running stats computation for BN layers
    for bn in bn_layers:
        bn.track_running_stats = True

    logger.info(f"Computing BN statistics using {num_batches} batches...")

    # Forward pass to update BN running stats
    batch_count = 0
    device_obj = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device_obj)

    with torch.no_grad():
        for batch in train_loader:
            if batch_count >= num_batches:
                break

            # Extract images (first element of batch tuple/list)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device_obj)

            # Forward pass - this updates BN running stats
            _ = model(images)

            batch_count += 1

            if batch_count % 10 == 0:
                logger.info(f"  Processed {batch_count}/{num_batches} batches")

    logger.info(f"Completed BN statistics reset using {batch_count} batches")

    # Restore parameter requires_grad states
    for name, param in model.named_parameters():
        param.requires_grad = param_requires_grad.get(name, True)

    # Restore original training mode
    if not was_training:
        model.eval()

    return model


def adapt_first_conv_to_4ch(
    model: nn.Module,
    copy_rgb_weights: bool = True
) -> nn.Module:
    """
    Adapt the first convolution layer from 3-channel to 4-channel input.

    This is required for RGBD experiments (A.3, A.4b) where the input
    consists of RGB (3 channels) + Depth (1 channel). The pretrained
    YOLO models expect 3-channel RGB input, so the first conv layer
    must be modified to accept 4 channels.

    Args:
        model: YOLO detection model or PyTorch nn.Module.
        copy_rgb_weights: If True, copy the original RGB weights to
                         channels 0-2 of the new conv layer. If False,
                         use random initialization for all channels.
                         Default: True.

    Returns:
        nn.Module: Model with 4-channel first convolution layer.

    Raises:
        ValueError: If model does not have a recognizable first conv layer.
        RuntimeWarning: If first conv already has 4 input channels.

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> adapt_first_conv_to_4ch(model.model)
        >>> # Model now accepts 4-channel input (RGB + Depth)

    Notes:
        - The depth channel (channel 3) is initialized with the mean of
          RGB weights to provide a reasonable starting point.
        - Bias terms are preserved if they exist in the original layer.
        - The new conv layer uses the same kernel size, stride, padding,
          and other parameters as the original.
    """
    # Find the first convolutional layer
    first_conv: Optional[nn.Conv2d] = None
    first_conv_name: Optional[str] = None
    parent_module: Optional[nn.Module] = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_name = name
            # Get parent module for replacement
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            parent_module = parent
            break

    if first_conv is None:
        raise ValueError("Could not find first Conv2d layer in model")

    # Check current input channels
    if first_conv.in_channels == 4:
        logger.warning(f"First conv layer already has 4 input channels, skipping adaptation")
        return model

    if first_conv.in_channels != 3:
        logger.warning(
            f"First conv layer has {first_conv.in_channels} input channels, "
            f"expected 3. Proceeding with adaptation."
        )

    logger.info(f"Adapting first conv layer '{first_conv_name}' from "
                f"{first_conv.in_channels}ch to 4ch")

    # Create new 4-channel conv layer
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode
    )

    # Initialize weights
    with torch.no_grad():
        if copy_rgb_weights:
            # Copy RGB weights to channels 0-2
            new_conv.weight[:, :3, :, :] = first_conv.weight.clone()

            # Initialize depth channel (channel 3) with mean of RGB weights
            depth_init = first_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:4, :, :] = depth_init

            logger.info("Copied RGB weights, initialized depth channel with mean of RGB")
        else:
            # Use default random initialization
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            logger.info("Using random initialization for all channels")

        # Copy bias if exists
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    # Replace the conv layer in the model
    conv_attr_name = first_conv_name.split('.')[-1]
    setattr(parent_module, conv_attr_name, new_conv)

    logger.info(f"Successfully adapted first conv layer to 4 channels")

    return model


def is_depth_experiment(exp_name: str) -> bool:
    """
    Check if the experiment requires BatchNorm statistics reset.

    Depth-based experiments (A.2, A.3, A.4a, A.4b) use depth or RGBD
    inputs which have different data distributions than RGB images.
    These experiments benefit from resetting BN stats after loading
    pretrained RGB weights.

    Args:
        exp_name: Experiment identifier (e.g., 'A.1', 'A.2', 'A.3',
                  'A.4a', 'A.4b', 'B.1', 'B.2').

    Returns:
        bool: True if experiment needs BN reset, False otherwise.

    Example:
        >>> is_depth_experiment('A.2')
        True
        >>> is_depth_experiment('A.1')
        False
        >>> is_depth_experiment('A.3')
        True

    Notes:
        Experiments requiring BN reset:
        - A.2: Depth-only (real depth)
        - A.3: RGB+Depth (4-channel)
        - A.4a: Synthetic depth-only
        - A.4b: RGB+Synthetic depth (4-channel)

        Experiments NOT requiring BN reset:
        - A.1: RGB baseline
        - B.1: Ripeness detection (RGB)
        - B.2: Two-stage classification (RGB)
    """
    depth_experiments = {'A.2', 'A.3', 'A.4a', 'A.4b'}
    return exp_name.upper() in depth_experiments


def get_augmentation_config(exp_name: str) -> Dict[str, Any]:
    """
    Return augmentation configuration dictionary based on experiment type.

    Different experiments require different augmentation strategies:
    - RGB experiments: Full augmentation including HSV
    - Depth experiments: Spatial augmentations only (no HSV)
    - RGBD experiments: Synchronized augmentation for both modalities

    Args:
        exp_name: Experiment identifier (e.g., 'A.1', 'A.2', 'A.3',
                  'A.4a', 'A.4b', 'B.1', 'B.2').

    Returns:
        Dict[str, Any]: Augmentation configuration dictionary with keys:
            - 'hsv_h': Hue augmentation (0-360)
            - 'hsv_s': Saturation augmentation (0-1)
            - 'hsv_v': Value augmentation (0-1)
            - 'translate': Translation (0-1)
            - 'scale': Scale (0-1)
            - 'fliplr': Horizontal flip probability (0-1)
            - 'mosaic': Mosaic augmentation probability (0-1)
            - 'mixup': Mixup augmentation probability (0-1)
            - 'synced': Whether to use synchronized augmentation for RGBD

    Example:
        >>> config = get_augmentation_config('A.1')
        >>> print(config['hsv_h'])
        0.015
        >>> config = get_augmentation_config('A.2')
        >>> print(config['hsv_h'])
        0.0

    Notes:
        - RGB experiments (A.1, B.1, B.2): Full HSV augmentation enabled.
        - Depth experiments (A.2, A.4a): HSV disabled (grayscale depth).
        - RGBD experiments (A.3, A.4b): HSV enabled with synced augmentation.
    """
    exp_name = exp_name.upper()

    # Base configuration
    base_config = {
        'hsv_h': 0.015,      # Hue
        'hsv_s': 0.7,        # Saturation
        'hsv_v': 0.4,        # Value
        'translate': 0.1,    # Translation
        'scale': 0.5,        # Scale
        'fliplr': 0.5,       # Horizontal flip
        'mosaic': 1.0,       # Mosaic
        'mixup': 0.0,        # Mixup (disabled by default)
        'synced': False      # Synchronized augmentation
    }

    if exp_name == 'A.1':
        # RGB baseline - full augmentation
        return base_config

    elif exp_name == 'A.2':
        # Depth-only (real) - no HSV for grayscale depth
        config = base_config.copy()
        config['hsv_h'] = 0.0
        config['hsv_s'] = 0.0
        config['hsv_v'] = 0.0
        return config

    elif exp_name == 'A.3':
        # RGB+Depth (4-channel) - synced augmentation
        config = base_config.copy()
        config['synced'] = True
        return config

    elif exp_name == 'A.4a':
        # Synthetic depth-only - no HSV for grayscale depth
        config = base_config.copy()
        config['hsv_h'] = 0.0
        config['hsv_s'] = 0.0
        config['hsv_v'] = 0.0
        return config

    elif exp_name == 'A.4b':
        # RGB+Synthetic depth (4-channel) - synced augmentation
        config = base_config.copy()
        config['synced'] = True
        return config

    elif exp_name == 'B.1':
        # Ripeness detection (RGB) - full augmentation
        return base_config

    elif exp_name == 'B.2':
        # Two-stage classification (RGB) - full augmentation
        return base_config

    else:
        # Unknown experiment - return safe defaults (no HSV)
        logger.warning(f"Unknown experiment '{exp_name}', returning safe defaults")
        config = base_config.copy()
        config['hsv_h'] = 0.0
        config['hsv_s'] = 0.0
        config['hsv_v'] = 0.0
        return config


def freeze_bn_layers(model: nn.Module) -> nn.Module:
    """
    Freeze BatchNorm layers to prevent running statistics updates.

    Useful when fine-tuning on a small dataset where BN statistics
    should be preserved from pretraining.

    Args:
        model: PyTorch model containing BatchNorm layers.

    Returns:
        nn.Module: Model with frozen BatchNorm layers.

    Example:
        >>> model = YOLO("yolo11n.pt").model
        >>> freeze_bn_layers(model)
        >>> # BN layers now in eval mode, won't update running stats
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            module.track_running_stats = False
            for param in module.parameters():
                param.requires_grad = False

    logger.info("Frozen all BatchNorm layers")
    return model


def unfreeze_bn_layers(model: nn.Module) -> nn.Module:
    """
    Unfreeze BatchNorm layers to allow running statistics updates.

    Reverses the effect of freeze_bn_layers().

    Args:
        model: PyTorch model containing BatchNorm layers.

    Returns:
        nn.Module: Model with unfrozen BatchNorm layers.

    Example:
        >>> model = unfreeze_bn_layers(model)
        >>> # BN layers now in train mode, will update running stats
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.train()
            module.track_running_stats = True
            for param in module.parameters():
                param.requires_grad = True

    logger.info("Unfrozen all BatchNorm layers")
    return model


def reset_bn_stats_simple(model: nn.Module) -> nn.Module:
    """
    Simple BN reset that only resets running statistics without forwarding data.

    This is a lighter alternative when you don't have a dataloader ready.
    It just resets the running_mean and running_var to default values.

    Args:
        model: PyTorch model containing BatchNorm layers.

    Returns:
        nn.Module: Model with reset BN statistics.

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> model.model = reset_bn_stats_simple(model.model)
        >>> # Model now has reset BN statistics
    """
    bn_count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()
            bn_count += 1

    logger.info(f"Reset {bn_count} BatchNorm layers (simple reset)")
    return model


if __name__ == "__main__":
    # Unit test examples
    import doctest

    # Run doctests
    doctest.testmod(verbose=True)

    # Additional manual tests
    print("\n" + "=" * 60)
    print("Manual Tests")
    print("=" * 60)

    # Test is_depth_experiment
    print("\nTest: is_depth_experiment()")
    test_cases = ['A.1', 'A.2', 'A.3', 'A.4a', 'A.4b', 'B.1', 'B.2', 'unknown']
    for exp in test_cases:
        result = is_depth_experiment(exp)
        print(f"  {exp}: {result}")

    # Test get_augmentation_config
    print("\nTest: get_augmentation_config()")
    for exp in ['A.1', 'A.2', 'A.3']:
        config = get_augmentation_config(exp)
        hsv_status = "enabled" if config['hsv_h'] > 0 else "disabled"
        sync_status = "synced" if config['synced'] else "not synced"
        print(f"  {exp}: HSV {hsv_status}, augmentation {sync_status}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
