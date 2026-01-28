"""
Uniform augmentation configuration for all 5_seed_v2 experiments.
As per professor's requirement: geometric only.
"""

GEOMETRIC_AUGMENTATION = {
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    'hsv_h': 0.0,
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'erasing': 0.0,
    'mosaic': 0.0,
    'mixup': 0.0,
}

TRAINING_DEFAULTS = {
    'epochs': 100,
    'patience': 30,
    'batch': 16,
    'imgsz': 640,
    'device': 0,  # GPU
    'workers': 8,
    'optimizer': 'SGD',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
}

SEEDS = [42, 123, 456, 789, 101]


def get_training_config(exp_name: str) -> dict:
    """
    Get complete training configuration for an experiment.

    Args:
        exp_name: One of 'A.1', 'A.2', 'A.3', 'A.4a', 'A.4b', 'A.5'

    Returns:
        dict with all training parameters
    """
    config = {
        **GEOMETRIC_AUGMENTATION,
        **TRAINING_DEFAULTS,
    }

    # Experiment-specific adjustments
    if exp_name in ['A.3', 'A.4b']:  # RGBD 4-channel
        config['batch'] = 16  # Or adjust if OOM
    elif exp_name == 'A.5':  # Late fusion
        config['batch'] = 8   # Smaller due to dual backbone

    return config


def print_config(config: dict) -> None:
    """Pretty print configuration"""
    print("Training Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)


if __name__ == '__main__':
    # Test the configuration
    for exp in ['A.1', 'A.2', 'A.3', 'A.4a', 'A.4b', 'A.5']:
        print(f"\nConfiguration for {exp}:")
        cfg = get_training_config(exp)
        print_config(cfg)
