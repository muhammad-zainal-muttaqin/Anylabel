"""
Shared utility functions for 5_seed_v2 experiments.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union


def setup_paths() -> Dict[str, Union[str, bool]]:
    """
    Setup paths based on environment (Kaggle vs Local).

    Returns:
        Dictionary containing base path, datasets path, and environment flag.
    """
    if os.path.exists('/kaggle'):
        return {
            'base': '/kaggle/working',
            'datasets': '/kaggle/input',
            'is_kaggle': True
        }
    else:
        return {
            'base': 'D:/Work/Assisten Dosen/Anylabel/Experiments',
            'datasets': 'D:/Work/Assisten Dosen/Anylabel/Experiments/datasets',
            'is_kaggle': False
        }


def save_results(results: Any, exp_name: str, seed: int, output_dir: str) -> None:
    """
    Save training results to file.

    Args:
        results: Training results object from YOLO
        exp_name: Experiment name (e.g., 'A.1', 'A.2')
        seed: Random seed used for training
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    result_file = os.path.join(output_dir, f'{exp_name.lower().replace(".", "")}_results.txt')

    with open(result_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Experiment: {exp_name}, Seed: {seed}\n")
        f.write(f"{'='*50}\n")
        f.write(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}\n")
        f.write(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}\n")
        f.write(f"Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}\n")
        f.write(f"Recall: {results.results_dict.get('metrics/recall(B)', 0):.4f}\n")


def aggregate_results(exp_name: str, seeds: List[int], results_list: List[Any]) -> Dict[str, Any]:
    """
    Aggregate results across multiple seeds.

    Args:
        exp_name: Experiment name
        seeds: List of seeds used
        results_list: List of results objects from each seed

    Returns:
        Dictionary containing aggregated metrics with mean and std
    """
    metrics = {
        'experiment': exp_name,
        'seeds': seeds,
        'mAP50': [],
        'mAP50-95': [],
        'precision': [],
        'recall': [],
    }

    for r in results_list:
        metrics['mAP50'].append(r.results_dict.get('metrics/mAP50(B)', 0))
        metrics['mAP50-95'].append(r.results_dict.get('metrics/mAP50-95(B)', 0))
        metrics['precision'].append(r.results_dict.get('metrics/precision(B)', 0))
        metrics['recall'].append(r.results_dict.get('metrics/recall(B)', 0))

    # Calculate statistics
    for key in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        metrics[f'{key}_mean'] = np.mean(metrics[key])
        metrics[f'{key}_std'] = np.std(metrics[key])

    return metrics


def cleanup_gpu() -> None:
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_summary_table(results_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted summary table of all experiment results.

    Args:
        results_dict: Dictionary mapping experiment names to their metrics
    """
    print("\n" + "="*80)
    print("5_SEED_V2 EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<12} {'mAP50':<20} {'mAP50-95':<20}")
    print("-"*80)

    for exp_name, metrics in results_dict.items():
        map50_mean = metrics.get('mAP50_mean', 0)
        map50_std = metrics.get('mAP50_std', 0)
        map5095_mean = metrics.get('mAP50-95_mean', 0)
        map5095_std = metrics.get('mAP50-95_std', 0)

        map50_str = f"{map50_mean:.4f} ± {map50_std:.4f}"
        map5095_str = f"{map5095_mean:.4f} ± {map5095_std:.4f}"

        print(f"{exp_name:<12} {map50_str:<20} {map5095_str:<20}")

    print("="*80)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        torch.device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # Test utilities
    paths = setup_paths()
    print(f"Environment: {'Kaggle' if paths['is_kaggle'] else 'Local'}")
    print(f"Base path: {paths['base']}")
    print(f"Datasets path: {paths['datasets']}")
    print(f"Device: {get_device()}")
