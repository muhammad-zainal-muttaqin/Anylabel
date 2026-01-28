# AGENTS.md - AI Coding Agent Guide

Guide for AI agents working on FFB Detection & Ripeness Classification (Oil Palm).

## Project Overview

- **Domain**: Computer Vision / Deep Learning (YOLOv8/YOLOv11)
- **Language**: Python 3.10+ (3.12 recommended)
- **Platform**: Windows 10/11 (primary), Linux compatible
- **Docs**: Indonesian for internal, English for configs

## Environment Setup

```powershell
# Activate virtual environment (REQUIRED)
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

## Build/Lint/Test Commands

```bash
# Run a Python script
cd Experiments/scripts
python train_a1_rgb.py

# Run a single training script
python train_a3_rgbd.py

# Lint check (if ruff installed)
ruff check Experiments/scripts/
ruff check --fix Experiments/scripts/

# Format code (if black installed)
black Experiments/scripts/

# Type check (if mypy installed)
mypy Experiments/scripts/

# Run tests (if pytest installed - currently not in use)
pytest tests/
pytest tests/test_specific.py -v
pytest tests/test_specific.py::test_function -v
```

## Code Style Guidelines

### Imports
```python
# Standard library first
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Union

# Third-party packages second
import torch
import numpy as np
from ultralytics import YOLO

# Local modules last
from common_utils import setup_paths
```

### Naming Conventions
- **Files**: `snake_case.py` (e.g., `train_a1_rgb.py`)
- **Functions**: `snake_case()` (e.g., `verify_dataset()`)
- **Variables**: `snake_case` (e.g., `train_config`, `dataset_path`)
- **Constants**: `UPPER_CASE` (e.g., `AUGMENT_PARAMS`, `SEEDS`)
- **Classes**: `PascalCase` (e.g., `CustomRGBDDataset`)
- **Experiments**: `exp_{id}_{description}_seed{number}`

### Type Hints
```python
# Use type hints where practical
def setup_paths() -> Dict[str, Union[str, bool]]:
    ...

def train_with_seed(seed: int, config: dict, output_file) -> dict:
    ...

# Use Path for file paths
from pathlib import Path
def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
```

### Error Handling
```python
try:
    model = YOLO(config['model'])
    results = model.train(...)
except Exception as e:
    print(f"ERROR: Training failed - {e}")
    return {'success': False, 'error': str(e)}
```

### Docstrings
```python
def function_name(param: type) -> return_type:
    """
    Brief description of what the function does.

    Args:
        param: Description of parameter

    Returns:
        Description of return value
    """
    pass
```

### File Paths
- **Always use absolute paths** in YAML configs
- Use `pathlib.Path` for path operations
- Use forward slashes in YAML configs: `D:/Work/...`
- Use Windows-style backslashes in Python strings: `r'D:\Work\...'`

### String Formatting
```python
# Use f-strings
print(f"mAP50: {map50:.4f}")

# For multi-line strings
header = f"""
{'='*70}
Eksperimen A.1: RGB Only
{'='*70}
"""
```

## Project Structure

```
Anylabel/
├── Experiments/
│   ├── scripts/           # Training scripts (run from here)
│   ├── configs/           # YOLO dataset YAML configs
│   ├── datasets/          # Processed data
│   ├── runs/              # Training outputs
│   └── notebooks/         # Jupyter notebooks
├── Dataset/               # Raw dataset storage
├── Reports/               # Final reports
└── venv/                  # Virtual environment
```

## Common Commands

```bash
# Training
cd Experiments/scripts
python train_a1_rgb.py       # RGB baseline
python train_a2_depth.py     # Depth only
python train_a3_rgbd.py      # RGB+Depth fusion

# Validation
yolo detect val model=runs/detect/exp_name/weights/best.pt \
    data=configs/ffb_localization.yaml split=test
```

## Key Conventions

1. **Always activate venv** before running scripts
2. **Use absolute paths** in all YAML configs
3. **Seeds**: Use 42 and 123 minimum (5 seeds for production)
4. **Reproducibility**: Set seeds for random, numpy, torch
5. **No emojis** in code (markdown only)
6. **Clean up GPU**: Call `torch.cuda.empty_cache()` when done
7. **Error handling**: Wrap training in try-except blocks
8. **Logging**: Write results to file with flush()

## Testing Standards

- **Detection**: mAP50, mAP50-95
- **Classification**: Top-1 Accuracy
- **Report**: mean ± std dev from multiple runs
- **Test Set**: Always evaluate on test split (not val)

## Notes for Agents

- Preserve existing code style when editing
- Clean up temporary files after failures
- No debug artifacts (`print()`, `pdb`) in final code
- Match documentation language: Indonesian for internal, English for configs
- All scripts should be run from `Experiments/scripts` directory

---

*Last Updated: 2026-01-28*
