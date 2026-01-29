# Action Plan: FFB Oil Palm Detection Experiments V2

**Created:** 2026-01-28
**Version:** 2.0
**Objective:** Standardized experiments with uniform augmentation and domain adaptation

---

## Summary of Changes

| Point | Change | Affected Experiments |
|:-----|:----------|:---------------------|
| 1 | Uniform geometric augmentation | A.1, A.2, A.3, A.4a, A.4b |
| 2 | BatchNorm statistics reset | A.2, A.3, A.4a, A.4b |
| 3 | Late Fusion Model | A.5 (new) |

**B Series (B.1, B.2):** Unchanged (baseline established)

---

## Change Details

### 1. Uniform Augmentation (All Experiments)

**Configuration:**

```python
augment_params = dict(
    translate=0.1,    # geometric
    scale=0.5,        # geometric
    fliplr=0.5,       # geometric
    hsv_h=0.0,        # disabled (non-geometric)
    hsv_s=0.0,        # disabled (non-geometric)
    hsv_v=0.0,        # disabled (non-geometric)
    erasing=0.0,      # disabled (non-geometric)
    mosaic=0.0,       # disabled (non-geometric)
    mixup=0.0,        # disabled (non-geometric)
)
```

**Affected:** A.1, A.2, A.3, A.4a, A.4b (all 5 seeds each)

---

### 2. Reset BatchNorm Running Stats

**For depth experiments:** A.2, A.3, A.4a, A.4b

#### A.2, A.4a (3-Channel Depth) - Real Training Images

> **Implementation:** BN reset using **100 real training images**.

```python
import yaml
from PIL import Image
from torchvision import transforms

# Load data config to get training image paths
with open(config_path) as f:
    data_dict = yaml.safe_load(f)

# Get training image paths
train_path = Path(data_dict['path']) / data_dict['train']
train_images = sorted(list(train_path.glob('*.png')) + list(train_path.glob('*.jpg')))

# Select 100 images (or all if less than 100)
num_images = min(100, len(train_images))
selected_images = train_images[:num_images]

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((IMGSZ, IMGSZ)),
    transforms.ToTensor(),
])

# Forward pass with real images (batch size 16)
batch_size = 16
with torch.no_grad():
    for i in range(0, num_images, batch_size):
        batch_paths = selected_images[i:i+batch_size]
        batch_tensors = []

        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(DEVICE)
            _ = model.model(batch)
```

#### A.3, A.4b (4-Channel RGBD) - Real Training Images

> **Implementation:** BN reset using **100 real training images** via callback `on_train_start` in `RGBD4ChTrainer`.

```python
class RGBD4ChTrainer(DetectionTrainer):
    def __init__(self, overrides=None):
        super().__init__(overrides=overrides)
        self.add_callback("on_train_start", self._bn_reset_callback)

    def _bn_reset_callback(self, trainer):
        """Reset BN with 100 real training images."""
        # 1. Reset running stats
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = 0.1

        # 2. Forward pass 100 training images
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for batch in self.train_loader:
                images = batch['img'].to(device)
                if images.dtype == torch.uint8:
                    images = images.float() / 255.0
                _ = self.model(images)
```

**Key Implementation Details:**
- `super().__init__(overrides=overrides)` - explicit keyword argument (fix TypeError)
- `images.float() / 255.0` - normalize uint8 to float32 (fix ByteTensor error)
- `int(self.model.stride.max())` - fix Tensor attribute error
- `data=self.data` - fix NoneType error in build_dataset()

---

### 3. Late Fusion Model (A.5)

**Multi-Scale Architecture (P3, P4, P5):**

```
Input RGB (3ch)          Input Depth (3ch)
     │                         │
     ▼                         ▼
┌─────────────┐           ┌─────────────┐
│  RGB Branch │           │ Depth Branch│
│ (A.1 Frozen)│           │ (A.2 Frozen)│
│  Backbone   │           │  Backbone   │
│  Outputs:   │           │  Outputs:   │
│  P3, P4, P5 │           │  P3, P4, P5 │
└──────┬──────┘           └──────┬──────┘
       │                         │
       ├───────┬───────┬─────────┤
       │       │       │         │
       ▼       ▼       ▼         ▼
    [Concat] [Concat] [Concat]  ← Each scale
       │       │       │
       ▼       ▼       ▼
   1x1 Conv 1x1 Conv 1x1 Conv   ← Fusion layers (trainable)
   512→256  512→256  512→256
       │       │       │
       └───────┼───────┘
               ▼
       ┌───────────────┐
       │  YOLO Detect  │  ← Detection Head (trainable)
       │   Head        │     Multi-scale output
       └───────────────┘
```

**Specifications:**
- RGB Branch: Load A.1 weights, freeze 100%
- Depth Branch: Load A.2 weights, freeze 100%
- Fusion layers: 3 Conv2d layers 512→256 + BatchNorm + SiLU (P3, P4, P5)
- Detection head: YOLOv11 Detect head trainable from scratch
- Loss: v8DetectionLoss (box_loss + cls_loss + dfl_loss)

**Note:** Architecture uses multi-scale (P3, P4, P5) for compatibility with YOLO Detect head requiring 3-level feature pyramid.

**Training:**
- Epochs: 100 (matching A.1/A.2 settings)
- Seeds: 42, 123, 456, 789, 101

---

## Total Training Runs

| Experiment | Seeds | New Augmentation | BN Reset | Status |
|:-----------|:-----:|:----------------:|:--------:|:-------|
| A.1 RGB Only | 5 | ✅ | - | Re-run |
| A.2 Real Depth | 5 | ✅ | ✅ | Re-run |
| A.3 RGB+Real Depth | 5 | ✅ | ✅ | Re-run |
| A.4a Synthetic Depth | 5 | ✅ | ✅ | Re-run |
| A.4b RGB+Synthetic | 5 | ✅ | ✅ | Re-run |
| **A.5 Late Fusion** | **5** | ✅ | - | **New** |
| **Total** | **30 runs** | | | |

---

## Estimated Timeline

| Phase | Duration | Description |
|:------|:---------|:------------|
| Update augmentation scripts | 2-3 hours | Modify 5 training scripts |
| Implement BN reset | 2-3 hours | Add functions + integration |
| Implement A.5 | 4-6 hours | Late fusion architecture |
| Training A.1-A.4b | 2-3 days | 25 runs × 100 epochs |
| Training A.5 | 1 day | 5 runs × 100 epochs |
| Evaluation & reporting | 1 day | Generate metrics, plots, update Results |
| **Total** | **4-6 days** | Parallel training can accelerate |

---

## Implementation Checklist

### Training Scripts

- [x] Update `train_a1_rgb.py` - geometric-only augmentation
- [x] Update `train_a2_depth.py` - augmentation + BN reset (100 real images)
- [x] Update `train_a3_rgbd.py` - augmentation + BN reset (100 real images, 4ch)
- [x] Update `train_a4a_synthetic_depth.py` - augmentation + BN reset (100 real images)
- [x] Update `train_a4b_rgbd_synthetic.py` - augmentation + BN reset (100 real images, 4ch)
- [x] Create `train_a5_late_fusion.py` - new model

### Notebooks v2 (Kaggle) - ✅ All Complete

- [x] `train_a1_rgb_v2.ipynb` - 5 seeds, uniform aug
- [x] `train_a2_depth_v2.ipynb` - 5 seeds, uniform aug + BN reset (100 real images)
- [x] `train_a3_rgbd_v2.ipynb` - 5 seeds, uniform aug + BN reset 100 real images + RGBD4ChTrainer + RGBD4ChValidator
- [x] `train_a4a_synthetic_depth_v2.ipynb` - 5 seeds, uniform aug + BN reset (100 real images)
- [x] `train_a4b_rgbd_synthetic_v2.ipynb` - 5 seeds, uniform aug + BN reset 100 real images + RGBD4ChTrainer + RGBD4ChValidator
- [x] `train_a5_late_fusion_v2.ipynb` - 5 seeds, multi-scale fusion + proper YOLO loss

### New Modules

- [x] `reset_bn.py` - BatchNorm reset functions (100 real images method)
- [x] `late_fusion_model.py` - LateFusionModel class (multi-scale P3/P4/P5)
- [x] `late_fusion_trainer.py` - Custom trainer for A.5 with v8DetectionLoss

---

## Output Structure

```
5_seed_v2/
├── train_a1_rgb/
│   ├── runs/detect/exp_a1_rgb_seed*/
│   └── kaggleoutput/a1_rgb_results.txt
├── train-a2-depth/
│   └── ...
├── train-a3-rgbd/
│   └── ...
├── train-a4a-synthetic-depth/
│   └── ...
├── train-a4b-rgbd-synthetic/
│   └── ...
├── train-a5-late-fusion/          # NEW
│   ├── runs/detect/exp_a5_fusion_seed*/
│   └── kaggleoutput/a5_fusion_results.txt
└── Results_v2.md                   # Updated with A.5
```

---

## Important Notes

1. **A.1 and A.2 Weights:** Use best weights from previous training as input for A.5.

2. **BN Reset (Critical):**
   - Only for depth models (A.2-A.4b)
   - **Must** call `model.model.to(DEVICE)` before BN reset
   - Use **100 real training images**

3. **A.5 Input:** Dataloader loads RGB and Depth separately (2 paths), not 4-channel fused.

4. **GPU Memory:** A.5 requires 2× backbone, consider batch size 8 if OOM.

5. **Metrics Access Pattern:**
   ```python
   # Correct (latest API)
   mAP50 = results.box.map50
   mAP50_95 = results.box.map

   # Incorrect (old API)
   mAP50 = results.results_dict['metrics/mAP50(B)']
   ```

---

## Risks & Mitigation

| Risk | Mitigation |
|:-------|:---------|
| Training time too long | Parallel runs on multiple GPU/Kaggle accounts |
| OOM on A.5 | Batch size 16→8, or use gradient accumulation |
| A.5 results not better | Document as exploration, focus on A.1-A.4b |
| BN reset no significant impact | Consider as ablation study |

---

## Update Log

### 2026-01-29 - Implementation Complete

**Technical Changes from Initial Plan:**

1. **BN Reset Method - A.2, A.4a (3-Channel)**
   - **Initial Plan:** Use `train_loader` with 100 batches
   - **Implementation:** Use **100 real training images** via PIL + transforms
   - **Reason:** Domain adaptation works better with real data
   - **Impact:** BN stats distribution matches target data (depth/synthetic)

2. **BN Reset Method - A.3, A.4b (4-Channel RGBD)**
   - **Initial Plan:** Use dummy input
   - **Implementation:** Use **100 real training images** via callback `on_train_start`
   - **Reason:** Domain adaptation works better with real data
   - **Implementation:** `RGBD4ChTrainer` with `_bn_reset_callback()`

3. **A.3, A.4b Custom Trainer Architecture**
   - **RGBD4ChTrainer:**
     - `get_model()` - Auto-convert 3ch → 4ch first conv layer
     - `build_dataset()` - Override with `RGBDDataset` (4-channel `load_image()`)
     - `_bn_reset_callback()` - BN reset with 100 real training images
     - `get_validator()` - Return `RGBD4ChValidator`
   - **RGBD4ChValidator:**
     - `build_dataset()` - 4-channel dataset for validation
     - `setup_model()` - Ensure 4ch conversion

4. **Cell 8 Evaluation Fix**
   - **Implementation:** `evaluate_rgbd_model()` using explicit `RGBD4ChValidator`
   - **Reason:** `model.val()` default doesn't handle 4-channel properly
   - **Return:** `validator.metrics` (not return value from `validator()`)

5. **A.5 Architecture**
   - **Initial Plan:** Single-scale (P3 only) fusion
   - **Implementation:** Multi-scale (P3, P4, P5) fusion
   - **Reason:** YOLO Detect head requires 3 level feature pyramid

6. **Device Transfer**
   - **Addition:** `model.model.to(DEVICE)` before BN reset (important!)
   - **Reason:** BN layers need to be on GPU before forward pass

7. **Metrics Access**
   - **Update:** `results.box.map50` instead of `results.results_dict['metrics/mAP50(B)']`
   - **Reason:** Latest Ultralytics API changed results object structure

**Status:** All v2 notebooks ready for training (30 runs total).
