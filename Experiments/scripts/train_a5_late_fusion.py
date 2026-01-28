"""
A.5 Late Fusion Training Script

Experiment A.5: Late Fusion Architecture
- Frozen RGB backbone (from A.1)
- Frozen Depth backbone (from A.2)
- Trainable fusion layers (P3, P4, P5)
- Trainable detection head
- Proper YOLO v8DetectionLoss

Usage:
    cd Experiments/scripts
    python train_a5_late_fusion.py

Requirements:
    - A.1 RGB model weights (best.pt)
    - A.2 Depth model weights (best.pt)
    - RGB and Depth datasets in YOLO format
"""

import os
import sys
import gc
import time
import random
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

# Configuration
SEEDS = [42, 123, 456, 789, 101]
EXP_PREFIX = "exp_a5_fusion"
EPOCHS = 100
PATIENCE = 30
IMGSZ = 640
BATCH_SIZE = 8  # Reduced for dual backbone
NUM_WORKERS = 4

# Uniform augmentation (matches A.1-A.4b)
AUGMENT_PARAMS = {
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    'hsv_h': 0.0,
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'erasing': 0.0,
    'mosaic': 0.0,
    'mixup': 0.0,
    'degrees': 0.0,
    'copy_paste': 0.0,
}

# Paths
BASE_PATH = Path(r'D:/Work/Assisten Dosen/Anylabel/Experiments')
RGB_DATASET = BASE_PATH / 'datasets' / 'ffb_localization'
DEPTH_DATASET = BASE_PATH / 'datasets' / 'ffb_localization_depth'
RGB_WEIGHTS_DIR = BASE_PATH / 'runs' / 'detect'
DEPTH_WEIGHTS_DIR = BASE_PATH / 'runs' / 'detect'
RUNS_PATH = BASE_PATH / 'runs' / 'detect'
KAGGLE_OUTPUT = BASE_PATH / 'kaggleoutput'

RUNS_PATH.mkdir(parents=True, exist_ok=True)
KAGGLE_OUTPUT.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LateFusionDataset(Dataset):
    """Dataset that loads RGB and Depth images separately for late fusion."""

    def __init__(
        self,
        rgb_img_dir: Path,
        depth_img_dir: Path,
        label_dir: Path,
        img_size: int = 640,
        augment: bool = False,
        augment_params: dict = None
    ):
        self.rgb_img_dir = Path(rgb_img_dir)
        self.depth_img_dir = Path(depth_img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.augment_params = augment_params or {}

        # Get list of files (use RGB as reference)
        self.image_files = sorted([p.name for p in self.rgb_img_dir.glob('*.png')])

        # Filter to only include files that exist in both RGB and Depth
        valid_files = []
        for fname in self.image_files:
            rgb_exists = (self.rgb_img_dir / fname).exists()
            depth_exists = (self.depth_img_dir / fname).exists()
            label_exists = (self.label_dir / fname.replace('.png', '.txt')).exists()
            if rgb_exists and depth_exists and label_exists:
                valid_files.append(fname)

        self.image_files = valid_files
        print(f"[LateFusionDataset] Loaded {len(self)} valid samples")

    def __len__(self) -> int:
        return len(self.image_files)

    def _apply_augmentation(self, rgb, depth, labels):
        """Apply synchronized geometric augmentation to RGB and Depth."""
        h, w = rgb.shape[:2]

        # Horizontal flip
        if random.random() < self.augment_params.get('fliplr', 0.0):
            rgb = cv2.flip(rgb, 1)
            depth = cv2.flip(depth, 1)
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]

        # Scale and translate (affine transform)
        scale = self.augment_params.get('scale', 0.0)
        translate = self.augment_params.get('translate', 0.0)

        if scale > 0 or translate > 0:
            s = random.uniform(1 - scale, 1 + scale)
            tx = random.uniform(-translate, translate) * w
            ty = random.uniform(-translate, translate) * h

            M = np.array([
                [s, 0, tx + (1 - s) * w / 2],
                [0, s, ty + (1 - s) * h / 2]
            ], dtype=np.float32)

            rgb = cv2.warpAffine(rgb, M, (w, h), borderValue=(114, 114, 114))
            depth = cv2.warpAffine(depth, M, (w, h), borderValue=(0, 0, 0))

            if len(labels) > 0:
                x_center = labels[:, 1] * w
                y_center = labels[:, 2] * h
                box_w = labels[:, 3] * w
                box_h = labels[:, 4] * h

                x_center = x_center * s + tx + (1 - s) * w / 2
                y_center = y_center * s + ty + (1 - s) * h / 2
                box_w = box_w * s
                box_h = box_h * s

                labels[:, 1] = x_center / w
                labels[:, 2] = y_center / h
                labels[:, 3] = box_w / w
                labels[:, 4] = box_h / h
                labels[:, 1:] = np.clip(labels[:, 1:], 0, 1)

                valid = (labels[:, 3] > 0.001) & (labels[:, 4] > 0.001)
                labels = labels[valid]

        return rgb, depth, labels

    def __getitem__(self, idx: int) -> Dict:
        fname = self.image_files[idx]

        rgb_path = self.rgb_img_dir / fname
        rgb = cv2.imread(str(rgb_path))

        depth_path = self.depth_img_dir / fname
        depth = cv2.imread(str(depth_path))

        if depth is None:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            if depth is not None:
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            else:
                depth = np.zeros_like(rgb)

        label_path = self.label_dir / fname.replace('.png', '.txt')
        if label_path.exists():
            labels = np.loadtxt(str(label_path), ndmin=2).astype(np.float32)
            if labels.size == 0:
                labels = np.zeros((0, 5), dtype=np.float32)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        if rgb.shape[:2] != (self.img_size, self.img_size):
            rgb = cv2.resize(rgb, (self.img_size, self.img_size))
            depth = cv2.resize(depth, (self.img_size, self.img_size))

        if self.augment:
            rgb, depth, labels = self._apply_augmentation(rgb, depth, labels)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.from_numpy(depth).permute(2, 0, 1).float() / 255.0
        labels_tensor = torch.from_numpy(labels).float()

        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'labels': labels_tensor,
            'batch_idx': torch.zeros(len(labels)),
            'img_path': str(rgb_path),
        }


def collate_fn(batch):
    """Custom collate function for variable-length labels."""
    rgb = torch.stack([item['rgb'] for item in batch])
    depth = torch.stack([item['depth'] for item in batch])

    labels_list = []
    for i, item in enumerate(batch):
        labels = item['labels']
        if len(labels) > 0:
            batch_idx = torch.full((len(labels), 1), i, dtype=torch.float32)
            labels_with_idx = torch.cat([batch_idx, labels], dim=1)
            labels_list.append(labels_with_idx)

    if labels_list:
        labels = torch.cat(labels_list, dim=0)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)

    return {
        'rgb': rgb,
        'depth': depth,
        'labels': labels,
        'img_paths': [item['img_path'] for item in batch],
    }


class LateFusionModel(nn.Module):
    """Late Fusion Model for FFB Detection with Multi-Scale Features."""

    def __init__(
        self,
        rgb_model_path: str,
        depth_model_path: str,
        num_classes: int = 1,
        device: str = 'cuda'
    ):
        super().__init__()

        print("\n" + "="*60)
        print("Initializing Late Fusion Model (Multi-Scale)")
        print("="*60)

        self.device = device
        self.num_classes = num_classes

        print(f"\nLoading RGB backbone from: {rgb_model_path}")
        self.rgb_yolo = YOLO(rgb_model_path)
        self.rgb_backbone = self.rgb_yolo.model.model[0]
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        self.rgb_backbone.eval()
        print(f"  RGB backbone frozen")

        print(f"\nLoading Depth backbone from: {depth_model_path}")
        self.depth_yolo = YOLO(depth_model_path)
        self.depth_backbone = self.depth_yolo.model.model[0]
        for param in self.depth_backbone.parameters():
            param.requires_grad = False
        self.depth_backbone.eval()
        print(f"  Depth backbone frozen")

        # Fusion layers for each scale
        self.fusion_p3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.fusion_p4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        self.fusion_p5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )

        for m in [self.fusion_p3, self.fusion_p4, self.fusion_p5]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        print(f"\nFusion layers initialized")

        self.detect = deepcopy(self.rgb_yolo.model.model[-1])
        for param in self.detect.parameters():
            param.requires_grad = True
        print(f"Detection head initialized (trainable)")

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"\n[Parameter Count]")
        print(f"  Total:      {total:,}")
        print(f"  Trainable:  {trainable:,} ({100*trainable/total:.1f}%)")
        print(f"  Frozen:     {frozen:,} ({100*frozen/total:.1f}%)")
        print("="*60)

    def _extract_features(self, backbone, x):
        """Extract multi-scale features (P3, P4, P5) from backbone."""
        features = []
        extract_layers = [4, 6, 8]

        for i, module in enumerate(backbone):
            x = module(x)
            if i in extract_layers:
                features.append(x)
                if len(features) == len(extract_layers):
                    break

        return features

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        with torch.no_grad():
            rgb_features = self._extract_features(self.rgb_backbone, rgb)
            depth_features = self._extract_features(self.depth_backbone, depth)

        fused_p3 = self.fusion_p3(torch.cat([rgb_features[0], depth_features[0]], dim=1))
        fused_p4 = self.fusion_p4(torch.cat([rgb_features[1], depth_features[1]], dim=1))
        fused_p5 = self.fusion_p5(torch.cat([rgb_features[2], depth_features[2]], dim=1))

        fused_features = [fused_p3, fused_p4, fused_p5]
        output = self.detect(fused_features)

        return output

    def train(self, mode=True):
        super().train(mode)
        self.rgb_backbone.eval()
        self.depth_backbone.eval()
        return self


class LateFusionTrainer:
    """Proper trainer for Late Fusion model using Ultralytics YOLO loss."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 100,
        patience: int = 30,
        lr: float = 0.01,
        save_dir: Path = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.save_dir = Path(save_dir) if save_dir else Path('runs/fusion')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=0.937,
            weight_decay=0.0005,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        self.scaler = GradScaler()
        self.criterion = self._create_loss_function()

        self.best_fitness = 0.0
        self.epochs_no_improve = 0
        self.history = []

    def _create_loss_function(self):
        """Create YOLO v8DetectionLoss function."""
        class DummyDetectionModel:
            def __init__(self, model):
                self.model = model
                self.nc = model.num_classes
                if hasattr(model, 'detect') and model.detect is not None:
                    self.stride = model.detect.stride
                else:
                    self.stride = torch.tensor([8., 16., 32.])

        dummy_model = DummyDetectionModel(self.model)
        return v8DetectionLoss(dummy_model)

    def _prepare_batch(self, batch: Dict) -> Dict:
        rgb = batch['rgb'].to(self.device)
        depth = batch['depth'].to(self.device)
        labels = batch['labels'].to(self.device)

        if len(labels) > 0:
            batch_idx = labels[:, 0].long()
            cls = labels[:, 1].long()
            bboxes = labels[:, 2:6]
        else:
            batch_idx = torch.zeros(0, dtype=torch.long, device=self.device)
            cls = torch.zeros(0, dtype=torch.long, device=self.device)
            bboxes = torch.zeros(0, 4, device=self.device)

        return {
            'img': (rgb, depth),
            'batch_idx': batch_idx,
            'cls': cls,
            'bboxes': bboxes,
        }

    def train_epoch(self, epoch: int):
        self.model.train()
        self.model.rgb_backbone.eval()
        self.model.depth_backbone.eval()

        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_dfl_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

        for batch in pbar:
            prepared_batch = self._prepare_batch(batch)
            self.optimizer.zero_grad()

            with autocast():
                rgb = prepared_batch['img'][0]
                depth = prepared_batch['img'][1]
                preds = self.model(rgb, depth)
                loss, loss_items = self.criterion(preds, prepared_batch)

                box_loss = loss_items[0] if len(loss_items) > 0 else 0
                cls_loss = loss_items[1] if len(loss_items) > 1 else 0
                dfl_loss = loss_items[2] if len(loss_items) > 2 else 0

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            epoch_box_loss += box_loss.item() if torch.is_tensor(box_loss) else box_loss
            epoch_cls_loss += cls_loss.item() if torch.is_tensor(cls_loss) else cls_loss
            epoch_dfl_loss += dfl_loss.item() if torch.is_tensor(dfl_loss) else dfl_loss

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'box': f"{box_loss.item() if torch.is_tensor(box_loss) else box_loss:.4f}",
            })

        num_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / num_batches,
            'box_loss': epoch_box_loss / num_batches,
            'cls_loss': epoch_cls_loss / num_batches,
            'dfl_loss': epoch_dfl_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0.0

        for batch in self.val_loader:
            prepared_batch = self._prepare_batch(batch)
            rgb = prepared_batch['img'][0]
            depth = prepared_batch['img'][1]
            preds = self.model(rgb, depth)
            loss, _ = self.criterion(preds, prepared_batch)
            val_loss += loss.item()

        return {'loss': val_loss / len(self.val_loader)}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        weights_dir = self.save_dir / 'weights'
        weights_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_fitness': self.best_fitness,
        }

        torch.save(checkpoint, weights_dir / 'last.pt')
        if is_best:
            torch.save(checkpoint, weights_dir / 'best.pt')

    def train(self):
        print(f"\nStarting training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            fitness = 1.0 / (val_metrics['loss'] + 1e-6)
            is_best = fitness > self.best_fitness

            if is_best:
                self.best_fitness = fitness
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, is_best)

            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'fitness': fitness,
            })

            print(f"Epoch {epoch+1}: train={train_metrics['loss']:.4f}, "
                  f"val={val_metrics['loss']:.4f}, fitness={fitness:.4f}")

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        return self.history


def find_best_weights(weights_dir: Path, exp_prefix: str) -> str:
    """Find the best weights file from experiment runs."""
    patterns = [
        f"{exp_prefix}_seed*/weights/best.pt",
        f"{exp_prefix}*/weights/best.pt",
    ]

    for pattern in patterns:
        matches = list(weights_dir.glob(pattern))
        if matches:
            return str(sorted(matches)[0])

    return None


def main():
    print("="*60)
    print("A.5 LATE FUSION TRAINING")
    print("="*60)

    # Find weights
    rgb_weights = find_best_weights(RGB_WEIGHTS_DIR, 'exp_a1_rgb')
    depth_weights = find_best_weights(DEPTH_WEIGHTS_DIR, 'exp_a2_depth')

    if rgb_weights is None or depth_weights is None:
        print("ERROR: Pre-trained weights not found!")
        print("Please run A.1 and A.2 experiments first.")
        return

    print(f"RGB weights: {rgb_weights}")
    print(f"Depth weights: {depth_weights}")

    results_all = {}

    for idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*60}")
        print(f"Training Seed {seed} ({idx}/{len(SEEDS)})")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        try:
            train_dataset = LateFusionDataset(
                rgb_img_dir=RGB_DATASET / 'images' / 'train',
                depth_img_dir=DEPTH_DATASET / 'images' / 'train',
                label_dir=RGB_DATASET / 'labels' / 'train',
                img_size=IMGSZ,
                augment=True,
                augment_params=AUGMENT_PARAMS
            )

            val_dataset = LateFusionDataset(
                rgb_img_dir=RGB_DATASET / 'images' / 'val',
                depth_img_dir=DEPTH_DATASET / 'images' / 'val',
                label_dir=RGB_DATASET / 'labels' / 'val',
                img_size=IMGSZ,
                augment=False
            )

            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
            )

            model = LateFusionModel(
                rgb_model_path=rgb_weights,
                depth_model_path=depth_weights,
                num_classes=1,
                device=str(DEVICE)
            )

            save_dir = RUNS_PATH / f"{EXP_PREFIX}_seed{seed}"

            trainer = LateFusionTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=DEVICE,
                epochs=EPOCHS,
                patience=PATIENCE,
                save_dir=save_dir,
            )

            history = trainer.train()

            results_all[seed] = {
                'completed': True,
                'best_fitness': trainer.best_fitness,
            }

            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            results_all[seed] = {'completed': False, 'error': str(e)}

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print(f"Successful: {sum(1 for r in results_all.values() if r.get('completed', False))}/{len(SEEDS)}")
    print("="*60)


if __name__ == "__main__":
    main()
