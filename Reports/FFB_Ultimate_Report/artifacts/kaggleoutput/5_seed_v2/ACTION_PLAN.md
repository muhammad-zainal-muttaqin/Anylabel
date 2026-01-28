# Action Plan: Revisi Eksperimen FFB Oil Palm Detection

**Dibuat:** 2026-01-28
**Berdasarkan:** Catatan Dosen Penelitian

---

## Ringkasan Perubahan

| Poin | Perubahan | Eksperimen Terdampak |
|:-----|:----------|:---------------------|
| 1 | Augmentasi seragam (geometri-only) | A.1, A.2, A.3, A.4a, A.4b |
| 2 | Reset BatchNorm stats | A.2, A.3, A.4a, A.4b |
| 3 | Late Fusion Model baru | A.5 (baru) |

**B Series (B.1, B.2):** Tidak berubah (sudah benar)

---

## Detail Perubahan

### 1. Augmentasi Seragam (Semua Eksperimen)

**Konfigurasi uniform:**

```python
augment_params = dict(
    translate=0.1,    # geometri ✅
    scale=0.5,        # geometri ✅
    fliplr=0.5,       # geometri ✅
    hsv_h=0.0,        # 0 (non-geometri)
    hsv_s=0.0,        # 0 (non-geometri)
    hsv_v=0.0,        # 0 (non-geometri)
    erasing=0.0,      # 0 (non-geometri)
    mosaic=0.0,       # 0 (non-geometri)
    mixup=0.0,        # 0 (non-geometri)
)
```

**Terdampak:** A.1, A.2, A.3, A.4a, A.4b (semua 5 seeds)

---

### 2. Reset BatchNorm Running Stats

**Untuk eksperimen depth:** A.2, A.3, A.4a, A.4b

#### A.2, A.4a (3-Channel Depth) - Dummy Input

> **Note:** API Ultralytics berubah - `build_dataloader()` tidak lagi menerima parameter `imgsz`.
> Solusi untuk A.2 dan A.4a: Menggunakan dummy input.

```python
def reset_bn_stats_dummy(model, num_batches=10, batch_size=16, device='cuda'):
    """
    Reset running stats BatchNorm menggunakan dummy input.
    Dipanggil setelah load pretrained weights dan model.to(device).
    """
    model.train()

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = 0.1

    dummy_input = torch.randn(batch_size, 3, 640, 640).to(device)
    with torch.no_grad():
        for i in range(num_batches):
            _ = model(dummy_input)

    return model
```

#### A.3, A.4b (4-Channel RGBD) - Real Training Images

> **Sesuai catatan dosen:** BN reset menggunakan **100 gambar training asli** (bukan dummy).
> Implementasi via callback `on_train_start` di `RGBD4ChTrainer`.

```python
class RGBD4ChTrainer(DetectionTrainer):
    def __init__(self, overrides=None):
        super().__init__(overrides=overrides)
        self.add_callback("on_train_start", self._bn_reset_callback)

    def _bn_reset_callback(self, trainer):
        """Reset BN dengan 100 gambar training asli."""
        # 1. Reset running stats
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = 0.1

        # 2. Forward pass 100 gambar training
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
- `images.float() / 255.0` - normalize uint8 ke float32 (fix ByteTensor error)
- `int(self.model.stride.max())` - fix Tensor attribute error
- `data=self.data` - fix NoneType error di build_dataset()

---

### 3. Late Fusion Model (A.5)

**Arsitektur Multi-Scale (P3, P4, P5):**

```
Input RGB (3ch)          Input Depth (3ch)
     │                         │
     ▼                         ▼
┌─────────────┐           ┌─────────────┐
│  RGB Branch │           │ Depth Branch│
│  (A.1 Frozen)│          │ (A.2 Frozen)│
│  Backbone   │           │  Backbone   │
│  Outputs:   │           │  Outputs:   │
│  P3, P4, P5 │           │  P3, P4, P5 │
└──────┬──────┘           └──────┬──────┘
       │                         │
       ├───────┬───────┬─────────┤
       │       │       │         │
       ▼       ▼       ▼         ▼
    [Concat] [Concat] [Concat]  ← Tiap scale
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

**Spesifikasi:**
- Cabang RGB: Load weights A.1, freeze 100%
- Cabang Depth: Load weights A.2, freeze 100%
- Fusion layers: 3 layer Conv2d 512→256 + BatchNorm + SiLU (P3, P4, P5)
- Detection head: YOLOv11 Detect head trainable dari awal
- Loss: v8DetectionLoss (box_loss + cls_loss + dfl_loss)

**Catatan:** Arsitektur menggunakan multi-scale (P3, P4, P5) agar kompatibel dengan YOLO Detect head yang memerlukan 3 level feature pyramid.

**Training:**
- Epoch: 100 (atau sesuai setting A.1/A.2)
- Seeds: 42, 123, 456, 789, 101

---

## Total Run Training

| Eksperimen | Seeds | Augmentasi Baru | Reset BN | Status |
|:-----------|:-----:|:---------------:|:--------:|:-------|
| A.1 RGB Only | 5 | ✅ | - | Re-run |
| A.2 Real Depth | 5 | ✅ | ✅ | Re-run |
| A.3 RGB+Real Depth | 5 | ✅ | ✅ | Re-run |
| A.4a Synthetic Depth | 5 | ✅ | ✅ | Re-run |
| A.4b RGB+Synthetic | 5 | ✅ | ✅ | Re-run |
| **A.5 Late Fusion** | **5** | ✅ | - | **Baru** |
| **Total** | **30 runs** | | | |

---

## Timeline Estimasi

| Tahap | Durasi | Keterangan |
|:------|:-------|:-----------|
| Update script augmentasi | 2-3 jam | Modifikasi 5 training script |
| Implementasi reset BN | 2-3 jam | Tambah fungsi + integrasi |
| Implementasi A.5 | 4-6 jam | Arsitektur late fusion |
| Training A.1-A.4b | 2-3 hari | 25 runs × 100 epoch |
| Training A.5 | 1 hari | 5 runs × 100 epoch |
| Evaluasi & laporan | 1 hari | Generate metrik, plot, update Results |
| **Total** | **4-6 hari** | Parallel training bisa lebih cepat |

---

## Checklist Implementasi

### Script Training

- [x] Update `train_a1_rgb.py` - augmentasi geometri-only
- [x] Update `train_a2_depth.py` - augmentasi + reset BN (dummy input)
- [x] Update `train_a3_rgbd.py` - augmentasi + reset BN (dummy input, 4ch)
- [x] Update `train_a4a_synthetic_depth.py` - augmentasi + reset BN (dummy input)
- [x] Update `train_a4b_rgbd_synthetic.py` - augmentasi + reset BN (dummy input, 4ch)
- [x] Buat `train_a5_late_fusion.py` - model baru

### Notebook v2 (Kaggle) - ✅ Semua Selesai

- [x] `train_a1_rgb_v2.ipynb` - 5 seeds, uniform aug
- [x] `train_a2_depth_v2.ipynb` - 5 seeds, uniform aug + BN reset (dummy input)
- [x] `train_a3_rgbd_v2.ipynb` - 5 seeds, uniform aug + **BN reset 100 real images** + RGBD4ChTrainer + RGBD4ChValidator
- [x] `train_a4a_synthetic_depth_v2.ipynb` - 5 seeds, uniform aug + BN reset (dummy input)
- [x] `train_a4b_rgbd_synthetic_v2.ipynb` - 5 seeds, uniform aug + **BN reset 100 real images** + RGBD4ChTrainer + RGBD4ChValidator
- [x] `train_a5_late_fusion_v2.ipynb` - 5 seeds, multi-scale fusion + proper YOLO loss

### Modul Baru

- [x] `reset_bn.py` - Fungsi reset BatchNorm stats (dummy input method)
- [x] `late_fusion_model.py` - Class LateFusionModel (multi-scale P3/P4/P5)
- [x] `late_fusion_trainer.py` - Trainer khusus A.5 dengan v8DetectionLoss

---

## Struktur Output

```
5_seed/
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
├── train-a5-late-fusion/          # BARU
│   ├── runs/detect/exp_a5_fusion_seed*/
│   └── kaggleoutput/a5_fusion_results.txt
└── Results_v3.md                   # Update dengan A.5
```

---

## Catatan Penting

1. **Bobot A.1 dan A.2:** Gunakan weights terbaik dari hasil training sebelumnya sebagai input untuk A.5.

2. **Reset BN (Critical):**
   - Hanya untuk model depth (A.2-A.4b)
   - **Wajib** panggil `model.model.to(DEVICE)` sebelum reset BN
   - Gunakan dummy input alih-alih dataloader (API berubah)

3. **A.5 Input:** Dataloader memuat RGB dan Depth secara terpisah (2 path), bukan 4-channel fused.

4. **GPU Memory:** A.5 memerlukan 2× backbone, pertimbangkan batch size 8 jika OOM.

5. **Metrics Access Pattern:**
   ```python
   # Benar (API terbaru)
   mAP50 = results.box.map50
   mAP50_95 = results.box.map

   # Salah (API lama)
   mAP50 = results.results_dict['metrics/mAP50(B)']
   ```

---

## Risiko & Mitigasi

| Risiko | Mitigasi |
|:-------|:---------|
| Training time terlalu lama | Parallel run di multiple GPU/akun Kaggle |
| OOM di A.5 | Batch size 16→8, atau pakai gradient accumulation |
| Hasil A.5 tidak lebih baik | Dokumentasikan sebagai exploration, fokus ke A.1-A.4b yang sudah solid |
| Reset BN tidak berdampak signifikan | Abaikan jika tidak ada improvement, atau jadikan ablation |

---

## Update Log

### 2026-01-29 - Implementasi Selesai

**Perubahan Teknis dari Rencana Awal:**

1. **Reset BN Method - A.2, A.4a (3-Channel)**
   - **Rencana:** Menggunakan `train_loader` dengan 100 batches
   - **Implementasi:** Menggunakan `dummy_input` (random tensors) dengan 10 batches
   - **Alasan:** API Ultralytics berubah, `build_dataloader()` tidak menerima `imgsz`
   - **Impact:** Tetap valid untuk reset running statistics, lebih cepat

2. **Reset BN Method - A.3, A.4b (4-Channel RGBD)**
   - **Rencana:** Menggunakan dummy input
   - **Implementasi:** Menggunakan **100 gambar training asli** via callback `on_train_start`
   - **Alasan:** Sesuai catatan dosen - domain adaptation lebih baik dengan real data
   - **Implementation:** `RGBD4ChTrainer` dengan `_bn_reset_callback()`

3. **A.3, A.4b Custom Trainer Architecture**
   - **RGBD4ChTrainer:**
     - `get_model()` - Auto-convert 3ch → 4ch first conv layer
     - `build_dataset()` - Override dengan `RGBDDataset` (4-channel `load_image()`)
     - `_bn_reset_callback()` - BN reset dengan 100 real training images
     - `get_validator()` - Return `RGBD4ChValidator`
   - **RGBD4ChValidator:**
     - `build_dataset()` - 4-channel dataset untuk validation
     - `setup_model()` - Ensure 4ch conversion

4. **Cell 8 Evaluation Fix**
   - **Implementasi:** `evaluate_rgbd_model()` menggunakan `RGBD4ChValidator` eksplisit
   - **Alasan:** `model.val()` default tidak handle 4-channel properly
   - **Return:** `validator.metrics` (bukan return value dari `validator()`)

5. **A.5 Arsitektur**
   - **Rencana:** Single-scale (P3 only) fusion
   - **Implementasi:** Multi-scale (P3, P4, P5) fusion
   - **Alasan:** YOLO Detect head memerlukan 3 level feature pyramid

6. **Device Transfer**
   - **Tambahan:** `model.model.to(DEVICE)` sebelum BN reset (penting!)
   - **Alasan:** Layer BN perlu di GPU sebelum forward pass

7. **Metrics Access**
   - **Update:** `results.box.map50` alih-alih `results.results_dict['metrics/mAP50(B)']`
   - **Alasan:** API Ultralytics terbaru mengubah struktur results object

**Status:** Semua notebook _v2 siap untuk training (30 runs total).
