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

**Implementasi:**

```python
def reset_bn_stats(model, train_loader, num_batches=100, device='cuda'):
    """
    Reset running stats BatchNorm dengan 100 batch training data.
    Dipanggil setelah load pretrained weights.
    """
    model.train()

    # Freeze semua layer kecuali BatchNorm
    for module in model.modules():
        if not isinstance(module, nn.BatchNorm2d):
            module.eval()

    # Forward pass 100 batches untuk update BN stats
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            imgs = batch['img'].to(device)
            _ = model(imgs)

    print(f"✅ Reset BN stats: {min(i+1, num_batches)} batches")
    return model
```

**Usage:**
```python
model = YOLO("yolo11n.pt")  # Load pretrained
model.model = reset_bn_stats(model.model, train_loader, num_batches=100)
# Lanjut training
```

---

### 3. Late Fusion Model (A.5)

**Arsitektur:**

```
Input RGB (3ch)          Input Depth (1-3ch)
     │                         │
     ▼                         ▼
┌─────────────┐           ┌─────────────┐
│  RGB Branch │           │ Depth Branch│
│  (A.1 Frozen)│          │ (A.2 Frozen)│
│  Backbone   │           │  Backbone   │
│  Output:    │           │  Output:    │
│  P3 (256ch) │           │  P3 (256ch) │
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
          │  (trainable)    │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │  Detection Head │
          │  (trainable)    │
          └─────────────────┘
```

**Spesifikasi:**
- Cabang RGB: Load weights A.1, freeze 100%
- Cabang Depth: Load weights A.2, freeze 100%
- Fusion layer: Conv2d 512→256 + ReLU
- Detection head: Trainable dari awal

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

- [ ] Update `train_a1_rgb.py` - augmentasi geometri-only
- [ ] Update `train_a2_depth.py` - augmentasi + reset BN
- [ ] Update `train_a3_rgbd.py` - augmentasi + reset BN
- [ ] Update `train_a4a_synthetic_depth.py` - augmentasi + reset BN
- [ ] Update `train_a4b_rgbd_synthetic.py` - augmentasi + reset BN
- [ ] Buat `train_a5_late_fusion.py` - model baru

### Notebook (Kaggle)

- [ ] Update `train_a1_rgb.ipynb`
- [ ] Update `train_a2_depth.ipynb`
- [ ] Update `train_a3_rgbd.ipynb`
- [ ] Update `train_a4a_synthetic_depth.ipynb`
- [ ] Update `train_a4b_rgbd_synthetic.ipynb`
- [ ] Buat `train_a5_late_fusion.ipynb`

### Modul Baru

- [ ] `reset_bn.py` - Fungsi reset BatchNorm stats
- [ ] `late_fusion_model.py` - Class LateFusionModel
- [ ] `late_fusion_trainer.py` - Trainer khusus A.5

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

1. **Bobot A.1 dan A.2:** Simpan hasil training lama untuk referensi A.5, tapi A.1 dan A.2 juga perlu di-re-run dengan augmentasi baru untuk fair comparison.

2. **Reset BN:** Hanya untuk model depth (A.2-A.4b). RGB-only (A.1) tidak perlu karena pretrained ImageNet sudah sesuai.

3. **A.5 Input:** Perlu memodifikasi dataloader untuk load RGB dan Depth secara terpisah (2 path), bukan 4-channel fused.

4. **GPU Memory:** A.5 late fusion memerlukan 2× backbone, perhatikan batch size (mungkin perlu turun dari 16 ke 8).

---

## Risiko & Mitigasi

| Risiko | Mitigasi |
|:-------|:---------|
| Training time terlalu lama | Parallel run di multiple GPU/akun Kaggle |
| OOM di A.5 | Batch size 16→8, atau pakai gradient accumulation |
| Hasil A.5 tidak lebih baik | Dokumentasikan sebagai exploration, fokus ke A.1-A.4b yang sudah solid |
| Reset BN tidak berdampak signifikan | Abaikan jika tidak ada improvement, atau jadikan ablation |

---

*Action Plan ini bersifat living document, update sesuai perkembangan.*
