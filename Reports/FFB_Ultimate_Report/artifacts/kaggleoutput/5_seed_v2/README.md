# 5 Seed Experiments - Version 2 (Action Plan)

**Folder:** `5_seed_v2/`
**Created:** 2026-01-28
**Purpose:** Hasil re-run eksperimen dengan revisi dari dosen

---

## Perubahan dari 5_seed v1

| Aspek | v1 (Lama) | v2 (Baru) |
|:------|:----------|:----------|
| **Augmentasi** | Mixed (A.1 default, A.3/A.4b HSV=0) | Seragam geometri-only (semua) |
| **Reset BN** | Tidak ada | Ada untuk depth experiments |
| **Late Fusion** | Tidak ada | A.5 (baru) |

---

## Struktur Folder

```
5_seed_v2/
├── README.md                          # File ini
├── ACTION_PLAN.md                     # Detail rencana kerja
├── Results_v3.md                      # (akan dibuat) Hasil akhir
│
├── train_a1_rgb/                      # A.1: RGB Only
│   ├── runs/detect/exp_a1_rgb_seed{N}/
│   └── kaggleoutput/a1_rgb_results.txt
│
├── train_a2_depth/                    # A.2: Real Depth Only
│   ├── runs/detect/exp_a2_depth_seed{N}/
│   └── kaggleoutput/a2_depth_results.txt
│
├── train_a3_rgbd/                     # A.3: RGB + Real Depth (4-ch)
│   ├── runs/detect/exp_a3_rgbd_seed{N}/
│   └── kaggleoutput/a3_rgbd_results.txt
│
├── train_a4a_synthetic_depth/         # A.4a: Synthetic Depth Only
│   ├── runs/detect/exp_a4a_depth_seed{N}/
│   └── kaggleoutput/a4a_depth_results.txt
│
├── train_a4b_rgbd_synthetic/          # A.4b: RGB + Synthetic Depth (4-ch)
│   ├── runs/detect/exp_a4b_rgbd_seed{N}/
│   └── kaggleoutput/a4b_rgbd_results.txt
│
└── train_a5_late_fusion/              # A.5: Late Fusion (BARU)
    ├── runs/detect/exp_a5_fusion_seed{N}/
    └── kaggleoutput/a5_fusion_results.txt
```

---

## Konfigurasi Training (v2)

### Augmentasi (Semua Eksperimen)

```yaml
translate: 0.1    # ✅ Aktif
scale: 0.5        # ✅ Aktif
fliplr: 0.5       # ✅ Aktif
hsv_h: 0.0        # ❌ Disable
hsv_s: 0.0        # ❌ Disable
hsv_v: 0.0        # ❌ Disable
erasing: 0.0      # ❌ Disable
mosaic: 0.0       # ❌ Disable
mixup: 0.0        # ❌ Disable
```

### Reset BatchNorm (A.2, A.3, A.4a, A.4b)

- Forward pass: 100 batch training data
- Setelah: load pretrained weights
- Sebelum: training utama dimulai

### Late Fusion (A.5)

- Backbone RGB: Frozen (bobot A.1)
- Backbone Depth: Frozen (bobot A.2)
- Fusion layer: Trainable

---

## Seeds

Semua eksperimen menggunakan 5 seeds:

```python
seeds = [42, 123, 456, 789, 101]
```

---

## Status

| Eksperimen | Status | Seeds Completed |
|:-----------|:------:|:---------------:|
| A.1 RGB | ⏳ Pending | 0/5 |
| A.2 Depth | ⏳ Pending | 0/5 |
| A.3 RGBD | ⏳ Pending | 0/5 |
| A.4a Synthetic | ⏳ Pending | 0/5 |
| A.4b RGBD Synthetic | ⏳ Pending | 0/5 |
| A.5 Late Fusion | ⏳ Pending | 0/5 |

**Legend:**
- ⏳ Pending: Belum dikerjakan
- 🔄 In Progress: Sedang berjalan
- ✅ Completed: Selesai

---

## Perbandingan dengan v1

| Metrik | v1 (Lama) | v2 (Baru - Estimasi) |
|:-------|:---------:|:--------------------:|
| A.1 RGB | 0.869±0.018 | ? (dengan augmentasi lebih keras) |
| A.2 Depth | 0.748±0.038 | ? (+ reset BN) |
| A.3 RGBD | 0.842±0.013 | ? (+ reset BN) |
| A.4a Synthetic | 0.708±0.029 | ? (+ reset BN) |
| A.4b RGBD Synth | 0.813±0.023 | ? (+ reset BN) |
| **A.5 Late Fusion** | - | ? (model baru) |

---

## Note

- Folder ini untuk hasil **baru** sesuai action plan dosen
- Hasil lama tetap ada di `5_seed/` untuk referensi
- A.5 adalah eksperimen baru yang tidak ada di v1
