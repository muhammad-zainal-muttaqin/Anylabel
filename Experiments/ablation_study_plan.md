# FFB Detection - Ablation Study Plan

## Tujuan
Mengisolasi variabel-variabel eksperimen (Model Size, Optimizer, Training Duration) untuk memahami faktor dominan yang meningkatkan performa deteksi FFB.

## 🏆 Ablation Leaderboard (Test Set - Final)

| ID | Eksperimen | Model | Optimizer | Epochs | **mAP50** | **mAP50-95** | Delta |
|:--:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 4** | **🏆 Gap 4 (Champion)** | **Small** | **SGD** | 300* | **0.875** | **0.433** 🥇 | **+0.063** |
| **Exp 1** | Gap 1 (Scaling) | Small | SGD | 50 | 0.899 | 0.418 🥈 | +0.048 |
| **Exp 2** | Gap 2 (Optimizer) | Nano | AdamW | 50 | 0.860 | 0.391 🥉 | +0.021 |
| **Exp 5** | Gap 5 (AdamW Long) | Small | AdamW | 300 | 0.833 | 0.374 | +0.004 |
| **A.1** | **Baseline** | Nano | SGD | 50 | 0.873 | 0.370 | - |
| **Exp 3** | Gap 3 (Duration) | Nano | SGD | 300* | 0.849 | 0.363 | -0.007 |

*\* EarlyStopping aktif (patience=100).*

**Key Takeaways (Test Set Evaluation):**
1.  **🏆 Champion:** **Small + SGD (300 Epochs)** memberikan skor tertinggi (**0.433** mAP50-95).
2.  **⚠️ Overfitting Alert:** Gap 5 (AdamW 300e) menunjukkan **generalization gap 9pp** (val: 0.466 → test: 0.374) — indikasi overfitting!
3.  **SGD > AdamW:** Untuk dataset kecil, SGD lebih robust — Gap 4 hanya drop ~4pp saat val→test, vs Gap 5 drop ~9pp.
4.  **Efficiency King:** **Small + SGD (50 Epochs)** adalah sweet spot untuk prototyping (0.418 dengan 1/6 training cost).
5.  **Duration Trade-off:** Training lebih lama tidak selalu lebih baik — Gap 3 (300e) lebih buruk dari Baseline (50e).

## Rencana Eksperimen (Filling the Gaps)

### 1. Isolasi Model Size (Exp 1)
Tujuan: Melihat apakah mengganti backbone ke Small (tanpa ubah optimizer/epoch) memberikan dampak signifikan.
*   **Status:** ✅ **Completed** (Result: **0.469**)
*   **Config:**
    *   Model: `yolo11s.pt` (Small)
    *   Optimizer: `auto` (Default/SGD)
    *   Epochs: 50
    *   Name: `exp_gap1_small_sgd_50e`
*   **Insight:** Upgrade ke Small memberikan boost terbesar (+9.9% dari baseline). SGD > AdamW untuk Small di setting ini.

### 2. Isolasi Optimizer (Exp 2)
Tujuan: Melihat apakah AdamW adalah faktor utama ("magic bullet") bahkan pada model Nano.
*   **Status:** ✅ **Completed** (Result: **0.438**)
*   **Config:**
    *   Model: `yolo11n.pt` (Nano)
    *   Optimizer: `AdamW` (lr0=0.001)
    *   Epochs: 50
    *   Name: `exp_gap2_nano_adamw_50e`
*   **Insight:** AdamW sangat efektif untuk Nano (+6.8%) dibanding SGD, mendekati performa Small.

### 3. Isolasi Durasi Training (Exp 3)
Tujuan: Melihat apakah model Nano sebenarnya hanya butuh waktu lebih lama untuk konvergen.
*   **Status:** ✅ **Completed** (Result: **0.417**)
*   **Config:**
    *   Model: `yolo11n.pt` (Nano)
    *   Optimizer: `auto` (Default/SGD)
    *   Epochs: 300 (Stopped at 36)
    *   Name: `exp_gap3_nano_sgd_300e`
*   **Insight:** Menambah durasi SGD membantu (+4.7%), tapi masih kalah efisien dibanding switch ke AdamW (Exp 2: +6.8%).

### 4. The Final Push (Gap 4)
Tujuan: Memaksimalkan konfigurasi terbaik (Small + SGD) dengan durasi penuh.
*   **Status:** ✅ **Completed** (Result: **0.477**)
*   **Config:**
    *   Model: `yolo11s.pt` (Small)
    *   Optimizer: `auto` (SGD)
    *   Epochs: 300 (Stopped at 243)
    *   Name: `exp_gap4_small_sgd_300e`
*   **Insight:** Peningkatan performa terjadi (+0.8%), namun cost waktu training naik 6x lipat.

### 5. ⚠️ AdamW Long Training (Gap 5) — OVERFITTING DETECTED
Tujuan: Menguji ulang Small+AdamW dengan patience yang proper (100 vs 20).
*   **Status:** ✅ **Completed** — ⚠️ **OVERFITTING DETECTED!**
*   **Config:**
    *   Model: `yolo11s.pt` (Small)
    *   Optimizer: `AdamW` (lr0=0.001)
    *   Epochs: 300 (Full, tidak early stop)
    *   Patience: 100
    *   Seeds: 42, 123
    *   Name: `exp_gap5_small_adamw_300e`
*   **Validation vs Test Set Comparison:**
    | Seed | Val mAP50-95 | **Test mAP50-95** | Gap |
    |:----:|:------------:|:-----------------:|:---:|
    | 42 | 0.464 | 0.367 | **-9.7pp** |
    | 123 | 0.468 | 0.380 | **-8.8pp** |
    | **Avg** | 0.466 | **0.374** | **-9.2pp** |
*   **⚠️ Temuan Overfitting:**
    1.  **Generalization Gap Besar:** mAP50-95 turun ~9pp dari validation ke test set.
    2.  **Training Terlalu Lama:** Loss menurun tapi test performance tidak membaik.
    3.  **SGD Lebih Robust:** Gap 4 (SGD) hanya drop ~4pp, sedangkan Gap 5 (AdamW) drop ~9pp.
    4.  **Regularisasi Kurang:** AdamW dengan weight decay 0.0005 tidak cukup untuk dataset kecil.

## Kesimpulan Akhir & Rekomendasi
1.  **Deployment Configuration:** Gunakan **YOLOv11 Small + SGD**.
2.  **Training Strategy:**
    *   Untuk hasil cepat (prototyping): Train **50 Epochs** (mAP50-95 **0.418** test set).
    *   Untuk hasil maksimal (production): Train **150-200 Epochs** (mAP50-95 **~0.433** test set).
3.  **⚠️ Hindari AdamW Long Training:** Pada dataset kecil, AdamW 300+ epochs cenderung overfit.
4.  **Nano Model:** Hanya gunakan jika resource sangat terbatas, dan **WAJIB** gunakan optimizer **AdamW**.
5.  **Selalu Evaluasi di Test Set:** Validation metrics dapat overoptimistic!

## Next Steps

### 🔥 Gap 6 — AdamW Extended (PLANNED)
Tujuan: Menguji apakah AdamW bisa menyaingi/melampaui SGD dengan epoch lebih panjang.
*   **Status:** 🚧 **In Progress**
*   **Config:**
    *   Model: `yolo11s.pt` (Small)
    *   Optimizer: `AdamW` (lr0=0.001)
    *   Epochs: **500**
    *   Patience: 100
    *   Name: `exp_gap6_small_adamw_500e`
*   **Hypothesis:** Jika Gap 5 (300e) mencapai 0.468 dan masih improving, maka 500e bisa mencapai **0.48+** dan mungkin melampaui SGD (0.477).

### Future Work
1.  **Data Augmentation:** Tuning hyperparameter augmentasi (Mosaic, Mixup) untuk mendorong skor ke 0.50+.
2.  **Hyperparameter Tuning:** Fine-tuning LR khusus untuk masing-masing optimizer.
