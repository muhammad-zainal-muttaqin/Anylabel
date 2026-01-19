from ultralytics import YOLO

# --- KONFIGURASI EKSPERIMEN (UBAH DISINI) ---
# Pilihan 1: Isolasi Model Size (Small + SGD Default + 50 Epoch)
# model_name = "yolo11s.pt"
# optimizer = "auto"
# epochs = 50
# run_name = "exp_gap1_small_sgd_50e"

# Pilihan 2: Isolasi Optimizer (Nano + AdamW + 50 Epoch)
model_name = "yolo11n.pt" 
optimizer = "AdamW"
epochs = 50
run_name = "exp_gap2_nano_adamw_50e"

# Pilihan 3: Isolasi Durasi (Nano + SGD Default + 300 Epoch)
# model_name = "yolo11n.pt"
# optimizer = "auto"
# epochs = 300
# run_name = "exp_gap3_nano_sgd_300e"
# ---------------------------------------------

# Load Model
model = YOLO(model_name)

# Train
model.train(
    data="/kaggle/working/ffb_localization.yaml",
    epochs=epochs,
    imgsz=640,
    batch=16,
    device="cuda",
    seed=42,
    optimizer=optimizer,
    lr0=0.001 if optimizer == 'AdamW' else 0.01, # Sesuaikan LR untuk AdamW vs SGD
    patience=20,
    save=True,
    project="/kaggle/working/runs/detect",
    name=run_name,
)
