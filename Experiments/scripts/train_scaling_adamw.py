from ultralytics import YOLO
import torch

def train():
    # 1. Load the model (Scaling: Nano -> Small)
    # Using 'yolo11s.pt' instead of 'yolo11n.pt' to increase model capacity
    model = YOLO('yolo11s.pt') 

    # 2. Train with new strategy
    # - epochs=100: double the previous 50 to allow fine-grained convergence
    # - optimizer='AdamW': often improved convergence for custom datasets
    # - lr0=0.001: typical starting LR for AdamW (lower than SGD's 0.01)
    results = model.train(
        data='D:/Work/Assisten Dosen/Anylabel/Experiments/ffb_localization.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='exp_scaling_adamw_s_100e',
        project='runs/detect',
        optimizer='AdamW',
        lr0=0.001,
        patience=20,  # Early stopping if no improvement for 20 epochs
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        exist_ok=True
    )

    print("Training finished!")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    train()
