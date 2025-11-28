import torch
from ultralytics import YOLO

def train():
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    if device == 'cpu':
        print("⚠️  GPU not available. Training will be slow on CPU.")
    
    # Load the fast Nano student model
    model = YOLO('yolo11n.pt')
    
    # Train on custom baseball dataset
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        plots=True,
        batch=16,
        name='yolo11n_baseball',
        device=device  # Use GPU if available
    )

if __name__ == "__main__":
    train()