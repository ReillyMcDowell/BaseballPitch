import torch
from ultralytics.models import YOLO

def train():
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    if device == 'cpu':
        print("⚠️  GPU not available. Training will be slow on CPU.")
    
    # Load the object detection model (not pose)
    model = YOLO('yolo11n.pt')
    
    # Train on custom pitcher detection dataset
    results = model.train(
        data='modules/release_detection/scripts/release_data.yaml',
        epochs=50,
        imgsz=640,
        plots=True,
        batch=16,
        name='yolo11n_pitcher',
        device=device,  # Use GPU if available
        project='modules/release_detection/runs/detect'
    )

if __name__ == "__main__":
    train()