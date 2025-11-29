import torch
from ultralytics.models import YOLO

def train():
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    if device == 'cpu':
        print("⚠️  GPU not available. Training will be slow on CPU.")
    
    # Load the pose model
    model = YOLO('yolo11n-pose.pt')
    
    # Train on custom pitcher pose dataset
    results = model.train(
        data='modules/pose_detection/scripts/pose_data.yaml',
        epochs=50,
        imgsz=640,
        plots=True,
        batch=16,
        name='yolo11n_pitcher_pose',
        device=device  # Use GPU if available
    )

if __name__ == "__main__":
    train()