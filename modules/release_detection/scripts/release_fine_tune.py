import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as vision_transforms
from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm
import time

# Import the model from the same directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from release_model_arch import PitchReleaseCSN

# 1. Dataset Logic
class PitchReleaseDataset(Dataset):
    """Dataset for pitch release detection"""
    def __init__(self, video_paths, labels, handedness, transform=None):
        """
        Args:
            video_paths: List of paths to video files
            labels: List of labels (0 or 1 for binary classification)
            handedness: List of handedness values (0 for left, 1 for right)
            transform: Video transformations
        """
        self.video_paths = video_paths
        self.labels = labels
        self.handedness = handedness
        self.transform = transform
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # Load video
        video = EncodedVideo.from_path(self.video_paths[idx])
        video_data = video.get_clip(start_sec=0, end_sec=video.duration)
        video_tensor = video_data['video']  # Shape: (C, T, H, W)
        
        # Apply transforms if any
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        # Get handedness and label
        handedness = torch.tensor([self.handedness[idx]], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return video_tensor, handedness, label

# 2. Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cpu':
    print("❌ ERROR: GPU/CUDA not available!")
    print("   This training requires a CUDA-enabled GPU.")
    print("   Please ensure:")
    print("   1. You have an NVIDIA GPU")
    print("   2. CUDA drivers are installed")
    print("   3. PyTorch is installed with CUDA support")
    exit(1)

print(f"✓ PyTorch version: {torch.__version__}")
if hasattr(torch.version, 'cuda') and torch.version.cuda:  # type: ignore
    print(f"✓ CUDA version: {torch.version.cuda}")  # type: ignore
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

model = PitchReleaseCSN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Define custom video transform (avoids pytorchvideo transform compatibility issues)
class VideoTransform:
    def __init__(self, num_frames=16, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
        self.num_frames = num_frames
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
    
    def __call__(self, video_tensor):
        # video_tensor shape: (C, T, H, W)
        C, T, H, W = video_tensor.shape
        
        # Uniformly sample num_frames
        if T > self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            video_tensor = video_tensor[:, indices, :, :]
        elif T < self.num_frames:
            # Repeat frames if not enough
            repeat_factor = (self.num_frames + T - 1) // T
            video_tensor = video_tensor.repeat(1, repeat_factor, 1, 1)[:, :self.num_frames, :, :]
        
        # Normalize
        video_tensor = (video_tensor - self.mean) / self.std
        
        return video_tensor

train_transform = VideoTransform(num_frames=16)

if __name__ == '__main__':
    # Load data from annotations.json
    import json
    import os

    # Get the project root directory (3 levels up from this script)
    project_root = Path(__file__).parent.parent.parent.parent
    annotations_path = project_root / "modules/release_detection/release_dataset/annotations.json"
    
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Convert relative paths to absolute paths
        # Paths in annotations are like "release_dataset/videos/..." 
        # Need to add "modules/release_detection/" prefix
        train_video_paths = [
            str(project_root / "modules" / "release_detection" / a['video']) 
            for a in annotations
        ]
        train_labels = [a['label'] for a in annotations]
        train_handedness = [a['handedness'] for a in annotations]
        
        print(f"Loaded {len(annotations)} video clips from dataset")
        print(f"  Positive (release): {sum(train_labels)}")
        print(f"  Negative (no release): {len(train_labels) - sum(train_labels)}")
    else:
        # Fallback to example data if annotations don't exist yet
        print("⚠️  No annotations.json found. Using placeholder data.")
        print("   Run release_labeler.py first to create your dataset!")
        train_video_paths = [str(project_root / 'modules/release_detection/release_dataset/videos/positive/example.mp4')]
        train_labels = [1]
        train_handedness = [1]

    train_dataset = PitchReleaseDataset(
        video_paths=train_video_paths,
        labels=train_labels,
        handedness=train_handedness,
        transform=train_transform
    )

    # Create DataLoader
    # Reduced batch size to fit in 8GB GPU memory
    # Using gradient accumulation to simulate larger batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Reduced from 8 to fit in GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    num_epochs = 10
    accumulation_steps = 4  # Accumulate gradients over 4 steps (2*4=8 effective batch size)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        avg_loss = 0.0
        epoch_start = time.time()
        
        # Create progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        
        for batch_idx, (video, hand, label) in enumerate(pbar):
            video, hand, label = video.to(device), hand.to(device), label.to(device)
            
            # Forward pass
            outputs = model(video, hand)
            loss = criterion(outputs, label)
            
            # Normalize loss to account for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Only update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics (use unnormalized loss for display)
            epoch_loss += loss.item() * accumulation_steps
            avg_loss = epoch_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f}\n")