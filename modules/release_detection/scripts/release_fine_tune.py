import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as vision_transforms
from pytorchvideo.data.encoded_video import EncodedVideo

# Import the model from the same directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from release_frame import PitchReleaseCSN

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
device = 'cuda'
model = PitchReleaseCSN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Define transforms for CSN (expects 16 frames at 224x224)
# Note: pytorchvideo transforms work differently than torchvision
from pytorchvideo.transforms import (
    UniformTemporalSubsample,
    Normalize,
)

train_transform = vision_transforms.Compose([
    UniformTemporalSubsample(16),  # Sample 16 frames
    Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
])

# Create dataset
# TODO: Replace these with your actual data
train_video_paths = ['path/to/video1.mp4', 'path/to/video2.mp4']  # Your video files
train_labels = [0, 1]  # 0 = no release, 1 = release
train_handedness = [1, 0]  # 1 = right-handed, 0 = left-handed

train_dataset = PitchReleaseDataset(
    video_paths=train_video_paths,
    labels=train_labels,
    handedness=train_handedness,
    transform=train_transform
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for epoch in range(10):
    model.train()
    for video, hand, label in train_loader:
        video, hand, label = video.to(device), hand.to(device), label.to(device)
        
        optimizer.zero_grad()
        outputs = model(video, hand)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch} finished.")