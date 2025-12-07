import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as vision_transforms
from pytorchvideo.data.encoded_video import EncodedVideo
from tqdm import tqdm
import time
import json
import os

# Import the model from the same directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from release_model_arch import PitchReleaseCSN

# 1. Dataset Logic
class PitchReleaseDataset(Dataset):
    """Dataset for pitch release detection"""
    def __init__(self, video_paths, labels, handedness, release_frames=None, transform=None):
        """
        Args:
            video_paths: List of video paths (relative paths from annotations)
            labels: List of binary labels (0=no release, 1=has release)
            handedness: List of handedness values (0=left, 1=right)
            release_frames: List of frame indices where release occurs (0-15), -1 for negative clips
            transform: Video transform function
        """
        self.video_paths = video_paths
        self.labels = labels
        self.handedness = handedness
        self.release_frames = release_frames if release_frames is not None else [7] * len(video_paths)
        self.transform = transform
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        handedness = self.handedness[idx]
        release_frame = self.release_frames[idx]
        
        # Load video
        base_dir = Path(__file__).parent.parent.parent.parent
        full_path = base_dir / "modules/release_detection" / video_path
        
        video = EncodedVideo.from_path(str(full_path))
        video_data = video.get_clip(start_sec=0, end_sec=video.duration)
        video_tensor = video_data['video']
        
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, torch.tensor(label), torch.tensor([handedness]), torch.tensor(release_frame)

# 2. Training Setup
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
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    print(f"✓ CUDA version: {torch.version.cuda}")  # type: ignore
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

model = PitchReleaseCSN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Class weights to handle imbalance (82 negative, 41 positive)
class_counts = torch.tensor([82.0, 41.0])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

print(f"\nClass distribution:")
print(f"  Negative (no release): {int(class_counts[0])} clips")
print(f"  Positive (release): {int(class_counts[1])} clips")
print(f"  Class weights: No-Release={class_weights[0]:.3f}, Release={class_weights[1]:.3f}")

criterion_classification = nn.CrossEntropyLoss(weight=class_weights)
criterion_frame = nn.CrossEntropyLoss()

print("\nLoss functions:")
print("  Classification loss: Weighted CrossEntropy (for has_release)")
print("  Frame loss: CrossEntropy (for which frame has release)")

# Define custom video transform
class VideoTransform:
    def __init__(self, num_frames=16, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
        self.num_frames = num_frames
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
    
    def __call__(self, video_tensor):
        C, T, H, W = video_tensor.shape
        if T > self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            video_tensor = video_tensor[:, indices, :, :]
        elif T < self.num_frames:
            repeat_factor = (self.num_frames + T - 1) // T
            video_tensor = video_tensor.repeat(1, repeat_factor, 1, 1)[:, :self.num_frames, :, :]
        
        video_tensor = (video_tensor - self.mean) / self.std
        return video_tensor

train_transform = VideoTransform(num_frames=16)

if __name__ == '__main__':
    # Load data from annotations.json
    project_root = Path(__file__).parent.parent.parent.parent
    annotations_path = project_root / "modules/release_detection/release_dataset/annotations.json"
    
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        all_video_paths = []
        all_labels = []
        all_handedness = []
        all_release_frames = []
        
        for anno in annotations:
            all_video_paths.append(anno['video'])
            all_labels.append(anno['label'])
            all_handedness.append(anno['handedness'])
            
            # Release is always at middle frame (8) for positive clips
            if anno['label'] == 1:
                all_release_frames.append(8)
            else:
                all_release_frames.append(-1)  # No release for negative clips
        
        # Stratified train/val split (80/20)
        from sklearn.model_selection import train_test_split
        
        indices = list(range(len(all_labels)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=all_labels,
            random_state=42
        )
        
        train_video_paths = [all_video_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        train_handedness = [all_handedness[i] for i in train_idx]
        train_release_frames = [all_release_frames[i] for i in train_idx]
        
        val_video_paths = [all_video_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        val_handedness = [all_handedness[i] for i in val_idx]
        val_release_frames = [all_release_frames[i] for i in val_idx]
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_labels)} clips")
        print(f"    Positive: {sum(train_labels)}, Negative: {len(train_labels) - sum(train_labels)}")
        print(f"  Validation: {len(val_labels)} clips")
        print(f"    Positive: {sum(val_labels)}, Negative: {len(val_labels) - sum(val_labels)}")
    else:
        print("⚠️  No annotations.json found. Run release_labeler.py first!")
        exit(1)

    train_dataset = PitchReleaseDataset(
        video_paths=train_video_paths,
        labels=train_labels,
        handedness=train_handedness,
        release_frames=train_release_frames,
        transform=train_transform
    )
    
    val_dataset = PitchReleaseDataset(
        video_paths=val_video_paths,
        labels=val_labels,
        handedness=val_handedness,
        release_frames=val_release_frames,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    num_epochs = 30
    accumulation_steps = 4
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        train_correct_cls = 0
        train_correct_frame = 0
        train_total = 0
        train_positive_count = 0
        
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        optimizer.zero_grad()
        for batch_idx, (videos, labels, handedness, release_frames) in enumerate(progress_bar):
            videos = videos.to(device)
            labels = labels.to(device)
            handedness = handedness.to(device)
            release_frames = release_frames.to(device)
            
            # Forward pass
            frame_logits, has_release_logits = model(videos, handedness)
            
            # Classification loss (has release or not)
            loss_cls = criterion_classification(has_release_logits, labels)
            
            # Frame loss (only for positive samples)
            positive_mask = labels == 1
            loss_frame = 0
            if positive_mask.sum() > 0:
                loss_frame = criterion_frame(
                    frame_logits[positive_mask],
                    release_frames[positive_mask]
                )
                train_positive_count += positive_mask.sum().item()
            
            # Combined loss
            loss = loss_cls + (0.5 * loss_frame if isinstance(loss_frame, torch.Tensor) else 0)
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Track metrics
            train_loss += loss.item() * accumulation_steps
            
            # Classification accuracy
            _, pred_cls = torch.max(has_release_logits, 1)
            train_correct_cls += (pred_cls == labels).sum().item()
            train_total += labels.size(0)
            
            # Frame accuracy (only for positive samples)
            if positive_mask.sum() > 0:
                _, pred_frame = torch.max(frame_logits[positive_mask], 1)
                train_correct_frame += (pred_frame == release_frames[positive_mask]).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100*train_correct_cls/train_total:.1f}%'
            })
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_correct_cls = 0
        val_correct_frame = 0
        val_total = 0
        val_positive_count = 0
        
        with torch.no_grad():
            for videos, labels, handedness, release_frames in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                handedness = handedness.to(device)
                release_frames = release_frames.to(device)
                
                frame_logits, has_release_logits = model(videos, handedness)
                
                loss_cls = criterion_classification(has_release_logits, labels)
                
                positive_mask = labels == 1
                loss_frame = 0
                if positive_mask.sum() > 0:
                    loss_frame = criterion_frame(
                        frame_logits[positive_mask],
                        release_frames[positive_mask]
                    )
                    val_positive_count += positive_mask.sum().item()
                
                loss = loss_cls + (0.5 * loss_frame if isinstance(loss_frame, torch.Tensor) else 0)
                val_loss += loss.item()
                
                _, pred_cls = torch.max(has_release_logits, 1)
                val_correct_cls += (pred_cls == labels).sum().item()
                val_total += labels.size(0)
                
                if positive_mask.sum() > 0:
                    _, pred_frame = torch.max(frame_logits[positive_mask], 1)
                    val_correct_frame += (pred_frame == release_frames[positive_mask]).sum().item()
        
        # Compute metrics
        epoch_time = time.time() - epoch_start_time
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100 * train_correct_cls / train_total
        train_frame_acc = 100 * train_correct_frame / train_positive_count if train_positive_count > 0 else 0
        
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * val_correct_cls / val_total
        val_frame_acc = 100 * val_correct_frame / val_positive_count if val_positive_count > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_loss_avg:.4f}, Acc={train_acc:.1f}%, Frame Acc={train_frame_acc:.1f}%")
        print(f"  Val:   Loss={val_loss_avg:.4f}, Acc={val_acc:.1f}%, Frame Acc={val_frame_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = Path(__file__).parent.parent / "runs"
            output_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = output_dir / "release_model_best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved (val_acc={val_acc:.1f}%)")
    
    # Save final model
    output_dir = Path(__file__).parent.parent / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = output_dir / "release_model_weights.pt"
    best_model_path = output_dir / "release_model_best.pt"
    
    torch.save(model.state_dict(), model_save_path)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Last model saved to: {model_save_path}")
    print(f"  Best model saved to: {best_model_path} (val_acc={best_val_acc:.1f}%)")
    print(f"{'='*60}")
