"""
Test script to verify release detection model on labeled dataset clips
"""
import torch
import json
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from release_model_arch import PitchReleaseCSN
from pytorchvideo.data.encoded_video import EncodedVideo

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
model = PitchReleaseCSN().to(device)

# Try to load trained weights
weight_path = Path(__file__).parent.parent / "runs" / "release_model_weights.pt"
if weight_path.exists():
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"✓ Loaded trained model from {weight_path}")
else:
    print("⚠️  No trained weights found - using untrained model")
    print("   Run release_fine_tune.py first to train the model")

model.eval()

# Load annotations
project_root = Path(__file__).parent.parent.parent.parent
annotations_path = project_root / "modules/release_detection/release_dataset/annotations.json"

if not annotations_path.exists():
    print(f"❌ Annotations not found at {annotations_path}")
    print("   Run release_labeler.py first to create the dataset")
    exit(1)

with open(annotations_path, 'r') as f:
    annotations = json.load(f)

print(f"\nLoaded {len(annotations)} clips from dataset")

# Test on a few positive and negative examples
positive_clips = [a for a in annotations if a['label'] == 1]
negative_clips = [a for a in annotations if a['label'] == 0]

print(f"  Positive (release): {len(positive_clips)}")
print(f"  Negative (no release): {len(negative_clips)}")

# Custom transform (same as training)
class VideoTransform:
    def __init__(self, num_frames=16):
        self.num_frames = num_frames
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    
    def __call__(self, video_tensor):
        C, T, H, W = video_tensor.shape
        if T > self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            video_tensor = video_tensor[:, indices, :, :]
        elif T < self.num_frames:
            repeat_factor = (self.num_frames + T - 1) // T
            video_tensor = video_tensor.repeat(1, repeat_factor, 1, 1)[:, :self.num_frames, :, :]
        return (video_tensor - self.mean) / self.std

transform = VideoTransform(num_frames=16)

def test_clip(annotation):
    """Test a single clip and return prediction"""
    video_path = project_root / "modules" / "release_detection" / annotation['video']
    
    if not video_path.exists():
        print(f"  ⚠️  Video not found: {video_path}")
        return None
    
    # Load video
    video = EncodedVideo.from_path(str(video_path))
    video_data = video.get_clip(start_sec=0, end_sec=video.duration)
    video_tensor = video_data['video']
    
    # Transform
    video_tensor = transform(video_tensor)
    video_tensor = video_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Prepare handedness
    handedness = torch.tensor([[annotation['handedness']]], dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        frame_logits, has_release_logits = model(video_tensor, handedness)
        has_release_probs = torch.softmax(has_release_logits, dim=1)
        frame_probs = torch.softmax(frame_logits, dim=1)
        predicted_frame = torch.argmax(frame_probs, dim=1).item()
    
    return has_release_probs[0].cpu(), predicted_frame

# Test some positive examples
print("\n" + "="*60)
print("Testing POSITIVE clips (should predict release):")
print("="*60)
for i, clip in enumerate(positive_clips[:5]):
    result = test_clip(clip)
    if result is not None:
        probs, pred_frame = result
        print(f"{i+1}. {Path(clip['video']).name}")
        print(f"   Has Release: No={probs[0]:.2%}, Yes={probs[1]:.2%}")
        print(f"   Predicted frame: {pred_frame}/15 (true: 8)")
        print(f"   {'✓ CORRECT' if probs[1] > 0.5 else '✗ WRONG'}")

# Test some negative examples
print("\n" + "="*60)
print("Testing NEGATIVE clips (should predict no release):")
print("="*60)
for i, clip in enumerate(negative_clips[:5]):
    result = test_clip(clip)
    if result is not None:
        probs, pred_frame = result
        print(f"{i+1}. {Path(clip['video']).name}")
        print(f"   Has Release: No={probs[0]:.2%}, Yes={probs[1]:.2%}")
        print(f"   Predicted frame: {pred_frame}/15")
        print(f"   {'✓ CORRECT' if probs[0] > 0.5 else '✗ WRONG'}")

print("\n" + "="*60)
print("Testing complete!")
print("="*60)
