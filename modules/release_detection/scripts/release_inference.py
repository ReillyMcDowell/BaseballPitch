import torch
import cv2
from ultralytics import YOLO  # type: ignore
import numpy as np
import os

# Import the model from the same directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from release_model_arch import PitchReleaseCSN

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output configuration
OUTPUT_DIR = "output_release_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_pipeline(video_path, is_right_handed, model_path='runs/pose/yolo11n_pitcher_pose4/weights/best.pt'):
    # 1. Setup
    detector = YOLO(model_path)
    classifier = PitchReleaseCSN().to(device)
    
    # Try to load trained weights if they exist
    weight_path = Path(__file__).parent.parent / "runs" / "release_model_weights.pt"
    if weight_path.exists():
        classifier.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✓ Loaded trained model from {weight_path}")
    else:
        print("⚠️  No trained weights found, using untrained model")
    
    classifier.eval() # Set to eval mode
    
    cap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem
    buffer = [] # Store last 16 cropped frames
    WINDOW_SIZE = 16
    
    frame_idx = 0
    release_detected = False
    release_frame_idx = None
    
    print(f"Processing video: {video_name}")
    print(f"Pitcher handedness: {'Right' if is_right_handed else 'Left'}")
    print(f"Device: {device}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # A. Detect Pitcher
        results = detector(frame, verbose=False)
        if results[0].boxes:
            # Get best box
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # B. Crop & Resize
            crop = frame[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (224, 224))
            
            # Transform to Tensor (Normalize, Channel First)
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1] and convert to tensor
            crop_tensor = torch.from_numpy(crop_rgb).float() / 255.0
            # Normalize with ImageNet stats
            crop_tensor = (crop_tensor - torch.tensor([0.45, 0.45, 0.45])) / torch.tensor([0.225, 0.225, 0.225])
            # Change from (H, W, C) to (C, H, W)
            crop_tensor = crop_tensor.permute(2, 0, 1)
            
            # C. Buffer Management
            buffer.append(crop_tensor)
            if len(buffer) > WINDOW_SIZE:
                buffer.pop(0)
            
            # D. Classify if buffer is full
            if len(buffer) == WINDOW_SIZE and not release_detected:
                # Stack to [1, 3, 16, 224, 224]
                input_clip = torch.stack(buffer, dim=1).unsqueeze(0).to(device)
                hand_tensor = torch.tensor([[1.0 if is_right_handed else 0.0]]).to(device)
                
                with torch.no_grad():
                    frame_logits, has_release_logits = classifier(input_clip, hand_tensor)
                    has_release_probs = torch.softmax(has_release_logits, dim=1)
                    frame_probs = torch.softmax(frame_logits, dim=1)
                    
                if has_release_probs[0][1] > 0.8: # Threshold for "Release" class
                    release_detected = True
                    predicted_frame = torch.argmax(frame_probs, dim=1).item()
                    # The predicted frame is relative to the buffer (0-15)
                    # Actual frame in video = current_frame_idx - WINDOW_SIZE + predicted_frame + 1
                    release_frame_idx = frame_idx - WINDOW_SIZE + predicted_frame + 1
                    
                    print(f"\n✓ Release detected!")
                    print(f"   Confidence: {has_release_probs[0][1]:.1%}")
                    print(f"   Predicted frame in window: {predicted_frame}/15")
                    print(f"   Actual video frame: {release_frame_idx}")
                    
                    # Save the release frame
                    output_path = os.path.join(OUTPUT_DIR, f"{video_name}_release_frame_{release_frame_idx}.jpg")
                    cv2.imwrite(output_path, frame)
                    print(f"   Saved to: {output_path}")
                    
        frame_idx += 1
    
    cap.release()
    print(f"\nProcessing complete. Total frames: {frame_idx}")
    
    if not release_detected:
        print("⚠️  No release point detected in video")
    
    return release_frame_idx

if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    VIDEO_PATH = "pitch_videos/PitchType-CH_Zone-4_PlayID-6ad381e6-48b4-34b7-8531-82318f1992e0_Date-2025-09-26.mp4"
    IS_RIGHT_HANDED = True  # Set to True for right-handed pitchers
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"⚠️  Video not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH to point to a valid video file")
    else:
        run_pipeline(VIDEO_PATH, IS_RIGHT_HANDED)