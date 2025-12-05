import torch
import cv2
from ultralytics import YOLO  # type: ignore
import numpy as np

# Import the model from the same directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from release_frame import PitchReleaseCSN

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_pipeline(video_path, is_right_handed):
    # 1. Setup
    detector = YOLO('runs/detect/train/weights/best.pt') # Your custom Nano
    classifier = PitchReleaseCSN().to(device)
    classifier.eval() # Set to eval mode
    
    cap = cv2.VideoCapture(video_path)
    buffer = [] # Store last 16 cropped frames
    WINDOW_SIZE = 16
    
    frame_idx = 0
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
            if len(buffer) == WINDOW_SIZE:
                # Stack to [1, 3, 16, 224, 224]
                input_clip = torch.stack(buffer, dim=1).unsqueeze(0).to(device)
                hand_tensor = torch.tensor([[1.0 if is_right_handed else 0.0]]).to(device)
                
                with torch.no_grad():
                    logits = classifier(input_clip, hand_tensor)
                    probs = torch.softmax(logits, dim=1)
                    
                if probs[0][1] > 0.8: # Threshold for "Release" class
                    print(f"⚾ RELEASE DETECTED at Frame {frame_idx}")
                    
        frame_idx += 1
    
    cap.release()
    print(f"Processing complete. Total frames: {frame_idx}")