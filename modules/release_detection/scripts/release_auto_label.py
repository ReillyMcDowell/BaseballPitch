import cv2
import os
import torch
import numpy as np
from ultralytics.models import YOLO
from tqdm import tqdm

# --- CONFIG ---
VIDEO_FOLDER = "pitch_videos_trimmed"
OUTPUT_BASE = "modules/release_detection/finetuning_dataset"
# Models for detection
POSE_MODEL = 'yolo11x-pose.pt'  # Detect people with pose keypoints
POSE_CONF = 0.25
IMG_SIZE = 1280
# Auto-detect GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VIDEOS = 5 # Set to a number (e.g., 5) to limit processing, or None for all videos
# --------------

def auto_label():
    pose_model = YOLO(POSE_MODEL)
    pose_model.to(DEVICE)
    
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cpu':
        print("⚠️  Running on CPU. To enable GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    
    # Limit videos if MAX_VIDEOS is set
    if MAX_VIDEOS is not None:
        videos = videos[:MAX_VIDEOS]
    
    print(f"Auto-labeling {len(videos)} videos for pitcher detection...")
    print(f"Detecting the largest person in frame as the pitcher (Class 0)")
    
    skipped_count = 0
    processed_count = 0
    
    for vid in tqdm(videos):
        path = os.path.join(VIDEO_FOLDER, vid)
        vid_name = os.path.splitext(vid)[0]
        
        # 90/10 Train/Val split
        split = 'val' if (hash(vid_name) % 10 == 0) else 'train'
        
        # Check if video already has labels - skip if so
        label_dir = f"{OUTPUT_BASE}/labels/{split}"
        existing_labels = [f for f in os.listdir(label_dir) if f.startswith(vid_name) and f.endswith('.txt')]
        if existing_labels:
            skipped_count += 1
            continue
        
        cap = cv2.VideoCapture(path)
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Detect people with pose keypoints
            pose_results = pose_model.track(frame, persist=True, conf=POSE_CONF, verbose=False, imgsz=IMG_SIZE, device=DEVICE)
            
            # Check if any people detected
            people_detected = (pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0)
            
            if people_detected:
                img_name = f"{vid_name}_{frame_num:04d}.jpg"
                label_path = f"{OUTPUT_BASE}/labels/{split}/{vid_name}_{frame_num:04d}.txt"
                
                # Save the image
                cv2.imwrite(f"{OUTPUT_BASE}/images/{split}/{img_name}", frame)
                
                # Get bounding boxes
                if pose_results[0].boxes is not None and pose_results[0].boxes.xywhn is not None:
                    boxes = pose_results[0].boxes.xywhn
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().numpy()
                    
                    # Find the largest person (likely the pitcher)
                    areas = [box[2] * box[3] for box in boxes]
                    largest_idx = areas.index(max(areas))
                    pitcher_box = boxes[largest_idx]
                    
                    # Save label with class 0 (pitcher)
                    with open(label_path, "w") as f:
                        f.write(f"0 {pitcher_box[0]:.6f} {pitcher_box[1]:.6f} {pitcher_box[2]:.6f} {pitcher_box[3]:.6f}\n")
            
            frame_num += 1
        cap.release()
        processed_count += 1
    
    print(f"\nProcessed: {processed_count} videos")
    print(f"Skipped (already labeled): {skipped_count} videos")

if __name__ == "__main__":
    auto_label()
    print("\nLabeling Complete!")
    print("Review images using: python modules/release_detection/scripts/release_review_labels.py")