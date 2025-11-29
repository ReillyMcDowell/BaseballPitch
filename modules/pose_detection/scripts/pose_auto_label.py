import cv2
import os
import torch
from ultralytics.models import YOLO
from tqdm import tqdm

# --- CONFIG ---
VIDEO_FOLDER = "pitch_videos"
OUTPUT_BASE = "modules/pose_detection/finetuning_dataset"
# Use pose model to detect pitcher keypoints
MODEL_NAME = 'yolo11x-pose.pt' 
CONF_THRESHOLD = 0.25
IMG_SIZE = 640
# Auto-detect GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VIDEOS = None  # Set to a number (e.g., 5) to limit processing, or None for all videos
# --------------

def auto_label():
    model = YOLO(MODEL_NAME)
    model.to(DEVICE)  # Move model to GPU
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cpu':
        print("⚠️  Running on CPU. To enable GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    
    # Limit videos if MAX_VIDEOS is set
    if MAX_VIDEOS is not None:
        videos = videos[:MAX_VIDEOS]
    
    print(f"Auto-labeling {len(videos)} videos using {MODEL_NAME}...")
    
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
            
            # Detect person poses (class 0 for person)
            results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False, imgsz=IMG_SIZE, device=DEVICE)
            
            # Check if pose keypoints are detected
            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                keypoints_data = results[0].keypoints.data
                if isinstance(keypoints_data, torch.Tensor):
                    keypoints = keypoints_data.cpu().numpy()
                else:
                    keypoints = keypoints_data
                
                # Filter: Keep only the largest person (likely the pitcher)
                if results[0].boxes is not None and len(results[0].boxes.xywhn) > 0:
                    boxes = results[0].boxes.xywhn
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().numpy()
                    
                    # Find person with largest bounding box area
                    areas = [box[2] * box[3] for box in boxes]  # width * height
                    largest_idx = areas.index(max(areas))
                    
                    # Save Image
                    img_name = f"{vid_name}_{frame_num:04d}.jpg"
                    cv2.imwrite(f"{OUTPUT_BASE}/images/{split}/{img_name}", frame)
                    
                    # Save only the largest person's keypoints
                    box = boxes[largest_idx]
                    person_kpts = keypoints[largest_idx]
                    
                    # Get frame dimensions for normalization
                    h, w = frame.shape[:2]
                    
                    with open(f"{OUTPUT_BASE}/labels/{split}/{vid_name}_{frame_num:04d}.txt", "w") as f:
                        # Write: class x_center y_center width height
                        f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                        
                        # Write keypoints (x, y, visibility for each of 17 keypoints)
                        # Normalize keypoints to 0-1 range
                        for kpt in person_kpts:
                            kpt_x_norm = kpt[0] / w
                            kpt_y_norm = kpt[1] / h
                            f.write(f" {kpt_x_norm:.6f} {kpt_y_norm:.6f} {kpt[2]:.1f}")
                        f.write("\n")
            
            frame_num += 1
        cap.release()
        processed_count += 1
    
    print(f"\nProcessed: {processed_count} videos")
    print(f"Skipped (already labeled): {skipped_count} videos")

if __name__ == "__main__":
    auto_label()
    print("Labeling Complete. Please review images in modules/pose_detection/finetuning_dataset/images/train/ manually!")