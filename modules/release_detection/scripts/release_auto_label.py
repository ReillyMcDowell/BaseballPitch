import cv2
import os
import torch
import numpy as np
from ultralytics.models import YOLO
from tqdm import tqdm

# --- CONFIG ---
VIDEO_FOLDER = "pitch_videos"
OUTPUT_BASE = "modules/release_detection/finetuning_dataset"
# Models for detection
POSE_MODEL = 'yolo11x-pose.pt'  # Detect pitcher pose
BALL_MODEL = 'yolo11x.pt'  # Detect baseball
POSE_CONF = 0.25
BALL_CONF = 0.15
IMG_SIZE = 1280
# Auto-detect GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_VIDEOS = None  # Set to a number (e.g., 5) to limit processing, or None for all videos
# Distance threshold (in normalized coordinates) to consider ball "in hand"
HAND_DISTANCE_THRESHOLD = 0.08  # ~8% of image width
# Keypoint indices
RIGHT_WRIST = 10
LEFT_WRIST = 9
# --------------

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def auto_label():
    pose_model = YOLO(POSE_MODEL)
    ball_model = YOLO(BALL_MODEL)
    pose_model.to(DEVICE)
    ball_model.to(DEVICE)
    
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cpu':
        print("⚠️  Running on CPU. To enable GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    
    # Limit videos if MAX_VIDEOS is set
    if MAX_VIDEOS is not None:
        videos = videos[:MAX_VIDEOS]
    
    print(f"Auto-labeling {len(videos)} videos for ball release detection...")
    print(f"Saving all pitcher frames:")
    print(f"  - Frames with ball detected -> Class 0: Ball in hand | Class 1: Ball released")
    print(f"  - Frames without ball detected -> Empty label (non-event)")
    
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
            
            h, w = frame.shape[:2]
            
            # Detect pitcher pose
            pose_results = pose_model.track(frame, persist=True, conf=POSE_CONF, verbose=False, imgsz=IMG_SIZE, device=DEVICE)
            
            # Detect baseball (class 32 = sports ball)
            ball_results = ball_model.track(frame, persist=True, classes=[32], conf=BALL_CONF, verbose=False, imgsz=IMG_SIZE, device=DEVICE)
            
            # Check if pitcher is detected
            pitcher_detected = (pose_results[0].keypoints is not None and len(pose_results[0].keypoints.data) > 0)
            ball_detected = (ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0)
            
            # Save ALL frames where pitcher is detected
            if pitcher_detected:
                img_name = f"{vid_name}_{frame_num:04d}.jpg"
                label_path = f"{OUTPUT_BASE}/labels/{split}/{vid_name}_{frame_num:04d}.txt"
                
                # Always save the image
                cv2.imwrite(f"{OUTPUT_BASE}/images/{split}/{img_name}", frame)
                
                labels_to_write = []
                
                # If ball is also detected, check if it's in hand or released
                if ball_detected:
                    # Get pitcher keypoints (largest person)
                    if pose_results[0].keypoints is not None:
                        keypoints_data = pose_results[0].keypoints.data
                        if isinstance(keypoints_data, torch.Tensor):
                            keypoints = keypoints_data.cpu().numpy()
                        else:
                            keypoints = keypoints_data
                    else:
                        keypoints = None
                    
                    if keypoints is not None and pose_results[0].boxes is not None and pose_results[0].boxes.xywhn is not None and len(pose_results[0].boxes.xywhn) > 0:
                        boxes = pose_results[0].boxes.xywhn
                        if isinstance(boxes, torch.Tensor):
                            boxes = boxes.cpu().numpy()
                        
                        # Find person with largest bounding box area
                        areas = [box[2] * box[3] for box in boxes]
                        largest_idx = areas.index(max(areas))
                        person_kpts = keypoints[largest_idx]
                        
                        # Get wrist positions (normalized coordinates)
                        right_wrist = person_kpts[RIGHT_WRIST]
                        left_wrist = person_kpts[LEFT_WRIST]
                        
                        # Get ball positions (normalized coordinates)
                        if ball_results[0].boxes is not None and ball_results[0].boxes.xywhn is not None:
                            ball_boxes_data = ball_results[0].boxes.xywhn
                            if isinstance(ball_boxes_data, torch.Tensor):
                                ball_boxes = ball_boxes_data.cpu().numpy()
                            else:
                                ball_boxes = ball_boxes_data
                            
                            # Check each detected ball
                            for ball_box in ball_boxes:
                                ball_center = np.array([ball_box[0], ball_box[1]])
                                
                                # Calculate distances to both wrists
                                dist_to_right = calculate_distance(ball_center, right_wrist[:2]) if right_wrist[2] > 0.5 else float('inf')
                                dist_to_left = calculate_distance(ball_center, left_wrist[:2]) if left_wrist[2] > 0.5 else float('inf')
                                
                                min_dist = min(dist_to_right, dist_to_left)
                                
                                # Classify: Class 0 (in hand) if close to wrist, Class 1 (released) otherwise
                                ball_class = 0 if min_dist < HAND_DISTANCE_THRESHOLD else 1
                                
                                labels_to_write.append(f"{ball_class} {ball_box[0]:.6f} {ball_box[1]:.6f} {ball_box[2]:.6f} {ball_box[3]:.6f}")
                
                # Save label file (empty if no ball detected = non-event)
                with open(label_path, "w") as f:
                    for label in labels_to_write:
                        f.write(label + "\n")
            
            frame_num += 1
        cap.release()
        processed_count += 1
    
    print(f"\nProcessed: {processed_count} videos")
    print(f"Skipped (already labeled): {skipped_count} videos")

if __name__ == "__main__":
    auto_label()
    print("\nLabeling Complete!")
    print("Review images using: python modules/release_detection/scripts/release_review_labels.py")