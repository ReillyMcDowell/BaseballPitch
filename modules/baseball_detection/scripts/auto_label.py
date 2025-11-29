import cv2
import os
import torch
from ultralytics.models import YOLO
from tqdm import tqdm

# --- CONFIG ---
VIDEO_FOLDER = "pitch_videos"
OUTPUT_BASE = "modules/baseball_detection/finetuning_dataset"
# Use Extra Large model as 'Teacher' to find small balls
MODEL_NAME = 'yolo11x.pt' 
CONF_THRESHOLD = 0.15
IMG_SIZE = 1280
# Auto-detect GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------------

def auto_label():
    model = YOLO(MODEL_NAME)
    model.to(DEVICE)  # Move model to GPU
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cpu':
        print("⚠️  Running on CPU. To enable GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    
    print(f"Auto-labeling {len(videos)} videos using {MODEL_NAME}...")
    
    for vid in tqdm(videos):
        path = os.path.join(VIDEO_FOLDER, vid)
        cap = cv2.VideoCapture(path)
        vid_name = os.path.splitext(vid)[0]
        frame_num = 0
        
        # 90/10 Train/Val split
        split = 'val' if (hash(vid_name) % 10 == 0) else 'train'
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Track 'sports ball' (class 32)
            results = model.track(frame, persist=True, classes=[32], conf=CONF_THRESHOLD, verbose=False, imgsz=IMG_SIZE, device=DEVICE)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # Normalize boxes - handle both tensor and numpy array cases
                boxes_data = results[0].boxes.xywhn
                if isinstance(boxes_data, torch.Tensor):
                    boxes = boxes_data.cpu().numpy()
                else:
                    boxes = boxes_data  # Already numpy array
                
                # Save Image
                img_name = f"{vid_name}_{frame_num:04d}.jpg"
                cv2.imwrite(f"{OUTPUT_BASE}/images/{split}/{img_name}", frame)
                # Save Label (Convert Class 32 -> Class 0 for custom model)
                with open(f"{OUTPUT_BASE}/labels/{split}/{vid_name}_{frame_num:04d}.txt", "w") as f:
                    for box in boxes:
                        f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
            
            frame_num += 1
        cap.release()

if __name__ == "__main__":
    # auto_label()
    print("Labeling Complete. Please review images in dataset/images/train/ manually!")
    pass