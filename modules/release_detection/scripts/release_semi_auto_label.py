import cv2
import os
import torch
from ultralytics.models import YOLO

# --- CONFIG ---
VIDEO_FOLDER = "pitch_videos_trimmed"
OUTPUT_BASE = "modules/release_detection/finetuning_dataset"
POSE_MODEL = 'yolo11x-pose.pt'  # Detect people with pose keypoints
POSE_CONF = 0.25
IMG_SIZE = 1280
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Frame skip: only label every Nth frame to speed up (1 = every frame)
FRAME_SKIP = 1  # Label every Nth frame
MAX_VIDEOS = None  # Set to a number to limit videos, or None for all
# --------------

class SemiAutoLabeler:
    def __init__(self):
        self.pose_model = YOLO(POSE_MODEL)
        self.pose_model.to(DEVICE)
        self.current_frame = None
        self.current_frame_num = 0
        self.video_name = ""
        self.split = "train"
        self.labeled_count = 0
        self.skipped_count = 0
        self.detected_boxes = []
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click to select pitcher"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find which detected person was clicked
            click_point = (x, y)
            selected_box = None
            
            for box in self.detected_boxes:
                x1, y1, x2, y2 = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_box = box
                    break
            
            if selected_box:
                self.save_label(selected_box)
    
    def save_label(self, box):
        """Save YOLO format label for selected box"""
        if self.current_frame is None:
            return
            
        h, w = self.current_frame.shape[:2]
        x1, y1, x2, y2 = box
        
        # Convert to YOLO format (normalized center x, y, width, height)
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        box_width = (x2 - x1) / w
        box_height = (y2 - y1) / h
        
        # Save image
        img_name = f"{self.video_name}_{self.current_frame_num:04d}.jpg"
        img_path = f"{OUTPUT_BASE}/images/{self.split}/{img_name}"
        cv2.imwrite(img_path, self.current_frame)
        
        # Save label
        label_path = f"{OUTPUT_BASE}/labels/{self.split}/{self.video_name}_{self.current_frame_num:04d}.txt"
        with open(label_path, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        self.labeled_count += 1
        print(f"✓ Labeled frame {self.current_frame_num} ({self.labeled_count} total)")
    
    def label_video(self, video_path):
        """Label all frames in a video"""
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.split = 'val' if (hash(self.video_name) % 10 == 0) else 'train'
        
        # Check if video already has labels
        label_dir = f"{OUTPUT_BASE}/labels/{self.split}"
        existing_labels = [f for f in os.listdir(label_dir) if f.startswith(self.video_name) and f.endswith('.txt')]
        if existing_labels:
            print(f"⏭️  Skipping {self.video_name} (already labeled)")
            return True
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Video: {self.video_name}")
        print(f"Split: {self.split}")
        print(f"Total frames: {total_frames} (labeling every {FRAME_SKIP} frames)")
        print(f"Using device: {DEVICE}")
        print(f"{'='*60}")
        print("\nControls:")
        print("  - Click on pitcher to select their bounding box")
        print("  - SPACE/RIGHT ARROW - Skip frame (no pitcher)")
        print("  - Q - Quit labeling session")
        print("  - N - Skip to next video")
        
        cv2.namedWindow("Semi-Auto Labeling")
        cv2.setMouseCallback("Semi-Auto Labeling", self.mouse_callback)
        
        frames_to_label = list(range(0, total_frames, FRAME_SKIP))
        
        for frame_idx in frames_to_label:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            self.current_frame = frame.copy()
            self.current_frame_num = frame_idx
            
            # Detect people in frame
            pose_results = self.pose_model.track(frame, persist=True, conf=POSE_CONF, 
                                                verbose=False, imgsz=IMG_SIZE, device=DEVICE)
            
            # Extract bounding boxes in pixel coordinates
            self.detected_boxes = []
            if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                boxes = pose_results[0].boxes.xyxy  # Get boxes in x1,y1,x2,y2 format
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                
                h, w = frame.shape[:2]
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    self.detected_boxes.append((x1, y1, x2, y2))
            
            while True:
                display_frame = self.current_frame.copy()
                
                # Draw all detected bounding boxes
                for i, (x1, y1, x2, y2) in enumerate(self.detected_boxes):
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Person {i+1}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add info overlay
                progress_idx = frames_to_label.index(frame_idx) + 1
                info_text = f"Frame {self.current_frame_num}/{total_frames} | Progress: {progress_idx}/{len(frames_to_label)}"
                cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                
                if self.detected_boxes:
                    instruction = "Click on the pitcher"
                else:
                    instruction = "No people detected - press SPACE to skip"
                cv2.putText(display_frame, instruction, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, instruction, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                cv2.imshow("Semi-Auto Labeling", display_frame)
                
                key = cv2.waitKeyEx(1)
                
                # Skip frame (no pitcher visible)
                if key in (32, 2555904, ord('s'), ord('S')):  # Space or Right arrow
                    self.skipped_count += 1
                    print(f"⏭️  Skipped frame {self.current_frame_num}")
                    break
                
                # Quit
                elif key in (ord('q'), ord('Q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                
                # Next video
                elif key in (ord('n'), ord('N')):
                    cap.release()
                    return True
                
                # Check if label was just saved (frame exists)
                img_path = f"{OUTPUT_BASE}/images/{self.split}/{self.video_name}_{self.current_frame_num:04d}.jpg"
                if os.path.exists(img_path):
                    break
        
        cap.release()
        return True
    
    def run(self):
        """Main labeling loop"""
        videos = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')])
        
        # Limit videos if MAX_VIDEOS is set
        if MAX_VIDEOS is not None:
            videos = videos[:MAX_VIDEOS]
        
        if not videos:
            print(f"No videos found in {VIDEO_FOLDER}")
            return
        
        print(f"Found {len(videos)} videos to label")
        print("\nStarting semi-automatic labeling session...")
        
        for i, video in enumerate(videos, 1):
            print(f"\n[Video {i}/{len(videos)}]")
            video_path = os.path.join(VIDEO_FOLDER, video)
            
            if not self.label_video(video_path):
                break  # User quit
        
        cv2.destroyAllWindows()
        print(f"\n{'='*60}")
        print(f"Labeling session complete!")
        print(f"Labeled: {self.labeled_count} frames")
        print(f"Skipped: {self.skipped_count} frames")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("1. Review labels: python modules/release_detection/scripts/release_review_labels.py")
        print("2. Train model: python modules/release_detection/scripts/release_training.py")

if __name__ == "__main__":
    labeler = SemiAutoLabeler()
    labeler.run()
