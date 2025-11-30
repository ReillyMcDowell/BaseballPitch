import cv2
import os
import glob

# --- CONFIG ---
VIDEO_FOLDER = "pitch_videos_trimmed"
OUTPUT_BASE = "modules/release_detection/finetuning_dataset"
# Frame skip: only label every Nth frame to speed up (1 = every frame)
FRAME_SKIP = 3  # Label every Nth frame
MAX_VIDEOS = 1  # Set to a number to limit videos, or None for all
# --------------

class ManualLabeler:
    def __init__(self):
        self.current_video = None
        self.current_frame = None
        self.current_frame_num = 0
        self.bbox_start = None
        self.bbox_end = None
        self.drawing = False
        self.video_name = ""
        self.split = "train"
        self.labeled_count = 0
        self.skipped_count = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding box"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bbox_start = (x, y)
            self.drawing = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.bbox_end = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.bbox_end = (x, y)
            self.drawing = False
            self.save_label()
    
    def save_label(self):
        """Save YOLO format label for current frame"""
        if self.bbox_start is None or self.bbox_end is None or self.current_frame is None:
            return
            
        h, w = self.current_frame.shape[:2]
        
        # Get box coordinates (handle reversed dragging)
        x1 = min(self.bbox_start[0], self.bbox_end[0])
        y1 = min(self.bbox_start[1], self.bbox_end[1])
        x2 = max(self.bbox_start[0], self.bbox_end[0])
        y2 = max(self.bbox_start[1], self.bbox_end[1])
        
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
        
        # Reset bbox
        self.bbox_start = None
        self.bbox_end = None
    
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
        print(f"{'='*60}")
        print("\nControls:")
        print("  - Click and drag to draw bounding box around pitcher")
        print("  - SPACE/RIGHT ARROW - Skip frame (no pitcher)")
        print("  - Q - Quit labeling session")
        print("  - N - Skip to next video")
        
        cv2.namedWindow("Manual Labeling")
        cv2.setMouseCallback("Manual Labeling", self.mouse_callback)
        
        self.current_frame_num = 0
        frames_to_label = list(range(0, total_frames, FRAME_SKIP))
        
        for frame_idx in frames_to_label:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            self.current_frame = frame.copy()
            self.current_frame_num = frame_idx
            
            while True:
                display_frame = self.current_frame.copy()
                
                # Draw current bounding box if being drawn
                if self.bbox_start and self.bbox_end:
                    cv2.rectangle(display_frame, self.bbox_start, self.bbox_end, (0, 255, 0), 2)
                
                # Add info overlay
                info_text = f"Frame {self.current_frame_num}/{total_frames} | Progress: {frames_to_label.index(frame_idx)+1}/{len(frames_to_label)}"
                cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                
                cv2.putText(display_frame, "Draw box around pitcher", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, "Draw box around pitcher", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                cv2.imshow("Manual Labeling", display_frame)
                
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
                
                # If bbox was saved, move to next frame
                if self.bbox_start is None and self.bbox_end is None:
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
        print("\nStarting manual labeling session...")
        
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
    labeler = ManualLabeler()
    labeler.run()
