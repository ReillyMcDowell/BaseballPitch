import cv2
import os
from ultralytics.models import YOLO

# --- CONFIG ---
VIDEO_PATH = "pitch_videos/PitchType-CH_Zone-7_PlayID-19c90630-0268-3254-9276-25b5739623a2_Date-2025-09-24.mp4" 
MODEL_PATH = "modules/release_detection/runs/detect/yolo11n_ball_release/weights/best.pt"  # Use trained ball release model
OUTPUT_DIR = "output_release_frames"
CLASS_NAMES = ['ball-in-hand', 'ball-released']
CLASS_COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for in-hand, Red for released
# --------------

def run_inference():
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print(f"Could not open video: {VIDEO_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    frame_count = 0
    release_frame_num = None
    release_frame_img = None
    in_hand_count = 0
    released_count = 0

    print("Analyzing video for ball release detection...")
    print("Green = Ball in hand | Red = Ball released")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Detect ball state
        results = model(frame, conf=0.3, verbose=False)
        
        annotated = results[0].plot()
        
        # Check detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                
                if cls == 0:  # Ball in hand
                    in_hand_count += 1
                elif cls == 1:  # Ball released
                    released_count += 1
                    # Save first release detection
                    if release_frame_num is None:
                        release_frame_num = frame_count
                        release_frame_img = frame.copy()
        
        # Add detection counts to display
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"In-Hand: {in_hand_count} | Released: {released_count}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Ball Release Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_count += 1

    # Save the first release detection frame
    if release_frame_img is not None:
        save_path = os.path.join(OUTPUT_DIR, f"release_frame_{release_frame_num}.jpg")
        cv2.imwrite(save_path, release_frame_img)
        print(f"\n🚀 Ball Release Detected!")
        print(f"   First release at frame: {release_frame_num}")
        print(f"   Saved to: {save_path}")
        print(f"\nSummary:")
        print(f"   Total frames with ball in-hand: {in_hand_count}")
        print(f"   Total frames with ball released: {released_count}")
    else:
        print(f"\n⚠️  No ball release detected in video")
        print(f"   Frames with ball in-hand: {in_hand_count}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Quick check to see if there are videos to test on
    if not os.path.exists(VIDEO_PATH):
        # Just grab the first one from the folder if the specific one doesn't exist
        vid_files = [f for f in os.listdir('pitch_videos') if f.endswith('.mp4')]
        if vid_files:
            VIDEO_PATH = os.path.join('pitch_videos', vid_files[0])
    
    run_inference()