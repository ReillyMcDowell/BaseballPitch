import cv2
import os
from ultralytics.models import YOLO

# --- CONFIG ---
VIDEO_PATH = "test_pitch_video.mp4"  # Path to test video
MODEL_PATH = "modules/release_detection/runs/detect/yolo11n_pitcher/weights/best.pt"  # Use trained pitcher model
OUTPUT_DIR = "output_pitcher_frames"
CLASS_NAMES = ['pitcher']
CLASS_COLORS = [(0, 255, 0)]  # Green for pitcher
# --------------

def run_inference():
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print(f"Could not open video: {VIDEO_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    frame_count = 0
    pitcher_detected_count = 0
    first_detection_frame = None
    first_detection_img = None

    print("Analyzing video for pitcher detection...")
    print("Green box = Pitcher detected")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Detect pitcher
        results = model(frame, conf=0.3, verbose=False)
        
        annotated = results[0].plot()
        
        # Check detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            pitcher_detected_count += 1
            # Save first detection
            if first_detection_frame is None:
                first_detection_frame = frame_count
                first_detection_img = frame.copy()
        
        # Add detection counts to display
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Pitcher detected: {pitcher_detected_count} frames", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pitcher Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_count += 1

    # Save the first detection frame
    if first_detection_img is not None:
        save_path = os.path.join(OUTPUT_DIR, f"pitcher_frame_{first_detection_frame}.jpg")
        cv2.imwrite(save_path, first_detection_img)
        print(f"\n✓ Pitcher Detection Results")
        print(f"   First detection at frame: {first_detection_frame}")
        print(f"   Saved to: {save_path}")
        print(f"\nSummary:")
        print(f"   Total frames with pitcher detected: {pitcher_detected_count}/{frame_count}")
        print(f"   Detection rate: {pitcher_detected_count/frame_count*100:.1f}%")
    else:
        print(f"\n⚠️  No pitcher detected in video")

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