import cv2
import os
from ultralytics import YOLO

# --- CONFIG ---
# Point this to a video you haven't seen yet
VIDEO_PATH = "pitch_videos/PitchType-CH_Zone-4_PlayID-6ad381e6-48b4-34b7-8531-82318f1992e0_Date-2025-09-26.mp4" 
# Use your newly trained model
MODEL_PATH = "runs/detect/yolo11n_baseball/weights/best.pt" 
OUTPUT_DIR = "output_release_frames"
# --------------

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please run 3_train_student.py first.")
        # Fallback to generic for testing if custom model missing
        model = YOLO('yolo11n.pt')
        target_class = 32
    else:
        model = YOLO(MODEL_PATH)
        target_class = 0 # 'baseball' in our custom model

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return # Handle missing video gracefully

    consecutive_detections = 0
    triggered = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Track
        results = model.track(frame, persist=True, classes=[target_class], conf=0.3, verbose=False)
        
        annotated = frame.copy()
        
        if results[0].boxes.id is not None:
            consecutive_detections += 1
            annotated = results[0].plot()
        else:
            consecutive_detections = 0

        # TRIGGER LOGIC: 3 consecutive frames means ball is in flight
        if not triggered and consecutive_detections >= 3:
            print(f"🚀 Release Detected at Frame {frame_count}")
            save_path = os.path.join(OUTPUT_DIR, f"release_frame_{frame_count}.jpg")
            cv2.imwrite(save_path, frame) # Save clean frame for SAM
            triggered = True

        cv2.imshow("Baseball Tracker", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_count += 1

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