import cv2
import os
from ultralytics.models import YOLO

# --- CONFIG ---
VIDEO_PATH = "pitch_videos/PitchType-CH_Zone-7_PlayID-19c90630-0268-3254-9276-25b5739623a2_Date-2025-09-24.mp4" 
MODEL_PATH = "runs/pose/yolo11n_pitcher_pose4/weights/best.pt"  # Use trained pitcher model
OUTPUT_DIR = "output_release_frames"
# Keypoint indices (COCO format)
RIGHT_SHOULDER = 6
RIGHT_ELBOW = 8
RIGHT_WRIST = 10
# --------------

def run_inference():
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print(f"Could not open video: {VIDEO_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    triggered = False
    frame_count = 0
    max_arm_extension = 0
    best_release_frame = None
    best_frame_num = 0

    print("Analyzing pitcher pose to detect release point...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Detect pose
        results = model(frame, conf=0.3, verbose=False)
        
        annotated = results[0].plot()
        
        # Check for keypoints and filter to largest person (pitcher)
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # If multiple people detected, get the largest one (pitcher)
            if results[0].boxes is not None and len(results[0].boxes.xywh) > 0:
                boxes = results[0].boxes.xywh.cpu().numpy()
                # Find largest bounding box (width * height)
                areas = [box[2] * box[3] for box in boxes]
                largest_idx = areas.index(max(areas))
                keypoints = results[0].keypoints.data[largest_idx].cpu().numpy()
            else:
                keypoints = results[0].keypoints.data[0].cpu().numpy()  # Fallback to first person
            
            # Get arm keypoints
            if len(keypoints) >= 17:
                shoulder = keypoints[RIGHT_SHOULDER]
                elbow = keypoints[RIGHT_ELBOW]
                wrist = keypoints[RIGHT_WRIST]
                
                # Check if keypoints are visible
                if shoulder[2] > 0.5 and elbow[2] > 0.5 and wrist[2] > 0.5:
                    # Calculate arm extension (shoulder to wrist distance)
                    arm_length = ((wrist[0] - shoulder[0])**2 + (wrist[1] - shoulder[1])**2)**0.5
                    
                    # Track maximum extension (likely release point)
                    if arm_length > max_arm_extension:
                        max_arm_extension = arm_length
                        best_release_frame = frame.copy()
                        best_frame_num = frame_count
                    
                    # Draw arm extension info
                    cv2.putText(annotated, f"Arm Extension: {arm_length:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Pitcher Pose Analysis", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_count += 1

    # Save the frame with maximum arm extension as release point
    if best_release_frame is not None:
        save_path = os.path.join(OUTPUT_DIR, f"release_frame_{best_frame_num}.jpg")
        cv2.imwrite(save_path, best_release_frame)
        print(f"🚀 Release Detected at Frame {best_frame_num}")
        print(f"   Saved to: {save_path}")
        print(f"   Max arm extension: {max_arm_extension:.1f}")
    else:
        print("⚠️  No pitcher pose detected in video")

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