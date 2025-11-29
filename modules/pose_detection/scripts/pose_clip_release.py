import cv2
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "pitch_videos/PitchType-CH_Zone-4_PlayID-6ad381e6-48b4-34b7-8531-82318f1992e0_Date-2025-09-26.mp4"            # Path to your input video
OUTPUT_DIR = "release_images"       # Where to save the "Triggered" release frame
CONF_THRESHOLD = 0.25            # Confidence threshold (keeping low cause baseballs are fast and small)
CLASS_ID_BALL = 32                  # COCO Class ID for 'sports ball' is 32
# ---------------------

# Load model and track video
model = YOLO('best.pt')
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = model.track(source=VIDEO_PATH, show=True, save=False, classes=[CLASS_ID_BALL], conf=CONF_THRESHOLD)
for i, result in enumerate(results):
    # Only save when a sports ball (class 32) is present
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        continue

    try:
        cls = boxes.cls.detach().cpu().numpy()
    except Exception:
        # Fallback: skip if class ids are unavailable
        continue

    if (cls == CLASS_ID_BALL).any():
        annotated = result.plot()  # BGR image
        out_path = os.path.join(OUTPUT_DIR, f"frame_{i}.jpg")
        cv2.imwrite(out_path, annotated)