import cv2
import os
import glob

# --- CONFIG ---
DATASET_BASE = "modules/pose_detection/finetuning_dataset"
SPLIT = "train"  # "train" or "val" or "test"
# --------------

# COCO keypoint pairs for skeleton drawing
KEYPOINT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def draw_yolo_pose(img, label_line):
    """Draw pose keypoints and skeleton from YOLO format label"""
    h, w = img.shape[:2]
    label_line = label_line.replace('\\n', '').strip()
    parts = label_line.strip().split()
    if len(parts) < 5:
        return
    
    # YOLO pose format: class x_center y_center width height kpt1_x kpt1_y kpt1_v ...
    cls, x_c, y_c, box_w, box_h = map(float, parts[:5])
    
    # Convert box to pixel coordinates
    x_c *= w
    y_c *= h
    box_w *= w
    box_h *= h
    
    x1 = int(x_c - box_w/2)
    y1 = int(y_c - box_h/2)
    x2 = int(x_c + box_w/2)
    y2 = int(y_c + box_h/2)
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "Pitcher", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw keypoints if available (17 keypoints * 3 values = 51 additional values)
    if len(parts) >= 56:  # 5 + 51
        keypoints = []
        for i in range(5, len(parts), 3):
            if i+2 < len(parts):
                kpt_x = float(parts[i]) * w
                kpt_y = float(parts[i+1]) * h
                kpt_v = float(parts[i+2])
                keypoints.append((int(kpt_x), int(kpt_y), kpt_v))
        
        # Draw keypoint connections (skeleton)
        for connection in KEYPOINT_CONNECTIONS:
            if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                kpt1 = keypoints[connection[0]]
                kpt2 = keypoints[connection[1]]
                if kpt1[2] > 0 and kpt2[2] > 0:  # Both visible
                    cv2.line(img, (kpt1[0], kpt1[1]), (kpt2[0], kpt2[1]), (255, 100, 0), 3)
        
        # Draw keypoints
        for idx, kpt in enumerate(keypoints):
            if kpt[2] > 0:  # Visible
                # Highlight important arm keypoints
                color = (0, 255, 255) if idx in [6, 8, 10] else (0, 0, 255)  # Yellow for right arm
                cv2.circle(img, (kpt[0], kpt[1]), 5, color, -1)
                cv2.circle(img, (kpt[0], kpt[1]), 6, (255, 255, 255), 1)  # White outline

def review_labels():
    img_dir = os.path.join(DATASET_BASE, "images", SPLIT)
    label_dir = os.path.join(DATASET_BASE, "labels", SPLIT)
    
    # Get all images
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    if not images:
        print(f"No images found in {img_dir}")
        return
    
    print(f"Found {len(images)} images in {SPLIT} set")
    print("\nControls:")
    print("  SPACE/RIGHT ARROW/S - Next image")
    print("  LEFT ARROW/A - Previous image")
    print("  D - Delete current image and label")
    print("  Q/ESC - Quit")
    print("\n" + "="*50)
    
    idx = 0
    deleted_count = 0
    
    while idx < len(images):
        img_path = images[idx]
        img_name = os.path.basename(img_path)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            idx += 1
            continue
        
        # Draw poses if label exists
        has_label = False
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    draw_yolo_pose(img, line)
                has_label = True
        
        # Add info overlay
        info_text = f"[{idx+1}/{len(images)}] {img_name}"
        if not has_label:
            info_text += " (NO LABEL)"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Show image
        cv2.imshow("Label Review", img)
        
        # Wait for key (use waitKeyEx for reliable arrow keys on Windows)
        key = cv2.waitKeyEx(0)

        # Quit
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break
        # Delete current image and label
        elif key in (ord('d'), ord('D')):  # Delete
            # Delete both image and label
            os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            images.pop(idx)
            deleted_count += 1
            print(f"Deleted: {img_name}")
            idx = max(0, idx)  # Stay at same position
        # Next image
        elif key in (
            32,               # Space
            2555904,          # Right arrow (Windows)
            83,               # Right arrow (Linux/X11)
            ord('s'), ord('S')  # 'S' key
        ):
            idx = min(idx + 1, len(images) - 1)
        # Previous image
        elif key in (
            2424832,          # Left arrow (Windows)
            81,               # Left arrow (Linux/X11)
            8,                # Backspace
            ord('a'), ord('A')  # 'A' key
        ):
            idx = max(idx - 1, 0)
    
    cv2.destroyAllWindows()
    print(f"\nReview complete!")
    print(f"Deleted: {deleted_count} images")
    print(f"Remaining: {len(images)} images")

if __name__ == "__main__":
    review_labels()
