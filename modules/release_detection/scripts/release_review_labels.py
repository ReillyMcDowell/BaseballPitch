import cv2
import os
import glob

# --- CONFIG ---
DATASET_BASE = "modules/release_detection/finetuning_dataset"
SPLIT = "train"  # "train" or "val" or "test"
# --------------

CLASS_NAMES = ['pitcher']
CLASS_COLORS = [(0, 255, 0)]  # Green for pitcher

def draw_yolo_box(img, label_line):
    """Draw bounding box from YOLO format label"""
    h, w = img.shape[:2]
    label_line = label_line.replace('\\n', '').strip()
    parts = label_line.strip().split()
    if len(parts) < 5:
        return
    
    # YOLO format: class x_center y_center width height
    cls = int(parts[0])
    x_c, y_c, box_w, box_h = map(float, parts[1:5])
    
    # Convert box to pixel coordinates
    x_c *= w
    y_c *= h
    box_w *= w
    box_h *= h
    
    x1 = int(x_c - box_w/2)
    y1 = int(y_c - box_h/2)
    x2 = int(x_c + box_w/2)
    y2 = int(y_c + box_h/2)
    
    # Get class info
    class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
    color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (255, 255, 255)
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label_text = class_name
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
    cv2.putText(img, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
        
        # Draw bounding boxes if label exists
        has_label = False
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    draw_yolo_box(img, line)
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
