import os
import glob

def fix_label_files():
    """Fix label files with incorrect class indices and formatting issues"""
    
    for split in ['train', 'val', 'test']:
        label_dir = f"dataset/labels/{split}"
        if not os.path.exists(label_dir):
            continue
            
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        
        fixed_count = 0
        deleted_count = 0
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                fixed_lines = []
                needs_fix = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Replace literal \n if present
                    line = line.replace('\\n', '')
                    
                    try:
                        parts = line.split()
                        if len(parts) != 5:
                            needs_fix = True
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Fix class index: change any non-zero class to 0
                        if class_id != 0:
                            class_id = 0
                            needs_fix = True
                        
                        # Validate coordinates are in [0, 1] range
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 < width <= 1 and 0 < height <= 1):
                            needs_fix = True
                            continue
                        
                        fixed_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
                        
                    except (ValueError, IndexError):
                        needs_fix = True
                        continue
                
                # If no valid lines remain, delete the file
                if not fixed_lines:
                    os.remove(label_file)
                    deleted_count += 1
                    needs_fix = True
                elif needs_fix:
                    with open(label_file, 'w') as f:
                        f.writelines(fixed_lines)
                    fixed_count += 1
                    
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
        
        print(f"{split} set: Fixed {fixed_count} files, deleted {deleted_count} empty/invalid files")

if __name__ == "__main__":
    fix_label_files()
    print("\nLabel files fixed! You can now run training.py")
