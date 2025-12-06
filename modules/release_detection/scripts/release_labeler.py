import cv2  # type: ignore
import os
import json
from pathlib import Path

# --- CONFIG ---
INPUT_FOLDER = "pitch_videos_trimmed"
OUTPUT_FOLDER = "modules/release_detection/release_dataset"
FRAMES_PER_CLIP = 16
FPS_TARGET = 60  # Standardize to 60 fps

class ReleaseDatasetCreator:
    def __init__(self):
        self.annotations = []
        self.current_video = None
        self.current_video_path = None
        self.frame_buffer = []
        self.total_frames = 0
        self.current_frame_idx = 0
        
        # Create output directories
        os.makedirs(f"{OUTPUT_FOLDER}/videos/positive", exist_ok=True)
        os.makedirs(f"{OUTPUT_FOLDER}/videos/negative", exist_ok=True)
        
    def label_video(self, video_path):
        """Interactive labeling for a video"""
        self.current_video_path = video_path
        video_name = Path(video_path).stem
        
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n{'='*60}")
        print(f"Video: {video_name}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {fps}")
        print(f"{'='*60}")
        print("\nFirst, identify the pitcher's handedness:")
        
        # Show first frame to identify handedness
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return
        
        cv2.imshow("Identify Handedness", frame)
        print("\nPress 'L' for LEFT-handed, 'R' for RIGHT-handed, 'Q' to skip video")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('l') or key == ord('L'):
                handedness = 0  # Left
                handedness_str = "LHP"
                break
            elif key == ord('r') or key == ord('R'):
                handedness = 1  # Right
                handedness_str = "RHP"
                break
            elif key == ord('q') or key == ord('Q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cv2.destroyAllWindows()
        
        # Now label release point
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        release_frame = None
        
        print(f"\nHandedness: {handedness_str}")
        print("\nNow find the RELEASE POINT:")
        print("  LEFT/RIGHT ARROW - Navigate frames")
        print("  SPACE - Mark current frame as RELEASE POINT")
        print("  P - Pause/Resume playback")
        print("  Q - Skip this video")
        print("  S - Save and continue")
        
        while True:
            # Only read new frame if we haven't set position manually
            ret, frame = cap.read()
            if not ret:
                # End of video - pause at last frame
                frame_idx = self.total_frames - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Draw frame info
            display = frame.copy()
            info = f"Frame {frame_idx}/{self.total_frames}"
            if release_frame is not None:
                info += f" | RELEASE: Frame {release_frame}"
                # Draw marker on release frame
                if frame_idx == release_frame:
                    cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), 
                                (0, 255, 0), 10)
                    cv2.putText(display, "RELEASE!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)
            cv2.imshow("Mark Release Point", display)
            
            key = cv2.waitKeyEx(30)  # Wait longer to pause playback
            
            # Space - mark release
            if key == 32:
                release_frame = frame_idx
                print(f"✓ Marked release at frame {release_frame}")
            
            # Right arrow - next frame
            elif key in (2555904, 83, ord('d'), ord('D')):
                frame_idx = min(frame_idx + 1, self.total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Left arrow - previous frame
            elif key in (2424832, 81, ord('a'), ord('A')):
                frame_idx = max(frame_idx - 1, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # P - Pause/Resume playback
            elif key in (ord('p'), ord('P')):
                print("⏸️  Paused. Use arrow keys to navigate, SPACE to mark, P to resume")
                # Wait for key without timeout
                while True:
                    pause_key = cv2.waitKeyEx(0)
                    if pause_key in (ord('p'), ord('P')):
                        print("▶️  Resumed")
                        break
                    elif pause_key == 32:  # Space during pause
                        release_frame = frame_idx
                        print(f"✓ Marked release at frame {release_frame}")
                    elif pause_key in (2555904, 83, ord('d'), ord('D')):
                        frame_idx = min(frame_idx + 1, self.total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            display = frame.copy()
                            info = f"Frame {frame_idx}/{self.total_frames}"
                            if release_frame is not None:
                                info += f" | RELEASE: Frame {release_frame}"
                                if frame_idx == release_frame:
                                    cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), 
                                                (0, 255, 0), 10)
                                    cv2.putText(display, "RELEASE!", (50, 100), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (255, 255, 255), 2)
                            cv2.imshow("Mark Release Point", display)
                    elif pause_key in (2424832, 81, ord('a'), ord('A')):
                        frame_idx = max(frame_idx - 1, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            display = frame.copy()
                            info = f"Frame {frame_idx}/{self.total_frames}"
                            if release_frame is not None:
                                info += f" | RELEASE: Frame {release_frame}"
                                if frame_idx == release_frame:
                                    cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), 
                                                (0, 255, 0), 10)
                                    cv2.putText(display, "RELEASE!", (50, 100), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (255, 255, 255), 2)
                            cv2.imshow("Mark Release Point", display)
                    elif pause_key in (ord('s'), ord('S')):
                        if release_frame is not None:
                            self.save_clips(video_path, release_frame, handedness, 
                                          handedness_str, video_name)
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        else:
                            print("⚠️  No release frame marked!")
                    elif pause_key in (ord('q'), ord('Q')):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            
            # S - Save
            elif key in (ord('s'), ord('S')):
                if release_frame is not None:
                    self.save_clips(video_path, release_frame, handedness, 
                                  handedness_str, video_name)
                    break
                else:
                    print("⚠️  No release frame marked!")
            
            # Q - Skip
            elif key in (ord('q'), ord('Q')):
                break
            
            # If no key pressed, advance to next frame
            else:
                frame_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_clips(self, video_path, release_frame, handedness, 
                   handedness_str, video_name):
        """Save positive and negative clips"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame ranges
        # Positive clip: centered on release (8 frames before, 7 after)
        pos_start = max(0, release_frame - 8)
        pos_end = min(self.total_frames, release_frame + 8)
        
        # Negative clips: before and after release
        neg1_start = max(0, release_frame - 30)
        neg1_end = max(0, release_frame - 14)
        
        neg2_start = min(self.total_frames, release_frame + 14)
        neg2_end = min(self.total_frames, release_frame + 30)
        
        # Save positive clip
        pos_path = f"{OUTPUT_FOLDER}/videos/positive/{video_name}_{handedness_str}_release.mp4"
        self.extract_clip(cap, pos_start, pos_end, pos_path, fps)
        self.annotations.append({
            "video": pos_path,
            "label": 1,
            "handedness": handedness,
            "frame_range": [pos_start, pos_end]
        })
        print(f"✓ Saved positive clip: {pos_path}")
        
        # Save negative clips
        if neg1_end - neg1_start >= FRAMES_PER_CLIP:
            neg1_path = f"{OUTPUT_FOLDER}/videos/negative/{video_name}_{handedness_str}_windup.mp4"
            self.extract_clip(cap, neg1_start, neg1_end, neg1_path, fps)
            self.annotations.append({
                "video": neg1_path,
                "label": 0,
                "handedness": handedness,
                "frame_range": [neg1_start, neg1_end]
            })
            print(f"✓ Saved negative clip: {neg1_path}")
        
        if neg2_end - neg2_start >= FRAMES_PER_CLIP:
            neg2_path = f"{OUTPUT_FOLDER}/videos/negative/{video_name}_{handedness_str}_followthrough.mp4"
            self.extract_clip(cap, neg2_start, neg2_end, neg2_path, fps)
            self.annotations.append({
                "video": neg2_path,
                "label": 0,
                "handedness": handedness,
                "frame_range": [neg2_start, neg2_end]
            })
            print(f"✓ Saved negative clip: {neg2_path}")
        
        cap.release()
    
    def extract_clip(self, cap, start_frame, end_frame, output_path, fps):
        """Extract and save a clip"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = None
        
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            out.write(frame)
        
        if out is not None:
            out.release()
    
    def run(self):
        """Process all videos"""
        videos = sorted([f for f in os.listdir(INPUT_FOLDER) 
                        if f.endswith('.mp4')])
        
        print(f"Found {len(videos)} videos to label")
        
        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] Processing {video}")
            video_path = os.path.join(INPUT_FOLDER, video)
            self.label_video(video_path)
        
        # Save annotations
        with open(f"{OUTPUT_FOLDER}/annotations.json", 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Dataset creation complete!")
        print(f"Total clips: {len(self.annotations)}")
        print(f"Positive (release): {sum(1 for a in self.annotations if a['label'] == 1)}")
        print(f"Negative (no release): {sum(1 for a in self.annotations if a['label'] == 0)}")
        print(f"Annotations saved to: {OUTPUT_FOLDER}/annotations.json")


if __name__ == "__main__":
    creator = ReleaseDatasetCreator()
    creator.run()