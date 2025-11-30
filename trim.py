import subprocess
import os
import glob

def trim_video(input_path, output_path, start_time, end_before=None):
    """
    Trims a video using FFmpeg.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        start_time (float): Start time in seconds
        end_before (float): Seconds to trim from end (optional)
    """
    
    # Build ffmpeg command
    if end_before and end_before > 0:
        # Trim both start and end using atrim filter
        # This approach: skip start, then cut end using trim filter
        command = [
            'ffmpeg', '-y', 
            '-ss', str(start_time),  # Skip first N seconds
            '-i', input_path,
            '-vf', f'trim=0:{-end_before},setpts=PTS-STARTPTS',  # Cut from end
            '-af', f'atrim=0:{-end_before},asetpts=PTS-STARTPTS',  # Audio trim
            '-c:v', 'libx264', '-crf', '23', 
            output_path
        ]
    else:
        # Just trim from start
        command = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', input_path,
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'copy',
            output_path
        ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✓ Trimmed: {os.path.basename(output_path)}")
        return True
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr if hasattr(e, 'stderr') else str(e)
        print(f"✗ Error trimming {os.path.basename(input_path)}")
        if stderr_output:
            # Print last line of error for debugging
            last_error = stderr_output.strip().split('\n')[-1] if stderr_output else ''
            if last_error:
                print(f"   {last_error}")
        return False

def batch_trim_videos(input_folder="pitch_videos", output_folder="pitch_videos_trimmed", trim_start=2.0, trim_end=2.0):
    """
    Trim the first and last N seconds from all videos in a folder.
    
    Args:
        input_folder: Folder containing input videos
        output_folder: Folder to save trimmed videos
        trim_start: Seconds to trim from start
        trim_end: Seconds to trim from end
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all video files
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    
    if not video_files:
        print(f"No MP4 files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} videos to trim")
    print(f"Trimming {trim_start}s from start and {trim_end}s from end")
    print("="*50)
    
    successful = 0
    failed = 0
    
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_folder, video_name)
        
        # Trim the video (start and end)
        if trim_video(video_path, output_path, start_time=trim_start, end_before=trim_end):
            successful += 1
        else:
            failed += 1
    
    print("="*50)
    print(f"\nComplete! Successfully trimmed: {successful}/{len(video_files)}")
    if failed > 0:
        print(f"Failed: {failed}")

# --- Example Usage ---
if __name__ == "__main__":
    # Batch trim all videos in pitch_videos folder
    batch_trim_videos(
        input_folder="pitch_videos",
        output_folder="pitch_videos_trimmed",
        trim_start=2.0,  # Remove first 2 seconds
        trim_end=2.0     # Remove last 2 seconds
    )