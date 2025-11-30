import subprocess
import os
import glob
import json

def get_video_duration(video_path):
    """Get video duration using ffprobe."""
    command = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
        return None

def trim_video(input_path, output_path, start_time, end_time):
    """
    Trims a video using FFmpeg with start and end times.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        start_time (float): Start time in seconds
        end_time (float): End time in seconds (absolute, not duration)
    """
    
    command = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', input_path,
        '-c:v', 'libx264', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
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

def batch_trim_videos(input_folder="pitch_videos", output_folder="pitch_videos_trimmed", trim_start=2.0, keep_duration=4.0):
    """
    Trim videos to keep a fixed duration after skipping the start.
    
    Args:
        input_folder: Folder containing input videos
        output_folder: Folder to save trimmed videos
        trim_start: Seconds to trim from start
        keep_duration: Seconds of video to keep after trim_start
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all video files
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    
    if not video_files:
        print(f"No MP4 files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} videos to trim")
    print(f"Skipping first {trim_start}s, keeping {keep_duration}s of video")
    print("="*50)
    
    successful = 0
    failed = 0
    
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_folder, video_name)
        
        # Calculate end time
        end_time = trim_start + keep_duration
        
        # Trim the video
        if trim_video(video_path, output_path, start_time=trim_start, end_time=end_time):
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
        trim_start=2.0,    # Skip first 2 seconds
        keep_duration=3.0  # Keep 3 seconds of video
    )