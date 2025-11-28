import subprocess
import os

def trim_video(input_path, output_path, start_time, duration=None, end_time=None):
    """
    Trims a video using FFmpeg.
    
    Args:
        start_time (str/float): Start time in seconds or "HH:MM:SS"
        duration (str/float): Duration to keep (optional)
        end_time (str/float): Timestamp to stop at (optional)
        
    Note: Use EITHER duration OR end_time, not both.
    """
    
    command = ['ffmpeg', '-y', '-i', input_path, '-ss', str(start_time)]
    
    if duration:
        command.extend(['-t', str(duration)])
    elif end_time:
        command.extend(['-to', str(end_time)])
    
    # Use re-encoding (-c:v libx264) for frame accuracy
    # Use '-c', 'copy' instead if speed is more important than precision
    command.extend(['-c:v', 'libx264', '-crf', '23', '-c:a', 'copy', output_path])
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Trimmed: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error trimming {input_path}: {e}")
        return False

# --- Example Usage ---
if __name__ == "__main__":
    # Trim the first 3 seconds of 'pitch.mp4'
    trim_video("pitch.mp4", "pitch_trimmed.mp4", start_time=2.5, duration=4.0)