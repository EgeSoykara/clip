import os
import subprocess

# Folder containing videos
VIDEO_FOLDER = os.path.join(os.getcwd(), "out_clips")

# Minimum duration in seconds
MIN_DURATION = 30


def get_video_duration(filepath):
    """
    Returns duration of video in seconds using ffprobe
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filepath
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except:
        return None


def delete_short_videos(folder):
    deleted_count = 0

    for filename in os.listdir(folder):
        if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            filepath = os.path.join(folder, filename)

            duration = get_video_duration(filepath)

            if duration is None:
                print(f"Could not read: {filename}")
                continue

            if duration < MIN_DURATION:
                os.remove(filepath)
                print(f"Deleted: {filename} ({duration:.2f}s)")
                deleted_count += 1
            else:
                print(f"Kept: {filename} ({duration:.2f}s)")

    print(f"\nDone. Deleted {deleted_count} videos.")


delete_short_videos(VIDEO_FOLDER)
