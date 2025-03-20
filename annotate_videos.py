import os
import cv2
import shutil
import pandas as pd
import subprocess

# Paths
video_folder = r"D:/Sam2/sam2/Object_present_files/videos"
csv_folder = r"D:/Sam2/sam2/Object_present_files/csv_folder"
frame_folder = r"D:/Sam2/sam2/frame_folder"
output_folder = r"D:/Sam2/sam2/All_annoted_data"


# Create necessary folders
os.makedirs(output_folder, exist_ok=True)

# Function to get video FPS using FFmpeg
def get_video_fps(video_path):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    try:
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        num, den = map(int, output.split('/')) if '/' in output else (int(output), 1)
        return num / den
    except Exception as e:
        print(f"Error getting FPS: {e}")
        return None

# Function to extract frames using FFmpeg
def extract_frames(video_path, temp_folder):
    # Clean and recreate frame folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    # Get video resolution
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json "{video_path}"'
    output = subprocess.check_output(cmd, shell=True).decode()
    video_info = eval(output)
    video_width, video_height = video_info["streams"][0]["width"], video_info["streams"][0]["height"]

    # Get FPS
    fps = get_video_fps(video_path)
    if fps is None:
        return None

    # Extract frames
    cmd = f'ffmpeg -i "{video_path}" -vf "fps={fps}" -start_number 0 "{temp_folder}/%04d.png" -hide_banner -loglevel error'
    subprocess.run(cmd, shell=True, check=True)
    return fps

# Function to create video from frames
def create_video_from_frames(frame_output_path, output_video_path, fps):
    cmd = f'ffmpeg -framerate {fps} -i "{frame_output_path}/%04d.png" -c:v libx264 -pix_fmt yuv420p "{output_video_path}" -hide_banner -loglevel error'
    subprocess.run(cmd, shell=True, check=True)

# Process each video
for video_name in os.listdir(video_folder):
    if not video_name.endswith(".mp4"):  # Process only MP4 videos
        continue

    video_path = os.path.join(video_folder, video_name)
    csv_path = os.path.join(csv_folder, os.path.splitext(video_name)[0] + ".csv")
    
    if not os.path.exists(csv_path):
        print(f"Skipping {video_name}: No matching CSV found.")
        continue

    # Extract frames
    fps = extract_frames(video_path, frame_folder)
    if fps is None:
        print(f"Skipping {video_name}: Failed to extract frames.")
        continue

    # Read CSV
    df = pd.read_csv(csv_path)

    # Process extracted frames
    for frame_file in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame_file)
        frame_idx = int(frame_file.split('.')[0]) - 1  # Extract frame number

        # Read frame
        frame = cv2.imread(frame_path)

        # Get XY coordinates for the current frame
        row = df[df['Frame'] == frame_idx]
        if not row.empty:
            x, y = int(row.iloc[0]['X']), int(row.iloc[0]['Y'])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dot

        # Overlay frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save annotated frame
        cv2.imwrite(frame_path, frame)

    # Reconstruct video from annotated frames
    output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_name)[0]}_annoted.mp4")
    create_video_from_frames(frame_folder, output_video_path, fps)

    print(f"Annotated video saved: {output_video_path}")

print("Processing completed.")


