import cv2
import pandas as pd
import os
import subprocess
import shutil




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
    try:
        print("Extracting frames...")
        # Clean and recreate frame folder

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
        print("len of frame list",len(os.listdir(temp_folder)),"...")
        return fps
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None

def visual_clips(video_path,csv_path,frame_folder,output_folder):
    try:
        os.makedirs(frame_folder, exist_ok=True)
        video_name = os.path.basename(video_path)
        # Check if CSV exists
        if not os.path.exists(csv_path):
            print(f"Skipping {video_name}: No matching CSV found.")

        # Extract frames
        fps = extract_frames(video_path, frame_folder)
        print(f"fps is {fps}")
        if fps is None:
            print(f"Skipping {video_name}: Failed to extract frames.")

        # Read CSV
        df = pd.read_csv(csv_path)
        print("reading csv done")

        # Process extracted frames

        for frame_file in sorted(os.listdir(frame_folder)):
            frame_path = os.path.join(frame_folder, frame_file)
            frame_idx = int(frame_file.split('.')[0])  # Extract frame number

            # Read frame
            frame = cv2.imread(frame_path)

            # Get XY coordinates for the current frame
            row = df[df['Frame'] == frame_idx]
            if not row.empty:
                x, y = int(row.iloc[0]['X']), int(row.iloc[0]['Y'])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red dot

            # Overlay frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save annotated frame
            cv2.imwrite(frame_path, frame)

        # Reconstruct video from annotated frames
        output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_name)[0]}_annoted.mp4")

        print("creating visual_clips_function")
        cmd = f'ffmpeg -framerate {fps} -i "{frame_folder}/%04d.png" -c:v libx264 -pix_fmt yuv420p "{output_video_path}" -hide_banner -loglevel error'
        subprocess.run(cmd, shell=True, check=True)

        print(f"Annotated video saved: {output_video_path}")

        #removed frame folder
        shutil.rmtree(frame_folder, ignore_errors=True)
    except Exception as e:
        print(f"Error processing inside visual_clips_function : {video_name}: {e}")




video_path = "D:/Sam2/sam2_Annotation/Sam_predictor_data/Object_present_files/videos/1000_scenecut_1.mp4"
csv_path = "D:/Ajit/football/ajith_corrected_csv/1000_scenecut_1.csv"
frame_folder = "ajith_frmaes"
output_folder = "D:/Sam2/sam2_Annotation/Sam_predictor_data/"
# videos dispalyed with  coordinates annotated  
visual_clips(video_path,csv_path,frame_folder,output_folder)