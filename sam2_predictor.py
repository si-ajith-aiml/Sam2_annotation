import os
import csv
import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
from sam2.build_sam import build_sam2_video_predictor
import subprocess
import pandas as pd
import shutil

# Define directories
#input files
video_folder = "preprocessed_data/All_scencuts_clips/"
text_folder = "preprocessed_data/All_text_files/"

#ouput_folder for no object present ex: no ball in the chunk
empty_csv_folder = "Sam_predictor_data/No_object_present_files/csv_folder/"
empty_video_folder = "Sam_predictor_data/No_object_present_files/videos/"

# output_folder for object present ex: Ball presented in the chunk
sam_csv_folder = "Sam_predictor_data/Object_present_files/csv_folder/"
sam_video_folder = "Sam_predictor_data/Object_present_files/videos/"

error_videos_save =  "Sam_predictor_data/error_files/videos/"
error_csv_save = "Sam_predictor_data/error_files/csv_folder/"
# for extract frames temp file for annotated save 
frame_folder = "frames_for_sam2/"

# videos dispalyed with  coordinates annotated  
sam2_visual_folder= "Sam_predictor_data/Sam2_visualization_annoted_clips/"

# change 0 to stop saving annotated videos 
save_annotate = 1

# Ensure output directories exist
os.makedirs(empty_csv_folder, exist_ok=True)
os.makedirs(empty_video_folder, exist_ok=True)
os.makedirs(sam_csv_folder, exist_ok=True)
os.makedirs(sam_video_folder, exist_ok=True)
os.makedirs(error_videos_save, exist_ok=True)
os.makedirs(error_csv_save, exist_ok=True)

# Initialize SAM2 Predictor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_cfg ="configs/sam2.1/sam2.1_hiera_l.yaml" # Update with actual model path
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"  # Update with actual checkpoint
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


def visual_clips(video_path,csv_path,frame_folder,output_folder):
    try:
        os.makedirs(sam2_visual_folder, exist_ok=True)
        os.makedirs(frame_folder, exist_ok=True)

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

def get_frame_count(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        return len(vr)
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None

def extract_frame_data(file_path):
    try:
        frame_points = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        frame_no, points = None, []
        for line in lines:
            line = line.strip()
            if line.startswith("frame_no:"):
                frame_no = int(line.split(":")[-1].strip())
            elif line.startswith("coordinates:"):
                points = eval(line.split(":")[-1].strip())
                frame_points[frame_no] = points
        return frame_points
    except Exception as e:
        print(f"Error processing inside extract_frame_data {file_path}: {e}")


def save_csv(csv_path, frame_data, total_frames):
    try:
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Visibility", "X", "Y"])

            for i in range(total_frames):
                if i in frame_data and frame_data[i]:  # Ensure data exists
                    x, y = frame_data[i][0]  # Extract from the first tuple in the list
                    visibility = 0 if (x == 0.0 and y == 0.0) else 1
                else:
                    visibility, x, y = 0, 0.0, 0.0

                writer.writerow([i, visibility, x, y])
    except Exception as e:
        print(f"Error processing inside save_csv {csv_path}: {e}")


def run_sam2_annotation(video_path, frame_data):
    try:
        inference_state = predictor.init_state(video_path=video_path)
        
        video_segments = {}
        for frame_no, points in frame_data.items():
            points = np.array(points, dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            
            _, detected_obj_ids, detected_mask_logits = predictor.add_new_points_or_box(
                inference_state, frame_no, obj_id=1, points=points, labels=labels)

            for out_frame_idx, detected_obj_ids, detected_mask_logits in predictor.propagate_in_video(inference_state):
                if len(detected_obj_ids) == len(detected_mask_logits):
                    video_segments[out_frame_idx] = {
                        obj_id: (detected_mask_logits[i] > 0.0).cpu().numpy()
                        for i, obj_id in enumerate(detected_obj_ids)
                    }
        
        frame_results = {}
        for i in sorted(video_segments.keys()):
            coordinates = []
            
            for obj_id, out_mask in video_segments[i].items():
                out_mask = np.squeeze(out_mask)
                contours, _ = cv2.findContours(out_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:

                # Get minimum enclosing circle instead of bounding box
                    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
                    # coordinates.append((center_x, center_y))  # Use circle center
                    coordinates.append((round(center_x, 1), round(center_y, 1)))

  
                #     x, y, w, h = cv2.boundingRect(contour)

                #     center_x, center_y = x + w / 2, y + h / 2  # Fallback

                #     coordinates.append((center_x, center_y))

            frame_results[i] = coordinates if coordinates else [(0.0, 0.0)]
        
        return frame_results
    except Exception as e:  
        print(f"Error processing inside run_sam2_annotation {video_path}: {e}")
        return None


video_list = [v for v in os.listdir(video_folder) if v.endswith(".mp4")]
total_videos = len(video_list)

# Process videos
for count, video_name in enumerate(video_list, start=1):
    try:
        print(f"Processing video {count}/{total_videos}: {video_name}...!")
        
        video_path = os.path.join(video_folder, video_name)
        text_path = os.path.join(text_folder, video_name + ".txt")
        
        if not os.path.exists(text_path):
            print(f"No text file for {video_name}, skipping.")
            continue
        
        frame_data = extract_frame_data(text_path)
        total_frames = get_frame_count(video_path)
        print("Total frames inside sam2 prediction",total_frames)
        if total_frames is None:
            print(f"Skipping {video_name} due to video read error.")
            continue
        
        values = list(frame_data.values())  # Convert to a list

        if len(values) == 1:
            print("values", values)  # Now it's a list 
            print("values[0][0]",  values[0][0][0])  # First element inside list

            print("values[0][1]",  values[0][0][1])  # Second element inside list

        if values[0][0][0] == 0 and values[0][0][1] == 0:

            # Empty coordinates case
            csv_path = os.path.join(empty_csv_folder, video_name.replace(".mp4", ".csv"))
            save_csv(csv_path, {}, total_frames)
            os.rename(video_path, os.path.join(empty_video_folder, video_name))
        else:
            # Run SAM2 annotation
            frame_results = run_sam2_annotation(video_path, frame_data)
            if frame_results is not None:
                csv_path = os.path.join(sam_csv_folder, video_name.replace(".mp4", ".csv"))
                save_csv(csv_path, frame_results, total_frames)
                os.rename(video_path, os.path.join(sam_video_folder, video_name))

                print(f"same predicted and saved video and csv !!!!!!{video_name} ")

                # for video annotation tool save ##########
                if save_annotate:
                    visual_video_path = os.path.join(sam_video_folder, video_name)
                    visual_csv_path = os.path.join(sam_csv_folder, os.path.splitext(video_name)[0] + ".csv")
                    visual_clips(visual_video_path,visual_csv_path,frame_folder,sam2_visual_folder)
                else:
                    print(f"Due to save_annotate is 0 , Skipping visual saving annotated clip for {video_name}")

            else:
                print(f"issues in sam prediction in this video moving to error file {video_name}")
                os.rename(video_path, os.path.join(error_videos_save, video_name))
                csv_path = os.path.join(error_csv_save, video_name.replace(".mp4", ".csv"))
                save_csv(csv_path, {}, total_frames)

    except Exception as e:
        print(f"Error processing inside sam 2 prediction {video_name}: {e}")

    print(f"Processed {video_name}")




