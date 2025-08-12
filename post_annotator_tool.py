import cv2
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from PIL import Image, ImageTk
import shutil
import subprocess
import numpy as np

# Global variables
current_frame = 0
video = None
annotations = None
img = None
image_cache = {}  # Cache for images
canvas = None
annotation_file = None
video_width = None
video_height = None
display_width, display_height = 1200, 700
video_files = []
current_video_index = 0
frames_folder = None
total_frames = None

base = "Sam_predicted_data/Object_present_files/"
corrected_csv_folder = base + "corrected_csv"
video_folder = base + "videos"
clips_done_folder = base + "clips_done"
annotated_videos_qc_pending = base + "videos"
annotation_folder = base + "csv_folder"

def downscale(x, y, video_width, video_height, display_width, display_height):
    x_scaled = (x / video_width) * display_width
    y_scaled = (y / video_height) * display_height
    return x_scaled, y_scaled

def upscale(x, y, video_width, video_height, display_width, display_height):
    x_original = (x / display_width) * video_width
    y_original = (y / display_height) * video_height
    return x_original, y_original

def get_video_fps(video_path):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    try:
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        num, den = map(int, output.split('/')) if '/' in output else (int(output), 1)
        return num / den
    except Exception as e:
        print(f"Error getting FPS: {e}")
        return None
def generate_frames(video_path):
    global video_width, video_height, total_frames
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json "{video_path}"'
        output = subprocess.check_output(cmd, shell=True).decode()
        video_info = eval(output)
        video_width, video_height = video_info["streams"][0]["width"], video_info["streams"][0]["height"]

        fps = get_video_fps(video_path)
        if fps is None:
            return None

        output_folder = "frames/" + os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_folder, exist_ok=True)

        # Change to start_number 1 if your frame sequence starts from 1
        cmd = f'ffmpeg -i "{video_path}" -vf "fps={fps}" -start_number 0 "{output_folder}/%d.png" -hide_banner -loglevel error'
        subprocess.run(cmd, shell=True, check=True)

        total_frames = len(os.listdir(output_folder))
        return output_folder
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None


def get_frame_with_ffmpeg(frames_folder, frame_number):
    path = f"{frames_folder}/{frame_number}.png"  # Adjust if frames start from 1
    if not os.path.exists(path):
        print(f"Frame {frame_number} not found: {path}")
        return None

    image = cv2.imread(path)
    if image is None:
        print(f"Failed to read the image at {path}")
        return None

    return image

def display_frame():
    global current_frame, img, canvas, image_cache, current_video_index, frames_folder

    frame = get_frame_with_ffmpeg(frames_folder, current_frame)

    if frame is not None:
        frame = cv2.resize(frame, (display_width, display_height))
        cv2.putText(frame, f"Frame: {current_frame}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        draw_annotations(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=frame_image)
        image_cache[current_frame] = img
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
    else:
        print(f"Frame {current_frame} could not be loaded.")

def draw_annotations(frame):
    global annotations, current_frame, video_width, video_height

    frame_annotations = annotations[annotations['Frame'] == current_frame]

    if frame_annotations.empty:
        cv2.putText(frame, "Click to add annotation", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        for index, row in frame_annotations.iterrows():
            x = int(row.get('X', 0))
            y = int(row.get('Y', 0))
            x_display, y_display = downscale(x, y, video_width, video_height, display_width, display_height)
            cv2.circle(frame, (int(x_display), int(y_display)), 2, (0, 0, 225), -1)

def remove_annotation(event):
    global annotations, current_frame

    if current_frame in annotations['Frame'].values:
        annotations.loc[annotations['Frame'] == current_frame, ['Visibility', 'X', 'Y']] = [0, 0, 0]
    else:
        new_annotation = pd.DataFrame({'Frame': [current_frame], 'Visibility': [0], 'X': [0], 'Y': [0]})
        annotations = annotations.append(new_annotation, ignore_index=True)

    save_annotations()
    display_frame()

def add_annotation(event):
    global annotations, current_frame, video_width, video_height

    x1_display = int(event.x)
    y1_display = int(event.y)
    x1_upscaled, y1_upscaled = upscale(x1_display, y1_display, video_width, video_height, display_width, display_height)

    if annotations[annotations['Frame'] == current_frame].empty:
        new_annotation = pd.DataFrame({'Frame': [current_frame], 'Visibility': [1], 'X': [x1_upscaled], 'Y': [y1_upscaled]})
        annotations = pd.concat([annotations, new_annotation], ignore_index=True)
    else:
        annotations.loc[annotations['Frame'] == current_frame, ['Visibility', 'X', 'Y']] = [1, x1_upscaled, y1_upscaled]

    save_annotations()
    display_frame()

def save_annotations():
    global annotation_file, total_frames

    all_frames = pd.DataFrame({'Frame': range(total_frames)})
    complete_annotations = all_frames.merge(annotations, on='Frame', how='left')

    # Fill missing values with default values
    complete_annotations['X'] = complete_annotations['X'].fillna(0.0).apply(lambda x: round(x, 1))
    complete_annotations['Y'] = complete_annotations['Y'].fillna(0.0).apply(lambda x: round(x, 1))
    complete_annotations['Visibility'] = complete_annotations['Visibility'].fillna(0)

    # Ensure correct data types
    complete_annotations['Frame'] = complete_annotations['Frame'].astype(int)
    complete_annotations['Visibility'] = complete_annotations['Visibility'].astype(int)

    # Save the updated annotations to the CSV file
    complete_annotations.to_csv(annotation_file, index=False)
    print("Annotations saved at " + annotation_file)
def next_frame(event):
    global current_frame

    if current_frame < total_frames - 1:
        current_frame += 1
    display_frame()

def prev_frame(event):
    global current_frame

    if current_frame > 0:
        current_frame -= 1
    display_frame()

def remove_annotation_and_next_frame(event):
    remove_annotation(event)
    next_frame(event)

def go_to_first_frame(event):
    global current_frame
    current_frame = 0
    display_frame()

def go_to_last_frame(event):
    global current_frame 
    current_frame = total_frames - 1
    display_frame()

def show_message_box(message):
    response = messagebox.askyesno("Information", message, icon='warning')
    return response

def finish_task(event):
    global annotations, current_video_index, video_files, video

    os.makedirs(clips_done_folder, exist_ok=True)
    destination_path = os.path.join(clips_done_folder, os.path.basename(video_files[current_video_index]))
    shutil.copy(video_files[current_video_index], destination_path)

    print(f"Video {video_files[current_video_index]} saved to '{clips_done_folder}' folder.")
    os.remove(video_files[current_video_index])

        # Delete extracted frames folder
    if frames_folder and os.path.exists(frames_folder):
        print(f"Deleting frames folder: {frames_folder}")
        shutil.rmtree(frames_folder)

    response = show_message_box("Do you want to continue with the next video?")
    if response:
        current_video_index += 1

        if current_video_index < len(video_files):
            start_video_processing()
        else:
            print("All videos processed.")
            window.destroy()
    else:
        print("Exiting...")
        window.destroy()
    save_annotations()

def start_video_processing():
    global annotations, current_frame, annotation_file, video_files, current_video_index, frames_folder, video

    video_path = video_files[current_video_index]
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    original_annotation_file = os.path.join(annotation_folder, base_name + ".csv")
    corrected_annotation_file = os.path.join(corrected_csv_folder, base_name + ".csv")

    if not os.path.exists(corrected_annotation_file):
        shutil.copyfile(original_annotation_file, corrected_annotation_file)

    annotation_file = corrected_annotation_file

    frames_folder = generate_frames(video_path)
    print("Extracted frames ")

    if frames_folder is not None:
        annotations = pd.read_csv(annotation_file)
        current_frame = 0
        display_frame()
    else:
        print("Frames could not be extracted.")

if __name__ == "__main__":
    os.makedirs(corrected_csv_folder, exist_ok=True)

    qc_video_files = [os.path.join(annotated_videos_qc_pending, f) for f in os.listdir(annotated_videos_qc_pending) if f.endswith('.mp4')]
    qc_video_files.sort()
    qc_video_names = [os.path.splitext(os.path.basename(f))[0].replace('_annotated', '') for f in qc_video_files]

    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
    video_files.sort()
    video_names = [os.path.splitext(os.path.basename(f))[0] for f in video_files]

    video_files = [f for f in video_files if os.path.splitext(os.path.basename(f))[0] in qc_video_names]

    if video_files:
        window = tk.Tk()
        window.title("CSV Corrector Tool")

        canvas = tk.Canvas(window, width=display_width, height=display_height)
        canvas.pack()

        window.bind("<r>", remove_annotation)
        window.bind("<Button-1>", add_annotation)
        window.bind("<space>", next_frame)
        window.bind("<p>", prev_frame)
        window.bind("<Return>", finish_task)
        window.bind("<f>", go_to_first_frame)
        window.bind("<l>", go_to_last_frame)
        window.bind("<q>", lambda event: window.destroy())
        window.bind("<c>", remove_annotation_and_next_frame)
        # ("<Right>", next_frame)
        start_video_processing()

        window.mainloop()
    else:
        print("No video files found in the specified folder.")
