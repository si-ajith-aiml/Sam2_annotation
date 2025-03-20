
import cv2
import os
import shutil
import subprocess
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from decord import VideoReader, cpu
from scene_cut import main
# Folder paths
video_folder = "input_videos"  # Change this path
temp_folder = "temp_frames"
txt_folder = "preprocessed_data/ALL_text_files"
clips_done_folder = "preprocessed_data/clips_done"
common_folder = "scene_cut"
annotated_folder = "preprocessed_data/All_scencuts_clips"

# Ensure necessary folders exist
os.makedirs(temp_folder, exist_ok=True)
os.makedirs(txt_folder, exist_ok=True)
os.makedirs(clips_done_folder, exist_ok=True)

# Get list of MP4 videos
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
current_video_index = 0
selected_points = []
frame_list = []
current_frame_index = 0

# Function to get FPS using ffmpeg
print(" for next frame press right arrow")
print(" for previous frame press left arrow")
print(" for first frame press f")
print(" for last frame press l")
print(" for save press enter")
print(" for exit press q")
print(" for remove last point press r")

def get_frame_count(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        return len(vr)
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None

# Function to extract frames without missing any
def extract_frames(video_path):
    global video_width, video_height, frame_list, display_width, display_height
    try:
        shutil.rmtree(temp_folder, ignore_errors=True)
        os.makedirs(temp_folder, exist_ok=True)

        # Get video resolution
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json "{video_path}"'
        output = subprocess.check_output(cmd, shell=True).decode()
        video_info = eval(output)
        video_width, video_height = video_info["streams"][0]["width"], video_info["streams"][0]["height"]

        # Set display size
        display_width, display_height = 1400, 700

        # Get frame count
        frame_count = get_frame_count(video_path)
        if frame_count is None:
            return None

        # Extract frames using frame count to avoid frame loss
        cmd = f'ffmpeg -i "{video_path}" -vsync 0 -frame_pts true "{temp_folder}/%04d.png" -hide_banner -loglevel error'
        subprocess.run(cmd, shell=True, check=True)

        frame_list = sorted(os.listdir(temp_folder), key=lambda x: int(os.path.splitext(x)[0]))
        print(f"Extracted {len(frame_list)} frames (Expected: {frame_count})")

        # Validate frame count
        if len(frame_list) != frame_count:
            print(f"Warning: Extracted frame count ({len(frame_list)}) does not match expected ({frame_count})!")

        return frame_list
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None
    

def get_video_fps(video_path):
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "{video_path}"'
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        num, den = map(int, output.split('/'))
        return num / den
    except Exception as e:
        print(f"Error getting FPS for {video_path}: {e}")
        return None

def display_frame():
    try:
        global current_frame_index
        if not frame_list or current_frame_index >= len(frame_list):
            return

        frame_path = os.path.join(temp_folder, frame_list[current_frame_index])
        img = cv2.imread(frame_path)
        if img is None:
            return

        img_resized = cv2.resize(img, (display_width, display_height))

        # Draw selected points
        for (x, y) in selected_points:
            x_resized, y_resized = int(x * display_width / video_width), int(y * display_height / video_height)
            cv2.circle(img_resized, (x_resized, y_resized), 3, (0, 255, 0), -1)

        # Add frame number with smaller text
        cv2.putText(img_resized, f"Frame {current_frame_index}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255,0), 2)

        img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
        image_label.config(image=img_tk)
        image_label.image = img_tk
    except Exception as e:
        print(f"Error processing inside display_frame: {e}")

# Convert clicked coordinates to original resolution
def convert_coordinates(x, y):
    return int(x * video_width / display_width), int(y * video_height / display_height)

# Mouse click event handler
def on_mouse_click(event):
    global selected_points
    if len(selected_points) < 3:
        print(f"displayed image clicked xy coordinates at: ({event.x}, {event.y})")
        x, y = convert_coordinates(event.x, event.y)
        print(f"orginal size converted clicked xy coordinates: ({x}, {y})")
        selected_points.append((x, y))
        display_frame()
    else:
        messagebox.showwarning("Limit Reached", "Maximum points selected!")

# Key bindings
def next_frame(event):
    global current_frame_index, selected_points
    if current_frame_index < len(frame_list) - 1:
        current_frame_index += 1
        selected_points = []
        display_frame()

def prev_frame(event):
    global current_frame_index, selected_points
    if current_frame_index > 0:
        current_frame_index -= 1
        selected_points = []
        display_frame()

def first_frame(event):
    global current_frame_index, selected_points
    current_frame_index = 0
    selected_points = []
    display_frame()

def last_frame(event):
    global current_frame_index, selected_points
    current_frame_index = len(frame_list) - 1
    selected_points = []
    display_frame()

def exit_application(event):
    print("Exiting without saving...")
    root.quit()

# Remove last clicked point
def remove_last_point(event):
    global selected_points
    if selected_points:
        selected_points.pop()
        display_frame()


def save_annotation():
    try:
        """
        Saves the annotation (frame number and selected points) for the current scene-cut clip.
        After saving, moves the clip and asks whether to proceed.
        """
        global selected_points, scene_files, current_scene_index, all_scenes_annotated

        scene_name = scene_files[current_scene_index]  # Get the current scene-cut clip
        txt_filename = os.path.join(txt_folder, f"{scene_name}.txt")

        frame_no = current_frame_index
        if not selected_points:
            selected_points = [(0,0)]

        with open(txt_filename, "w") as f:
            f.write(f"frame_no: {frame_no}\n")
            f.write(f"coordinates: {selected_points}\n")

        # Ask the user if they want to proceed to the next scene
        proceed = messagebox.askyesno("Save Confirmation", "Saved Successfully! Move to the next scene?")

        if proceed:
            # Mark this scene-cut as annotated
            all_scenes_annotated.append(scene_name)
            # Move to the next scene
            move_scene_and_ask_next()
        else:
            # Stay on the current scene
            return  

        # messagebox.showinfo("Success", "Saved Successfully!")

        # # Mark this scene-cut as annotated
        # all_scenes_annotated.append(scene_name)

        # # Move the processed scene-cut clip and ask for the next one
        # move_scene_and_ask_next()
    except Exception as e:
        print(f"Error processing inside save_annotation: {e}")


def move_scene_and_ask_next():
    try:
        """
        Moves the processed scene-cut clip to 'annotated_folder' and continues to the next scene.
        If all scene clips are processed, move the original video and ask whether to continue.
        """
        global scene_files, current_scene_index, video_files, current_video_index, all_scenes_annotated

        scene_name = scene_files[current_scene_index]
        scene_path = os.path.join(common_folder, scene_name)

        # Ensure the annotated folder exists
        os.makedirs(annotated_folder, exist_ok=True)

        # Move the scene-cut clip to annotated folder if annotation was saved
        if scene_name in all_scenes_annotated:
            shutil.move(scene_path, os.path.join(annotated_folder, scene_name))
            print(f"{scene_name} moved to annotated folder")
        else:
            print(f"Skipping move for {scene_name} as it was not annotated.")

        if current_scene_index + 1 < len(scene_files):
            print(f"Processing next scene: {scene_files[current_scene_index + 1]}...")
            current_scene_index += 1
            process_scene_cut()  # Process next scene-cut clip
        else:
            # All scene-cut clips are done for this video
            if set(scene_files) == set(all_scenes_annotated):
                move_video_to_done()
            else:
                print("Some scene cuts were not annotated. Keeping original video.")

            # Ask to continue with the next video
            response = messagebox.askyesno("Continue?", "Continue with next video?")
            if response:
                move_video_and_ask_next()
            else:
                root.destroy()

    except Exception as e:
        print(f"Error processing inside move_scene_and_ask_next: {e}")


def move_video_to_done():
    try:
        """
        Moves the original video to 'clips_done_folder' after all scene-cut clips are processed.
        """
        global video_files, current_video_index

        video_name = video_files[current_video_index]
        video_path = os.path.join(video_folder, video_name)

        shutil.move(video_path, os.path.join(clips_done_folder, video_name))
        print(f"Original video {video_name} moved to clips_done")

    except Exception as e:
        print(f"Error processing inside move_video_to_done: {e}")


def move_video_and_ask_next():
    try:
        """
        After processing all scene-cut clips, move to the next original video.
        """
        global video_files, current_video_index

        if current_video_index + 1 < len(video_files):
            print(f"Processing next video: {video_files[current_video_index+1]}...")
            current_video_index += 1
            start_annotation_loop()
        else:
            messagebox.showinfo("Completed", "All videos processed!")
            root.destroy()

    except Exception as e:
        print(f"Error processing inside move_video_and_ask_next: {e}")


def start_annotation_loop():
    try:
        """
        Starts processing videos by first extracting scene cuts using `main()`.
        Then processes each scene-cut clip one by one.
        """
        global current_frame_index, scene_files, current_scene_index, all_scenes_annotated

        if current_video_index >= len(video_files):
            messagebox.showinfo("Completed", "No more videos to process!")
            root.destroy()
            return

        video_name = video_files[current_video_index]  # Example: "shot_on_goal_2.mp4"
        video_base_name = os.path.splitext(video_name)[0]  # Removes ".mp4", now "shot_on_goal_2"
        video_path = os.path.join(video_folder, video_name)

        print(f"Processing video: {video_name} ({current_video_index+1}/{len(video_files)})...")

        # Step 1: Extract scene cuts and store in `common_folder`
        main(video_path)  # Calls main() to extract scene cuts into `common_folder`


        # Step 3: Get all scene-cut clips from `common_folder`
        scene_files = sorted([f for f in os.listdir(common_folder) if f.startswith(video_base_name + "_scenecut_")])

        print("Scene Files Found:", scene_files)  # Debug print

        if not scene_files:
            # print(f"No scene cuts found for {video_name}. Skipping...")
            move_video_and_ask_next()
            return

        # Step 4: Prepare for scene annotation
        current_scene_index = 0
        all_scenes_annotated = []  # Track annotated scenes
        process_scene_cut()
    except Exception as e:
        print(f"Error processing video inside start_Annotation_loop{video_name}: {e}")



def process_scene_cut():
    try:
        """
        Extracts frames from the current scene-cut clip and starts annotation.
        """
        global current_frame_index, selected_points

        scene_name = scene_files[current_scene_index]
        scene_path = os.path.join(common_folder, scene_name)
        print(f"Processing scene-cut: {scene_name} ({current_scene_index+1}/{len(scene_files)})...")

        frame_list = extract_frames(scene_path)

        if not frame_list:
            print(f"Skipping {scene_name} due to errors.")
            move_scene_and_ask_next()
            return

        current_frame_index = 0
        selected_points = []
        display_frame()
    except Exception as e:
        print(f"Error processing inside process_scene-cut {scene_name}: {e}")


root = tk.Tk()
root.title("Video Annotation Tool")

image_label = tk.Label(root)
image_label.pack()

root.bind("<Right>", next_frame)
root.bind("<Left>", prev_frame)
root.bind("<f>", first_frame)
root.bind("<l>", last_frame)
root.bind("<Return>", lambda event: save_annotation())
root.bind("<q>", lambda e: root.destroy())  # Exit on 'q'
root.bind("<r>", remove_last_point)
root.bind("<Button-1>", on_mouse_click)  # Left-click to add point

start_annotation_loop()
root.mainloop()








