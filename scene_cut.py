import subprocess
import ffmpeg
import os
import re

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
import os

def get_scene_cuts(input_video, threshold=27.0):
    """
    Detects scene cuts using PySceneDetect and returns a list of (start_frame, end_frame).
    Includes both soft and hard cuts.
    
    :param input_video: Path to the input video file.
    :param threshold: Content threshold for detecting scene changes.
    :param hardcut_threshold: Threshold for detecting hard cuts.
    :return: List of tuples (start_frame, end_frame) for detected scenes.
    """
    try:
        video = open_video(input_video)
        if video is None:
            print(f"Error: Unable to open video {input_video}")
            return None

        scene_manager = SceneManager()
        
        # Add both detectors
        scene_manager.add_detector(ContentDetector(threshold=threshold))


        # Detect scenes
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            print("No scene cuts detected. Try adjusting the thresholds.")
            return None

        # Convert FrameTimecode to frame numbers
        scene_frames = [(scene[0].get_frames(), scene[1].get_frames()) for scene in scene_list]

        print(f"Detected Scene Cuts: {scene_frames}")
        return scene_frames
    except Exception as e:
        print(f"Error in function 'get_scene_cuts': {e}")
        return None

def get_video_fps_duration(input_video):
    """Gets FPS and duration of the video using FFmpeg."""
    try:
        probe = ffmpeg.probe(input_video)
        frame_rate_str = next(
            (stream['r_frame_rate'] for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )

        if frame_rate_str is None:
            raise ValueError(f"FPS not found for video: {input_video}")

        frame_rate = eval(frame_rate_str)  # Convert string to float
        video_duration = float(probe['format']['duration'])

        return frame_rate, video_duration
    except Exception as e:
        print(f"Error in function 'get_video_fps_duration': {e}")
        return None, None  # Return None to skip this video


def filter_scene_intervals(scene_frames, total_frames, fps):
    """Filters scene intervals to remove short scenes (less than FPS) and keeps only detected scenes."""
    try:
        if not scene_frames:
            return None

        # Filter out scenes shorter than 1 second (fps)
        scene_intervals = [
            (start, end)
            for start, end in scene_frames
            if (end - start) >= fps
        ]
        
        return scene_intervals
    except Exception as e:
        print(f"Error in function 'filter_scene_intervals': {e}")
        return None


def save_scene_clips(input_video, scene_intervals, output_folder, fps, approximate_frame_count):
    """Saves each scene cut as a separate video file with frame correction."""
    try:
        if scene_intervals is None:
            return

        os.makedirs(output_folder, exist_ok=True)

        for i, (start_frame, end_frame) in enumerate(scene_intervals):
            # Ensure end_frame does not exceed the approximate total frame count
            if end_frame >= approximate_frame_count:
                end_frame = approximate_frame_count - 1

            start_time = start_frame / fps
            duration = (end_frame - start_frame) / fps
            output_video = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_video))[0]}_scenecut_{i+1}.mp4")

            command = (
                f"ffmpeg -i \"{input_video}\" -ss {start_time:.2f} -t {duration:.2f} -r {fps} -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k \"{output_video}\""
            )

            subprocess.run(command, shell=True, check=True)
            print(f"Saved scene: {output_video}")
    except Exception as e:
        print(f"Error in function 'save_scene_clips': {e}")


def main(input_video):
    """Main function to detect, filter, and save scene cuts as separate videos."""
    try:
        print(f"Processing video: {input_video}")

        # Extract scene cuts
        scene_frames = get_scene_cuts(input_video)
        print("scene_frames",scene_frames)
        if scene_frames is None:
            print(f"Skipping video due to scene detection failure: {input_video}")
            return

        # Get FPS and total frames
        fps, video_duration = get_video_fps_duration(input_video)
        fps = round(fps)
        print("fps",fps)
        print("video_duration",video_duration)
        if fps is None or video_duration is None:
            print(f"Skipping video due to missing FPS: {input_video}")
            return

        total_frames = int(video_duration * fps)

        # Filter scene cuts based on FPS
        scene_intervals = filter_scene_intervals(scene_frames, total_frames, fps)
        print("filter_scene_intervals",scene_intervals)

        common_folder = "scene_cut"
        # Extract video name without extension
        # video_name = os.path.splitext(os.path.basename(input_video))[0]

        # # Create folder for scene clips
        # output_folder = os.path.join(common_folder, f"{video_name}_scenecut")
        output_folder = common_folder
        
        save_scene_clips(input_video, scene_intervals, output_folder, fps,total_frames)

        print(f"All scene cuts saved in: {output_folder}")

    except Exception as e:
        print(f"Error in function 'main': {e}")
