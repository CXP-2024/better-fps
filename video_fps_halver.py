import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import tempfile
from tqdm import tqdm

# Fix MoviePy import for Python 3.13 compatibility
try:
    from moviepy import VideoFileClip  # MoviePy 2.1.2 with Python 3.13
except ImportError:
    try:
        #from moviepy.editor import VideoFileClip  # For older versions
        pass
    except ImportError:
        print("Warning: MoviePy not found. Audio will not be preserved.")
        VideoFileClip = None

def halve_fps(input_path, output_path, output_folder, num_visualize=5):
    """
    Halve the FPS of a video by keeping only every other frame and save visualizations
    
    Args:
        input_path: Path to input MP4 video
        output_path: Path to save output MP4 video with audio preserved
        output_folder: Folder to save visualization frames
        num_visualize: Number of consecutive frames to visualize
    """
    # Create folder for output images if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create subfolders for different types of frames
    originals_folder = os.path.join(output_folder, "originals")
    kept_folder = os.path.join(output_folder, "kept")
    
    # Create all subfolders
    os.makedirs(originals_folder, exist_ok=True)
    os.makedirs(kept_folder, exist_ok=True)
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Create temporary file for video without audio
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # Create output video with half FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps/2, (width, height))
    
    # Choose frames to visualize (around the middle of the video)
    mid_frame = frame_count // 2
    visual_start = max(0, mid_frame - num_visualize)
    visual_end = min(frame_count - 1, visual_start + num_visualize * 2)
    frame_idx = 0
    visualization_frames = []
    
    # Process each frame with a progress bar
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Keep only even-indexed frames (0, 2, 4, etc.)
            if frame_idx % 2 == 0:
                # Add frame to output video
                out.write(frame)
                
                # If this is a frame we want to visualize
                if visual_start <= frame_idx <= visual_end:
                    # Get current visualization index
                    vis_idx = (frame_idx - visual_start) // 2
                    
                    # Save as kept frame
                    cv2.imwrite(f"{kept_folder}/{vis_idx}.jpg", frame)
            
            # If this is a frame we want to visualize (both kept and discarded)
            if visual_start <= frame_idx <= visual_end:
                # Save original frame
                cv2.imwrite(f"{originals_folder}/{frame_idx - visual_start}.jpg", frame)
            
            # Update for next iteration
            frame_idx += 1
            
            # Update progress bar
            pbar.update(1)
    
    # Release OpenCV resources
    cap.release()
    out.release()
      # Extract audio from original video and add to the new video
    print("Adding audio to the output video...")
    try:
        if VideoFileClip is None:
            raise ImportError("MoviePy module not available")
            
        # Load the original video to get audio
        original_clip = VideoFileClip(input_path)
        
        # Load the newly created video without audio
        new_clip = VideoFileClip(temp_output)
        
        # Add audio from original to new clip
        if original_clip.audio is not None:
            new_clip = new_clip.with_audio(original_clip.audio)
        
        # Write final video with audio
        new_clip.write_videofile(output_path)
        
        # Close the clips
        original_clip.close()
        new_clip.close()
        
        # Wait a moment to ensure file handles are released
        import time
        time.sleep(1)
        
        # Remove the temporary file
        os.unlink(temp_output)
    except Exception as e:
        print(f"Error adding audio: {e}")
        print(f"Saving video without audio to {output_path}")
        # If something goes wrong, just use the video without audio
        if os.path.exists(temp_output):
            try:
                # Close any open file handles that might be holding the file
                import gc
                gc.collect()
                
                # Wait a moment to ensure file handles are released
                import time
                time.sleep(1)
                
                # Make sure output path is free
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except Exception:
                        print(f"Could not delete existing output file: {output_path}")
                
                # Copy instead of rename to avoid permission issues
                import shutil
                shutil.copy2(temp_output, output_path)
                
                # Try to delete the temporary file
                try:
                    os.unlink(temp_output)
                except Exception:
                    print(f"Could not delete temporary file: {temp_output}")
            except Exception as copy_err:
                print(f"Error copying temporary file: {copy_err}")
                print(f"Temporary file remains at: {temp_output}")
    
    print(f"Processing complete. Output video saved to {output_path}")
    print(f"Visualization frames saved to {output_folder}")
    print(f"Original FPS: {fps}, New FPS: {fps/2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Halve video FPS by keeping only every other frame')
    parser.add_argument('input', help='Input video file (MP4)')
    parser.add_argument('--output', default=None, help='Output video file')
    parser.add_argument('--vis-folder', default='halved_visualization_frames', help='Folder for visualization frames')
    parser.add_argument('--vis-count', type=int, default=10, help='Number of frames to visualize')
    
    args = parser.parse_args()
    
    # If no output path is specified, create one based on the input filename and target FPS
    if args.output is None:
        # Get input video properties to determine the FPS
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.input}")
            exit(1)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Extract the base name without extension
        input_path = Path(args.input)
        base_name = input_path.stem
        extension = input_path.suffix
        
        # Create output filename with halved FPS info
        halved_fps = fps / 2
        output_filename = f"{base_name}_halved_to_{int(halved_fps)}{extension}"
        output_path = str(input_path.with_name(output_filename))
    else:
        output_path = args.output
    
    halve_fps(args.input, output_path, args.vis_folder, args.vis_count)
