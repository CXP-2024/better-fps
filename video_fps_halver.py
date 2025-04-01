import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def halve_fps(input_path, output_path, output_folder, num_visualize=5):
    """
    Halve the FPS of a video by keeping only every other frame and save visualizations
    
    Args:
        input_path: Path to input MP4 video
        output_path: Path to save output MP4 video 
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
    
    # Create output video with half FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps/2, (width, height))
    
    # Choose frames to visualize (around the middle of the video)
    mid_frame = frame_count // 2
    visual_start = max(0, mid_frame - num_visualize)
    visual_end = min(frame_count - 1, visual_start + num_visualize * 2)
    
    frame_idx = 0
    visualization_frames = []
    
    # Process each frame
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
        
        # Show progress
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{frame_count}")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Processing complete. Output video saved to {output_path}")
    print(f"Visualization frames saved to {output_folder}")
    print(f"Original FPS: {fps}, New FPS: {fps/2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Halve video FPS by keeping only every other frame')
    parser.add_argument('input', help='Input video file (MP4)')
    parser.add_argument('--output', default='output_halved.mp4', help='Output video file')
    parser.add_argument('--vis-folder', default='halved_visualization_frames', help='Folder for visualization frames')
    parser.add_argument('--vis-count', type=int, default=10, help='Number of frames to visualize')
    
    args = parser.parse_args()
    
    halve_fps(args.input, args.output, args.vis_folder, args.vis_count)
