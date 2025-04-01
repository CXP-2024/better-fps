import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def average_frames(prev_frame, curr_frame):
    """Create an intermediate frame by simple averaging of two consecutive frames"""
    # Simple averaging of the two frames (50% blend)
    interpolated = cv2.addWeighted(prev_frame, 0.5, curr_frame, 0.5, 0)
    return interpolated

def double_fps_with_average(input_path, output_path, output_folder, num_visualize=5):
    """
    Double the FPS of a video using simple frame averaging and save visualizations
    
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
    interpolated_folder = os.path.join(output_folder, "interpolated")
    combined_folder = os.path.join(output_folder, "combined")
    
    # Create all subfolders
    os.makedirs(originals_folder, exist_ok=True)
    os.makedirs(interpolated_folder, exist_ok=True)
    os.makedirs(combined_folder, exist_ok=True)
    
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
    
    # Create output video with double FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps*2, (width, height))
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the first frame")
        return
    
    # Add first frame to output video
    out.write(prev_frame)
    
    # Choose frames to visualize (around the middle of the video)
    mid_frame = frame_count // 2
    visual_start = max(1, mid_frame - num_visualize // 2)
    visual_end = min(frame_count - 1, visual_start + num_visualize - 1)
    
    frame_idx = 1
    visualization_frames = []
    
    # Process each frame
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Generate intermediate frame using simple averaging
        interpolated = average_frames(prev_frame, curr_frame)
        
        # Add interpolated and current frames to output video
        out.write(interpolated)
        out.write(curr_frame)
        
        # If this is a frame we want to visualize
        if visual_start <= frame_idx <= visual_end:
            # Get current visualization index
            vis_idx = len(visualization_frames)
            
            # Save original frames with consecutive numbering
            cv2.imwrite(f"{originals_folder}/{vis_idx*2}.jpg", prev_frame)
            
            # Save interpolated frame with consecutive numbering
            cv2.imwrite(f"{interpolated_folder}/{vis_idx}.jpg", interpolated)
            
            # Save combined frames (originals and interpolated) with consecutive numbering
            cv2.imwrite(f"{combined_folder}/{vis_idx*2}.jpg", prev_frame)
            cv2.imwrite(f"{combined_folder}/{vis_idx*2+1}.jpg", interpolated)
            if vis_idx < num_visualize - 1:  # Don't duplicate the last frame
                cv2.imwrite(f"{combined_folder}/{vis_idx*2+2}.jpg", curr_frame)
            
            # Store frames for visualization
            visualization_frames.append((prev_frame.copy(), curr_frame.copy(), interpolated.copy()))
        
        # Update for next iteration
        prev_frame = curr_frame
        frame_idx += 1
        
        # Show progress
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{frame_count}")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Processing complete. Output video saved to {output_path}")
    print(f"Visualization frames saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Double video FPS using simple frame averaging')
    parser.add_argument('input', help='Input video file (MP4)')
    parser.add_argument('--output', default='output_averaged.mp4', help='Output video file')
    parser.add_argument('--vis-folder', default='average_visualization_frames', help='Folder for visualization frames')
    parser.add_argument('--vis-count', type=int, default=10, help='Number of frames to visualize')
    
    args = parser.parse_args()
    
    double_fps_with_average(args.input, args.output, args.vis_folder, args.vis_count)
