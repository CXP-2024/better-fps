import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow between two frames using Farneback method"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def visualize_flow(flow):
    """Convert optical flow to RGB visualization"""
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    
    # Convert flow to polar coordinates (magnitude, angle)
    magnitude, angle = cv2.cartToPolar(fx, fy)
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue according to direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value according to magnitude
    
    # Convert HSV to BGR for visualization
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_vis

def interpolate_frame(prev_frame, curr_frame, flow, t=0.5):
    """Interpolate an intermediate frame using optical flow"""
    h, w = prev_frame.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Calculate the pixel coordinates in the previous frame
    fx, fy = flow[:,:,0] * t, flow[:,:,1] * t
    src_x = (x_coords - fx).clip(0, w-1)
    src_y = (y_coords - fy).clip(0, h-1)
    
    # Interpolate the intermediate frame
    interpolated = np.zeros_like(prev_frame)
    
    # For each channel
    for i in range(3):
        interpolated[:,:,i] = cv2.remap(prev_frame[:,:,i], src_x, src_y, cv2.INTER_LINEAR)
    
    return interpolated.astype(np.uint8)

def create_flow_color_wheel(size=200, margin=20):
    """
    Create a color wheel legend image explaining optical flow colors
    
    Args:
        size: Size of the wheel image in pixels
        margin: Margin around the wheel for text
    
    Returns:
        Color wheel image with annotations
    """
    # Create a black background image
    img_size = size + 2 * margin
    wheel = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Draw the color wheel
    center = img_size // 2
    radius = size // 2
    
    # Generate HSV color wheel
    for y in range(img_size):
        for x in range(img_size):
            # Calculate distance from center
            dx = x - center
            dy = y - center
            distance = np.sqrt(dx**2 + dy**2)
            
            # Within the circle
            if distance <= radius:
                # Calculate angle and normalize to [0, 1]
                angle = np.arctan2(dy, dx) + np.pi
                angle_normalized = angle / (2 * np.pi)
                
                # Calculate normalized distance [0, 1]
                distance_normalized = distance / radius
                
                # Convert to HSV and then to BGR
                h = angle_normalized * 180
                s = distance_normalized * 255
                v = 255
                
                # Set pixel color
                wheel[y, x] = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    
    # Add annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Direction labels
    cv2.putText(wheel, "Right", (center + radius + 5, center), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(wheel, "Left", (center - radius - 40, center), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(wheel, "Down", (center - 20, center + radius + 20), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(wheel, "Up", (center - 10, center - radius - 10), font, font_scale, (255, 255, 255), thickness)
    
    # Title and explanation
    cv2.putText(wheel, "Optical Flow Color Legend", (10, 20), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(wheel, "Hue: Direction of motion", (10, img_size - 40), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(wheel, "Saturation: Magnitude of motion", (10, img_size - 20), font, font_scale, (255, 255, 255), thickness)
    
    return wheel

def double_fps_with_optical_flow(input_path, output_path, output_folder, num_visualize=5):
    """
    Double the FPS of a video using optical flow interpolation and save visualizations
    
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
    flow_folder = os.path.join(output_folder, "flow")
    
    # Create all subfolders
    os.makedirs(originals_folder, exist_ok=True)
    os.makedirs(interpolated_folder, exist_ok=True)
    os.makedirs(combined_folder, exist_ok=True)
    os.makedirs(flow_folder, exist_ok=True)
    
    # Create and save optical flow color wheel legend
    color_wheel = create_flow_color_wheel()
    cv2.imwrite(os.path.join(flow_folder, "color_legend.jpg"), color_wheel)
    
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
        
        # Calculate optical flow from prev_frame to curr_frame
        flow = calculate_optical_flow(prev_frame, curr_frame)
        
        # Interpolate intermediate frame
        interpolated = interpolate_frame(prev_frame, curr_frame, flow)
        
        # Add interpolated and current frames to output video
        out.write(interpolated)
        out.write(curr_frame)
        
        # If this is a frame we want to visualize
        if visual_start <= frame_idx <= visual_end:
            # Visualize optical flow
            flow_vis = visualize_flow(flow)
            
            # Get current visualization index
            vis_idx = len(visualization_frames)
            
            # Save original frames with consecutive numbering
            cv2.imwrite(f"{originals_folder}/{vis_idx*2}.jpg", prev_frame)
            # cv2.imwrite(f"{originals_folder}/{vis_idx*2+1}.jpg", curr_frame), this we needn't show
            
            # Save interpolated frame with consecutive numbering
            cv2.imwrite(f"{interpolated_folder}/{vis_idx}.jpg", interpolated)
            
            # Save combined frames (originals and interpolated) with consecutive numbering
            cv2.imwrite(f"{combined_folder}/{vis_idx*2}.jpg", prev_frame)
            cv2.imwrite(f"{combined_folder}/{vis_idx*2+1}.jpg", interpolated)
            if vis_idx < num_visualize - 1:  # Don't duplicate the last frame
                cv2.imwrite(f"{combined_folder}/{vis_idx*2+2}.jpg", curr_frame)
            
            # Save optical flow visualization with consecutive numbering
            cv2.imwrite(f"{flow_folder}/{vis_idx}.jpg", flow_vis)
            
            # Store frames for visualization
            visualization_frames.append((prev_frame.copy(), curr_frame.copy(), interpolated.copy(), flow_vis))
        
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
    parser = argparse.ArgumentParser(description='Double video FPS using optical flow')
    parser.add_argument('input', help='Input video file (MP4)')
    parser.add_argument('--output', default='output_doubled.mp4', help='Output video file')
    parser.add_argument('--vis-folder', default='visualization_frames', help='Folder for visualization frames')
    parser.add_argument('--vis-count', type=int, default=10, help='Number of frames to visualize')
    
    args = parser.parse_args()
    
    double_fps_with_optical_flow(args.input, args.output, args.vis_folder, args.vis_count)
