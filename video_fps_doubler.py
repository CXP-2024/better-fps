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

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow between two frames using Farneback method"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

'''
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray, None,  # 前一帧、当前帧、初始光流（None表示无需初始猜测）
    0.5,    # pyr_scale: 金字塔缩放因子（每层缩小为上一层的 0.5 倍）
    3,      # levels: 金字塔层数（3层）
    15,     # winsize: 邻域窗口大小（15x15）
    3,      # iterations: 每层的迭代次数（3次）
    5,      # poly_n: 多项式展开的邻域大小（5x5）
    1.2,    # poly_sigma: 多项式高斯平滑标准差（标准差 1.2）
    0       # flags: 优化选项（0 表示默认）
)'''

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
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value according to magnitude of speed
    
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
    
    # Create temporary file for video without audio
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # Create output video with double FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps*2, (width, height))
    
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
    
    # Process each frame with a progress bar
    with tqdm(total=frame_count-1, desc="Processing frames") as pbar:
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
        
        # Update progress bar
        pbar.update(1)
    
    # Release resources
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
    print(f"Original FPS: {fps}, New FPS: {fps*2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Double video FPS using optical flow')
    parser.add_argument('input', default='jg_480P_30fps_28s.mp4', help='Input video file (MP4)')
    parser.add_argument('--output', default=None, help='Output video file')
    parser.add_argument('--vis-folder', default='visualization_frames', help='Folder for visualization frames')
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
        
        # Create output filename with doubled FPS info
        doubled_fps = fps * 2
        output_filename = f"{base_name}_flow_to_{int(doubled_fps)}{extension}"
        output_path = str(input_path.with_name(output_filename))
    else:
        output_path = args.output
    
    double_fps_with_optical_flow(args.input, output_path, args.vis_folder, args.vis_count)
