import cv2
import numpy as np
import argparse

def calculate_optical_flow(frame1, frame2):
    """Calculate optical flow between two frames using Farneback algorithm."""
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    return flow

def warp_frame(frame, flow, t=0.5):
    """Warp frame according to optical flow vectors with weight t."""
    h, w = flow.shape[:2]
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    
    # Create mapping grid
    for y in range(h):
        for x in range(w):
            flow_map[y, x, 0] = x + t * flow[y, x, 0]
            flow_map[y, x, 1] = y + t * flow[y, x, 1]
    
    # Warp the frame using the flow map
    warped_frame = cv2.remap(
        frame, flow_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped_frame

def interpolate_frame(frame1, frame2):
    """Generate an intermediate frame between two consecutive frames."""
    # Calculate forward and backward optical flows
    flow_forward = calculate_optical_flow(frame1, frame2)
    flow_backward = calculate_optical_flow(frame2, frame1)
    
    # Warp both frames
    warped_frame1 = warp_frame(frame1, flow_forward, 0.5)
    warped_frame2 = warp_frame(frame2, flow_backward, 0.5)
    
    # Blend the warped frames for smoother results
    interpolated_frame = cv2.addWeighted(warped_frame1, 0.5, warped_frame2, 0.5, 0)
    
    return interpolated_frame

def double_video_fps(input_path, output_path):
    """Double the frame rate of a video using optical flow interpolation."""
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer with double the FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps * 2, (width, height))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read from video file")
        return
    
    # Write the first frame
    out.write(prev_frame)
    
    frame_idx = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Interpolate a frame between prev_frame and curr_frame
        mid_frame = interpolate_frame(prev_frame, curr_frame)
        
        # Write interpolated frame and current frame
        out.write(mid_frame)
        out.write(curr_frame)
        
        # Update prev_frame
        prev_frame = curr_frame
        
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{frame_count}", end='\r')
    
    # Release resources
    cap.release()
    out.release()
    print(f"\nFrame interpolation complete. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double video FPS using optical flow interpolation")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path")
    args = parser.parse_args()
    
    double_video_fps(args.input, args.output)
