import cv2
import numpy as np

def get_flow_magnitude(flow):
    """Calculate the magnitude of flow vectors."""
    fx, fy = flow[:,:,0], flow[:,:,1]
    return np.sqrt(fx*fx + fy*fy)

def detect_occlusions(flow_forward, flow_backward, threshold=0.5):
    """Detect occlusions based on forward-backward flow consistency."""
    h, w = flow_forward.shape[:2]
    
    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Warp coordinates using forward flow
    pos_x = x_coords + flow_forward[:,:,0]
    pos_y = y_coords + flow_forward[:,:,1]
    
    # Sample backward flow at warped positions (bilinear interpolation)
    back_flow_x = cv2.remap(flow_backward[:,:,0], pos_x, pos_y, cv2.INTER_LINEAR)
    back_flow_y = cv2.remap(flow_backward[:,:,1], pos_x, pos_y, cv2.INTER_LINEAR)
    
    # Calculate flow difference
    diff_x = flow_forward[:,:,0] + back_flow_x
    diff_y = flow_forward[:,:,1] + back_flow_y
    
    # Occlusion map based on forward-backward consistency
    occlusion_map = (diff_x*diff_x + diff_y*diff_y) > threshold
    
    return occlusion_map

def advanced_frame_interpolation(frame1, frame2, num_intermediate=1):
    """Generate multiple intermediate frames with occlusion handling."""
    # Calculate forward and backward flows
    flow_forward = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    flow_backward = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Detect occlusions
    occlusion_forward = detect_occlusions(flow_forward, flow_backward)
    occlusion_backward = detect_occlusions(flow_backward, flow_forward)
    
    # Generate intermediate frames
    intermediate_frames = []
    for i in range(1, num_intermediate + 1):
        t = i / (num_intermediate + 1)
        
        # Scale flows for this intermediate frame
        forward_t = flow_forward * t
        backward_t = flow_backward * (1 - t)
        
        # Create coordinate grid
        h, w = frame1.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Warping coordinates
        pos1_x = x_coords - forward_t[:,:,0]
        pos1_y = y_coords - forward_t[:,:,1]
        
        pos2_x = x_coords + backward_t[:,:,0]
        pos2_y = y_coords + backward_t[:,:,1]
        
        # Warp frames
        warped1 = cv2.remap(frame1, pos1_x, pos1_y, cv2.INTER_LINEAR)
        warped2 = cv2.remap(frame2, pos2_x, pos2_y, cv2.INTER_LINEAR)
        
        # Account for occlusions
        weights1 = (1 - t) * np.expand_dims(~occlusion_forward, axis=2)
        weights2 = t * np.expand_dims(~occlusion_backward, axis=2)
        
        sum_weights = weights1 + weights2
        sum_weights[sum_weights == 0] = 1  # Avoid division by zero
        
        # Blend frames using calculated weights
        interpolated = (warped1 * weights1 + warped2 * weights2) / sum_weights
        intermediate_frames.append(interpolated.astype(np.uint8))
    
    return intermediate_frames

def increase_video_fps(input_path, output_path, target_multiplier=2):
    """Increase video FPS by a specified multiplier using advanced interpolation."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Number of intermediate frames to generate
    num_intermediate = target_multiplier - 1
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps * target_multiplier, (width, height))
    
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
        
        # Generate intermediate frames
        intermediate_frames = advanced_frame_interpolation(prev_frame, curr_frame, num_intermediate)
        
        # Write intermediate frames
        for frame in intermediate_frames:
            out.write(frame)
        
        # Write current frame
        out.write(curr_frame)
        
        # Update prev_frame
        prev_frame = curr_frame
        
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{frame_count}", end='\r')
    
    # Release resources
    cap.release()
    out.release()
    print(f"\nAdvanced frame interpolation complete. Output saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Increase video FPS using advanced optical flow interpolation")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path")
    parser.add_argument("--multiplier", type=int, default=2, help="FPS multiplier (default: 2)")
    args = parser.parse_args()
    
    increase_video_fps(args.input, args.output, args.multiplier)
