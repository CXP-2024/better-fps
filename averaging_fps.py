import cv2
import numpy as np
import argparse

def double_fps_with_averaging(input_video, output_video):
    """
    Double the FPS of a video by generating intermediate frames through averaging
    consecutive frames.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object with double the FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps*2, (width, height))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Couldn't read the first frame")
        return
    
    # Write the first frame
    out.write(prev_frame)
    
    # Process remaining frames
    frame_num = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Create an intermediate frame by averaging the previous and current frames
        intermediate_frame = np.uint8((prev_frame.astype(np.int32) + curr_frame.astype(np.int32)) / 2)
        
        # Write the intermediate frame and then the current frame
        out.write(intermediate_frame)
        out.write(curr_frame)
        
        # Update previous frame
        prev_frame = curr_frame
        
        # Print progress
        frame_num += 1
        if frame_num % 30 == 0:
            print(f"Processed {frame_num}/{frame_count} frames ({100 * frame_num / frame_count:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Done! Output saved to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double video FPS by frame averaging")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_video", help="Path for the output video file")
    args = parser.parse_args()
    
    double_fps_with_averaging(args.input_video, args.output_video)
