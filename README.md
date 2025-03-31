# Better FPS - Traditional Optical Flow Video Interpolation

This project provides tools to increase the frame rate of videos using traditional computer vision techniques based on optical flow, without requiring deep learning.

## Requirements

- Python 3.6+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

## Basic Usage

### Basic Frame Interpolation

The basic interpolation doubles the frame rate of the input video:

```bash
python optical_flow_interpolation.py input_video.mp4 output_video.mp4
```

### Advanced Frame Interpolation

The advanced version provides better quality with occlusion handling:

```bash
python advanced_interpolation.py input_video.mp4 output_video.mp4 --multiplier 2
```

You can adjust the `--multiplier` parameter to increase the frame rate by different factors (2x, 3x, 4x, etc.).

## How It Works

1. **Optical Flow Calculation**: Computes the motion vectors between consecutive frames using the Farneback algorithm
2. **Frame Warping**: Uses the motion vectors to warp frames to intermediate positions
3. **Occlusion Handling**: Detects and handles regions where objects appear or disappear
4. **Frame Blending**: Combines warped frames to create smooth transitions

## Limitations

- Fast motion can cause artifacts in the interpolated frames
- Occlusions (when objects appear/disappear) may not be perfectly handled
- Processing is significantly slower than real-time for high-resolution videos

## Performance Tips

- For faster processing, consider resizing the input video to a smaller resolution
- Adjust the parameters of the optical flow algorithm for a speed/quality tradeoff
