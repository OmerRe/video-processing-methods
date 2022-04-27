import os
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image


def stabilize_video(input_video: cv2.VideoCapture, config: dict) -> cv2.VideoCapture:
    # Get frame count
    transforms = find_motion_between_frames(input_video)
    transforms_smooth = calc_smooth_transforms(config, transforms)
    apply_smooth_motion_to_frames(input_video, transforms_smooth, config)
    return input_video

def calc_smooth_transforms(config, transforms):
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory, config)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    # Calculate newer transformation array
    transforms_smooth = transforms + difference
    return transforms_smooth

def read_first_frame(input_video):
    _, prev = input_video.read()
    # Convert frame to grayscale
    return convert_to_gray(prev)

def convert_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def find_motion_between_frames(input_video):
    # Pre-define transformation-store array
    video_params = extract_video_parameters(input_video)
    transforms = np.zeros((video_params['n_frames'] - 1, 3), np.float32)
    prev_frame_gray = read_first_frame(input_video)
    for i in range(video_params['n_frames'] - 2):
        # Detect feature points in previous frame
        prev_frame_pts = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        # Read next frame
        is_frame_return, current_frame = input_video.read()
        if not is_frame_return:
            break

        current_frame_gray = convert_to_gray(current_frame)
        # Calculate optical flow (i.e. track feature points)
        curr_frame_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, prev_frame_pts, None)
        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_frame_pts = prev_frame_pts[idx]
        curr_frame_pts = curr_frame_pts[idx]
        # Find transformation matrix
        m = cv2.estimateAffine2D(prev_frame_pts, curr_frame_pts)[0]  # will only work with OpenCV-3 or less
        # Extract translation
        dx = m[0, 2]
        dy = m[1, 2]
        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])
        # Store transformation
        transforms[i] = [dx, dy, da]
        # Move to next frame
        prev_frame_gray = current_frame_gray
        print("Frame: " + str(i) + "/" + str(video_params['n_frames']) + " -  Tracked points : " + str(len(prev_frame_pts)))

    return transforms

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve

    return curve_smoothed

def smooth(trajectory, config):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius= config['SMOOTHING_RADIUS'])

    return smoothed_trajectory

def apply_smooth_motion_to_frames(input_video, transforms_smooth, config):
    # Reset stream to first frame
    input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    video_params = extract_video_parameters(input_video)
    output_video = cv2.VideoWriter(f'../Outputs/stabilized_{config["ID_1"]}_{config["ID_2"]}.avi', video_params['fourcc'], video_params['fps'], (video_params['w'], video_params['h']))

    # Write n_frames-1 transformed frames
    for i in range(video_params['n_frames'] - 2):
        # Read next frame
        is_frame_return, frame = input_video.read()
        if not is_frame_return:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (video_params['w'], video_params['h']))

        # Write the frame to the file
        # frame_out = cv2.hconcat([frame, frame_stabilized])

        # cv2.imshow("Before and After", frame_out)
        output_video.write(frame_stabilized)


def extract_video_parameters(input_video):
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # Get width and height of video stream
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    n_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    return dict(fourcc=fourcc, fps=fps, h=h, w=w, n_frames=n_frames)