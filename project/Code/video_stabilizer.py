import os
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from Code.utils import extract_video_parameters, release_video, load_video, write_video, fixBorder, convert_to_gray


def stabilize_video(input_video: cv2.VideoCapture, config: dict) -> None:
    """Creating a stabilized video from an arbitrary input video.
    Args:
        input_video: cv2.VideoCapture. Video we want to stabilize.
        config: dict. Dictionary which contains useful constants.
    Returns:
        None, but creates stabilized video from the input video.
    Details:

    """
    print("Starting Video Stabilization...")
    video_params = extract_video_parameters(input_video)
    video_frames = load_video(input_video)

    transforms = find_motion_between_frames(video_params, video_frames, config)
    transforms_smooth = calc_smooth_transforms(config, transforms)
    stabilized_frames = apply_smooth_motion_to_frames(video_params, video_frames, transforms_smooth)

    release_video(input_video)
    write_video(f'../Outputs/stabilized_{config["ID_1"]}_{config["ID_2"]}.avi', stabilized_frames,
                video_params['fps'], (video_params['w'], video_params['h']), is_color=True)
    print("Video Stabilization Finished")


def find_motion_between_frames(video_params: dict, video_frames: list,  config: dict) -> np.ndarray:
    # Pre-define transformation-store array
    transforms = np.zeros((video_params['n_frames'] - 1, 9), np.float32)
    prev_frame_gray = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)

    for frame_idx, current_frame in enumerate(video_frames[1:]):
        # Detecting feature points in previous frame
        prev_frame_pts = cv2.goodFeaturesToTrack(prev_frame_gray,
                                                 maxCorners=config['MAX_CORNERS'],
                                                 qualityLevel=config['QUALITY_LEVEL'],
                                                 minDistance=config['MIN_DISTANCE'],
                                                 blockSize=config['BLOCK_SIZE'])
        current_frame_gray = convert_to_gray(current_frame)

        # Calculating optical flow and keeping only the valid features points
        curr_frame_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, prev_frame_pts, None)
        idx = np.where(status == 1)[0]
        prev_frame_pts, curr_frame_pts = prev_frame_pts[idx], curr_frame_pts[idx]

        # Finding transformation matrix
        transform_matrix, _ = cv2.findHomography(prev_frame_pts, curr_frame_pts)
        transforms[frame_idx] = transform_matrix.flatten()

        print(f"Video Stabilizing: calculating transformation for frame: {frame_idx + 1} "
              f"/ {video_params['n_frames'] - 1} -  Tracked points: {len(prev_frame_pts)}")

        prev_frame_gray = current_frame_gray
    return transforms


def apply_smooth_motion_to_frames(video_params: dict, video_frames: list, transforms_smooth: np.ndarray) -> list:
    stabilized_frames = [fixBorder(video_frames[0])]
    # Write n_frames-1 transformed frames
    for frame_idx, current_frame in enumerate(video_frames[:-1]):
        print(f"Video Stabilizing: applying transformation to frame: {frame_idx + 1} "
              f"/ {video_params['n_frames'] - 1}")
        transform_matrix = transforms_smooth[frame_idx].reshape((3, 3))
        # Apply homography wrapping to the given frame
        frame_stabilized = cv2.warpPerspective(current_frame, transform_matrix, (video_params['w'], video_params['h']))
        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)
        stabilized_frames.append(frame_stabilized)
    return stabilized_frames


def movingAverage(curve: np.ndarray, radius: int) -> np.ndarray:
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


def smooth(trajectory: np.ndarray, config: dict) -> np.ndarray:
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=config['SMOOTHING_RADIUS'])
    return smoothed_trajectory


def calc_smooth_transforms(config: dict, transforms: np.ndarray) -> np.ndarray:
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory, config)
    # Calculate difference between smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    # Calculate smooth transformation array
    transforms_smooth = transforms + difference
    return transforms_smooth
