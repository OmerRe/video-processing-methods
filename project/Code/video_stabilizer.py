import cv2
import numpy as np
from Code.utils import fixBorder, convert_to_gray


def stabilize_video(video_frames: list, config: dict) -> list:
    """Creating a stabilized video from an arbitrary input video.
    Args:
        input_video: cv2.VideoCapture. Video we want to stabilize.
        config: dict. Dictionary which contains useful constants.
    Returns:
        None, but creates stabilized video from the input video.
    Details:

    """
    print("Starting Video Stabilization...")
    transforms = find_motion_between_frames(config['video_params'], video_frames, config)
    transforms_smooth = calc_smooth_transforms(config, transforms)
    stabilized_frames = apply_smooth_motion_to_frames(config['video_params'], video_frames, transforms_smooth)
    print("Video Stabilization Finished")

    return stabilized_frames


def find_motion_between_frames(video_params: dict, video_frames: list,  config: dict) -> np.ndarray:
    # Pre-define transformation-store array
    transforms = np.zeros((video_params['n_frames'] - 1, 9), np.float32)
    prev_frame_gray = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)

    for frame_idx, current_frame in enumerate(video_frames[1:]):
        # Detecting feature points in previous frame
        prev_frame_pts = []
        curr_frame_pts = []
        current_frame_gray = convert_to_gray(current_frame)
        # Calculating optical flow and keeping only the valid features points
        detector = cv2.FastFeatureDetector.create()
        orb = cv2.ORB_create()
        kp1 = detector.detect(prev_frame_gray, None)
        kp2 = detector.detect(current_frame_gray, None)
        kp1, des1 = orb.compute(prev_frame_gray, kp1)
        kp2, des2 = orb.compute(current_frame_gray, kp2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        # img3 = cv2.drawMatches(prev_frame_gray, kp1, current_frame_gray, kp2, matches, None,
        #                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3), plt.show()

        prev_frame_pts.append(np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2))
        curr_frame_pts.append(np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2))
        prev_frame_pts = np.squeeze(np.array(prev_frame_pts))
        curr_frame_pts = np.squeeze(np.array(curr_frame_pts))

        transform_matrix, mask = cv2.findHomography(prev_frame_pts, curr_frame_pts, cv2.RANSAC, 5.0)
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
