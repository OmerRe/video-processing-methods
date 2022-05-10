


def check_in_dict(dict, element, function):
    if element in dict:
        return dict[element]
    else:
        dict[element] = function(np.asarray(element))[0]
        return dict[element]


import numpy as np
import cv2
from scipy.stats import gaussian_kde


def fixBorder(frame):
    h, w = frame.shape[0],frame.shape[1]
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame


def get_video_files(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height, fps


def release_video_files(cap):
    cap.release()
    cv2.destroyAllWindows()


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'reflect')

    '''Fix padding manually'''
    for i in range(radius):
        curve_pad[i] = curve_pad[radius] - curve_pad[i]

    for i in range(len(curve_pad) - 1, len(curve_pad) - 1 - radius, -1):
        curve_pad[i] = curve_pad[len(curve_pad) - radius - 1] - curve_pad[i]

    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth(trajectory, smooth_radius):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=smooth_radius)
    return smoothed_trajectory


def write_video(output_path, frames, fps, out_size, is_color):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(output_path, fourcc, fps, out_size, isColor=is_color)
    for frame in frames:
        video_out.write(frame)
    video_out.release()


def scale_matrix_0_to_255(input_matrix):
    if input_matrix.dtype == np.bool:
        input_matrix = np.uint8(input_matrix)
    input_matrix = input_matrix.astype(np.uint8)
    scaled = 255 * (input_matrix - np.min(input_matrix)) / np.ptp(input_matrix)
    return np.uint8(scaled)


def load_entire_video(cap, color_space='bgr'):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        if color_space == 'bgr':
            frames.append(curr)
        elif color_space == 'yuv':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2YUV))
        elif color_space == 'bw':
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames)


def apply_mask_on_color_frame(frame, mask):
    frame_after_mask = np.copy(frame)
    frame_after_mask[:, :, 0] = frame_after_mask[:, :, 0] * mask
    frame_after_mask[:, :, 1] = frame_after_mask[:, :, 1] * mask
    frame_after_mask[:, :, 2] = frame_after_mask[:, :, 2] * mask
    return frame_after_mask


def choose_indices_for_foreground(mask, number_of_choices):
    indices = np.where(mask == 1)
    if len(indices[0]) == 0:
        return np.column_stack((indices[0],indices[1]))
    indices_choices = np.random.choice(len(indices[0]), number_of_choices)
    return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))


def choose_indices_for_background(mask, number_of_choices):
    indices = np.where(mask == 0)
    if len(indices[0]) == 0:
        return np.column_stack((indices[0],indices[1]))
    indices_choices = np.random.choice(len(indices[0]), number_of_choices)
    return np.column_stack((indices[0][indices_choices], indices[1][indices_choices]))

def new_estimate_pdf(omega_values, bw_method):
    pdf = gaussian_kde(omega_values.T, bw_method=bw_method)
    return lambda x: pdf(x.T)

def estimate_pdf(original_frame, indices, bw_method):
    omega_f_values = original_frame[indices[:, 0], indices[:, 1], :]
    pdf = gaussian_kde(omega_f_values.T, bw_method=bw_method)
    return lambda x: pdf(x.T)


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def disk_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))

# font = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (10, 50)
# fontScale = 3
# fontColor = (255, 255, 255)
# lineType = 2
#
# cv2.putText(weighted_mask, str(i),
#             bottomLeftCornerOfText,
#             font,
#             fontScale,
#             fontColor,
#             lineType)
#
# cv2.imshow('s',weighted_mask)
# cv2.waitKey(0)

# # Write the frame to the file
# concat_frame = cv2.hconcat([mask_or, mask_or_erosion])
# # If the image is too big, resize it.
# if concat_frame.shape[1] > 1920:
#     concat_frame = cv2.resize(concat_frame, (int(concat_frame.shape[1]), int(concat_frame.shape[0])))
# cv2.imshow("Before and After", concat_frame)
# cv2.waitKey(0)

# image = np.copy(frame_after_or_and_blue_flt)
# for index in range(chosen_pixels_indices.shape[0]):
#     image = cv2.circle(image, (chosen_pixels_indices[index][1], chosen_pixels_indices[index][0]), 5, (0, 255, 0), 2)
# Displaying the image
# cv2.imshow('sas', image)
# cv2.waitKey(0)

