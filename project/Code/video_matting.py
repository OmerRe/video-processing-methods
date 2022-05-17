import cv2
import numpy as np
from scipy.stats import gaussian_kde
from Code.utils import geodesic_distance_2d, convert_frames_color


def create_scribbles(mask: np.ndarray, type: str, num_of_points: int) -> np.ndarray:
    if type == 'foreground':
        indices = np.where(mask == 255)
    elif type == 'background':
        indices = np.where(mask == 0)
    scribbles = np.zeros((num_of_points, 2))
    chosen_indices = np.random.choice(len(indices[0]), num_of_points)
    scribbles[:, 0] = indices[0][chosen_indices]
    scribbles[:, 1] = indices[1][chosen_indices]
    return scribbles

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

def video_matting(stabilized_frames, binary_frames, background_image, config):
    frames_bgr = stabilized_frames
    frames_hsv = convert_frames_color(stabilized_frames, color_space=cv2.COLOR_BGR2HSV)
    # frames_binary = convert_frames_color(binary_frames, color_space=cv2.COLOR_BGR2GRAY)
    n_frames = config['video_params']['n_frames']
    w = config['video_params']['w']
    h = config['video_params']['h']

    '''Resize new background'''
    new_background = cv2.resize(background_image, (w, h))
    num_of_points = 100
    '''Starting Matting Process'''
    full_matted_frames_list, alpha_frames_list = [], []
    for frame_index in range(n_frames):
        print(f'Stage 3, Matting: - Frame: {frame_index} / {n_frames}')
        _, _, gray_frame = cv2.split(frames_hsv[frame_index])
        bgr_frame = frames_bgr[frame_index]
        binary_frame = binary_frames[frame_index]
        binary_frame = ((binary_frame > 200).astype(np.uint8))*255

        '''Resizing frames'''
        DELTA = 20
        y_foreground_part = np.where(binary_frame == 255)[0]
        x_foreground_part = np.where(binary_frame == 255)[1]
        left_index, right_index = max(0, np.min(x_foreground_part) - DELTA), min(w - 1, np.max(x_foreground_part) + DELTA)
        top_index, bottom_index = max(0, np.min(y_foreground_part) - DELTA), min(h - 1, np.max(y_foreground_part) + DELTA)

        smaller_gray_frame = gray_frame[top_index:bottom_index, left_index:right_index]
        smaller_bgr_frame = bgr_frame[top_index:bottom_index, left_index:right_index]
        smaller_binary_frame = binary_frame[top_index:bottom_index, left_index:right_index]
        smaller_new_background = new_background[top_index:bottom_index, left_index:right_index]

        # create trimap
        d_kernel = np.ones((3, 3))
        erode = cv2.erode(smaller_binary_frame, d_kernel, iterations=2)
        dilate = cv2.dilate(smaller_binary_frame, d_kernel, iterations=1)
        unknown1 = cv2.bitwise_xor(erode, smaller_binary_frame)
        unknown2 = cv2.bitwise_xor(dilate, smaller_binary_frame)
        unknowns = cv2.add(unknown1, unknown2)
        unknowns[unknowns == 255] = 127
        trimap = (smaller_binary_frame + unknowns)

        foreground_scribbles = create_scribbles(trimap, 'foreground', num_of_points)
        background_scribbles = create_scribbles(trimap, 'background', num_of_points)
        foreground_values = gray_frame[np.uint32(foreground_scribbles[:, 0]), np.uint32(foreground_scribbles[:, 1])]
        background_values = gray_frame[np.uint32(background_scribbles[:, 0]), np.uint32(background_scribbles[:, 1])]
        undecided_region = (trimap == 127).astype(np.uint8) * 255
        undecided_x_indices = (np.where(undecided_region == 255))[1]
        undecided_y_indices = (np.where(undecided_region == 255))[0]
        undecided_region_colors = smaller_gray_frame[undecided_y_indices, undecided_x_indices]

        foreground_pdf_undecided_region = kde_scipy(foreground_values, undecided_region_colors)
        background_pdf_undecided_region = kde_scipy(background_values, undecided_region_colors)
        posterior_foreground_undecided_region = foreground_pdf_undecided_region / (foreground_pdf_undecided_region + background_pdf_undecided_region)
        posterior_background_undecided_region = background_pdf_undecided_region / (foreground_pdf_undecided_region + background_pdf_undecided_region)

        foreground_region = (trimap == 255).astype(np.uint8) * 255
        foreground_x_indices = (np.where(foreground_region == 255))[1]
        foreground_y_indices = (np.where(foreground_region == 255))[0]
        background_region = (trimap == 255).astype(np.uint8) * 255
        background_x_indices = (np.where(background_region == 0))[1]
        background_y_indices = (np.where(background_region == 0))[0]

        posterior_foreground_map = np.zeros(smaller_binary_frame.shape)
        posterior_foreground_map[foreground_y_indices, foreground_x_indices] = 1
        posterior_foreground_map[background_y_indices, background_x_indices] = 0
        posterior_foreground_map[undecided_y_indices, undecided_x_indices] = posterior_foreground_undecided_region

        posterior_background_map = np.zeros(smaller_binary_frame.shape)
        posterior_background_map[foreground_y_indices, foreground_x_indices] = 0
        posterior_background_map[background_y_indices, background_x_indices] = 1
        posterior_background_map[undecided_y_indices, undecided_x_indices] = posterior_background_undecided_region

        foreground_seeds = np.zeros(smaller_binary_frame.shape)
        background_seeds = np.zeros(smaller_binary_frame.shape)
        foreground_seeds[np.uint32(foreground_scribbles[:, 0]), np.uint32(foreground_scribbles[:, 1])] = 1
        background_seeds[np.uint32(background_scribbles[:, 0]), np.uint32(background_scribbles[:, 1])] = 1
        foreground_distance_map = geodesic_distance_2d(np.float32(posterior_foreground_map), np.uint8(foreground_seeds), 1.0, 1)
        background_distance_map = geodesic_distance_2d(np.float32(posterior_background_map),  np.uint8(background_seeds), 1.0, 1)

        w_f = np.power(foreground_distance_map[undecided_y_indices, undecided_x_indices], -2) * posterior_foreground_map[undecided_y_indices, undecided_x_indices]
        w_b = np.power(background_distance_map[undecided_y_indices, undecided_x_indices], -2) * posterior_background_map[undecided_y_indices, undecided_x_indices]
        small_alpha_narrow_band = w_f / (w_f + w_b)

        small_alpha_frame = np.zeros(smaller_binary_frame.shape)
        small_alpha_frame[foreground_y_indices, foreground_x_indices] = 1
        small_alpha_frame[background_y_indices, background_x_indices] = 0
        small_alpha_frame[undecided_y_indices, undecided_x_indices] = small_alpha_narrow_band

        # '''Naive implementation for matting as described in algorithm'''
        # bgr_frame_copy = smaller_bgr_frame.copy()
        # window_size = 10
        # h_smaller = smaller_bgr_frame.shape[0]
        # w_smaller = smaller_bgr_frame.shape[1]
        # for idx in range(len(undecided_y_indices)):
        #     print(f'idx: {idx}')
        #     y = undecided_y_indices[idx]
        #     x = undecided_x_indices[idx]
        #     color = smaller_bgr_frame[y, x]
        #     top_index, bottom_index = max(0, y - window_size), min(h_smaller - 1, y + window_size)
        #     left_index, right_index = max(0, x - window_size), min(w_smaller - 1, x + window_size)
        #     patch = smaller_bgr_frame[top_index:bottom_index, left_index:right_index]
        #     small_alpha_frame_patch = small_alpha_frame[top_index:bottom_index, left_index:right_index]
        #     foreground_patch_indices = np.where(small_alpha_frame_patch == 1)
        #     background_patch_indices = np.where(small_alpha_frame_patch == 0)
        #     error = 0
        #     if len(foreground_patch_indices[0])*len(background_patch_indices[0]) == 0:
        #         continue
        #     for i in range(len(foreground_patch_indices[0])):
        #         fg_color = patch[foreground_patch_indices[0][i], foreground_patch_indices[1][i]]
        #         for j in range(len(background_patch_indices[0])):
        #             # for j in range(len(background_patch_indices)):
        #             bg_color = patch[background_patch_indices[0][j], background_patch_indices[1][j]]
        #             estimate = small_alpha_frame[y, x, np.newaxis] * fg_color + (1 - small_alpha_frame[y, x, np.newaxis]) * bg_color
        #             diff = color - estimate
        #             error_new = np.linalg.norm(diff)
        #             if error_new < error:
        #                 error = error_new
        #                 bgr_frame_copy[y, x] = fg_color

        smaller_matted_frame = small_alpha_frame[:, :, np.newaxis] * smaller_bgr_frame + (1 - small_alpha_frame[:, :, np.newaxis]) * smaller_new_background
        smaller_matted_frame = smaller_matted_frame.astype(np.uint8)

        '''move from small rectangle to original size'''
        matted_frame = np.copy(new_background)
        matted_frame[top_index:bottom_index, left_index:right_index] = smaller_matted_frame
        full_matted_frames_list.append(matted_frame)

        alpha_frame = np.zeros(binary_frame.shape)
        alpha_frame[top_index:bottom_index, left_index:right_index] = small_alpha_frame
        alpha_frame = (alpha_frame * 255).astype(np.uint8)
        alpha_frames_list.append(alpha_frame)

        return full_matted_frames_list, alpha_frames_list