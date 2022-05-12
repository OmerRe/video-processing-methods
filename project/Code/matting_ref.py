import GeodisTK
import cv2
import numpy as np
import logging
from constants import (
    EPSILON_NARROW_BAND,
    ERODE_ITERATIONS,
    DILATE_ITERATIONS,
    GEODISTK_ITERATIONS,
    KDE_BW,
    R
)
from utils_ref import (
    load_entire_video,
    get_video_files,
    choose_indices_for_foreground,
    choose_indices_for_background,
    write_video,
    fixBorder,
    estimate_pdf
)

my_logger = logging.getLogger('MyLogger')


def video_matting(input_stabilize_video, binary_video_path, new_background):
    my_logger.info('Starting Matting')

    # Read input video
    cap_stabilize, w, h, fps_stabilize = get_video_files(path=input_stabilize_video)
    cap_binary, _, _, fps_binary = get_video_files(path=binary_video_path)

    # Get frame count
    n_frames = int(cap_stabilize.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_bgr = load_entire_video(cap_stabilize, color_space='bgr')
    frames_yuv = load_entire_video(cap_stabilize, color_space='yuv')
    frames_binary = load_entire_video(cap_binary, color_space='bw')

    '''Resize new background'''
    new_background = cv2.resize(new_background, (w, h))

    '''Starting Matting Process'''
    full_matted_frames_list, alpha_frames_list = [], []
    for frame_index in range(n_frames):
        print(f'[Matting] - Frame: {frame_index} / {n_frames}')
        luma_frame, _, _ = cv2.split(frames_yuv[frame_index])
        bgr_frame = frames_bgr[frame_index]

        original_mask_frame = frames_binary[frame_index]
        original_mask_frame = (original_mask_frame > 150).astype(np.uint8)

        '''Find indices for resizing image to work only on relevant part!'''
        DELTA = 20
        binary_frame_rectangle_x_axis = np.where(original_mask_frame == 1)[1]
        left_index, right_index = np.min(binary_frame_rectangle_x_axis), np.max(binary_frame_rectangle_x_axis)
        left_index, right_index = max(0, left_index - DELTA), min(right_index + DELTA, original_mask_frame.shape[1] - 1)
        binary_frame_rectangle_y_axis = np.where(original_mask_frame == 1)[0]
        top_index, bottom_index = np.min(binary_frame_rectangle_y_axis), np.max(binary_frame_rectangle_y_axis)
        top_index, bottom_index = max(0, top_index - DELTA), min(bottom_index + DELTA, original_mask_frame.shape[0] - 1)

        ''' Resize images '''
        smaller_luma_frame = luma_frame[top_index:bottom_index, left_index:right_index]
        smaller_bgr_frame = bgr_frame[top_index:bottom_index, left_index:right_index]
        smaller_new_background = new_background[top_index:bottom_index, left_index:right_index]

        '''Erode & Resize foreground mask & Build distance map for foreground'''
        foreground_mask = cv2.erode(original_mask_frame, np.ones((3, 3)), iterations=ERODE_ITERATIONS)
        smaller_foreground_mask = foreground_mask[top_index:bottom_index, left_index:right_index]
        smaller_foreground_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_foreground_mask,
                                                                          1.0, GEODISTK_ITERATIONS)

        '''Dilate & Resize image & Build distance map for background'''
        background_mask = cv2.dilate(original_mask_frame, np.ones((3, 3)), iterations=DILATE_ITERATIONS)
        background_mask = 1 - background_mask
        smaller_background_mask = background_mask[top_index:bottom_index, left_index:right_index]
        smaller_background_distance_map = GeodisTK.geodesic2d_raster_scan(smaller_luma_frame, smaller_background_mask,
                                                                          1.0, GEODISTK_ITERATIONS)

        ''' Building narrow band undecided zone'''
        smaller_foreground_distance_map = smaller_foreground_distance_map / (smaller_foreground_distance_map + smaller_background_distance_map)
        smaller_background_distance_map = 1 - smaller_foreground_distance_map
        smaller_narrow_band_mask = (np.abs(smaller_foreground_distance_map - smaller_background_distance_map) < EPSILON_NARROW_BAND).astype(np.uint8)
        smaller_narrow_band_mask_indices = np.where(smaller_narrow_band_mask == 1)

        smaller_decided_foreground_mask = (smaller_foreground_distance_map < smaller_background_distance_map - EPSILON_NARROW_BAND).astype(np.uint8)
        smaller_decided_background_mask = (smaller_background_distance_map >= smaller_foreground_distance_map - EPSILON_NARROW_BAND).astype(np.uint8)
        '''Building KDEs for foreground & background to calculate priors for alpha calculation'''
        omega_f_indices = choose_indices_for_foreground(smaller_decided_foreground_mask, 200)
        omega_b_indices = choose_indices_for_background(smaller_decided_background_mask, 200)
        foreground_pdf = estimate_pdf(original_frame=smaller_bgr_frame, indices=omega_f_indices, bw_method=KDE_BW)
        background_pdf = estimate_pdf(original_frame=smaller_bgr_frame, indices=omega_b_indices, bw_method=KDE_BW)
        smaller_narrow_band_foreground_probs = foreground_pdf(smaller_bgr_frame[smaller_narrow_band_mask_indices])
        smaller_narrow_band_background_probs = background_pdf(smaller_bgr_frame[smaller_narrow_band_mask_indices])

        '''Start creating alpha map'''
        w_f = np.power(smaller_foreground_distance_map[smaller_narrow_band_mask_indices],-R) * smaller_narrow_band_foreground_probs
        w_b = np.power(smaller_background_distance_map[smaller_narrow_band_mask_indices],-R) * smaller_narrow_band_background_probs
        alpha_narrow_band = w_f / (w_f + w_b)
        smaller_alpha = np.copy(smaller_decided_foreground_mask).astype(np.float)
        smaller_alpha[smaller_narrow_band_mask_indices] = alpha_narrow_band

        '''Naive implementation for matting as described in algorithm'''
        smaller_matted_frame = smaller_alpha[:, :, np.newaxis] * smaller_bgr_frame + (1 - smaller_alpha[:, :, np.newaxis]) * smaller_new_background

        '''move from small rectangle to original size'''
        full_matted_frame = np.copy(new_background)
        full_matted_frame[top_index:bottom_index, left_index:right_index] = smaller_matted_frame
        full_matted_frames_list.append(full_matted_frame)

        full_alpha_frame = np.zeros(original_mask_frame.shape)
        full_alpha_frame[top_index:bottom_index, left_index:right_index] = smaller_alpha
        full_alpha_frame = (full_alpha_frame * 255).astype(np.uint8)
        alpha_frames_list.append(full_alpha_frame)

    write_video(output_path='../Outputs/matted.avi', frames=full_matted_frames_list, fps=fps_stabilize, out_size=(w, h),
                is_color=True)
    write_video(output_path='../Outputs/alpha.avi', frames=alpha_frames_list, fps=fps_stabilize, out_size=(w, h), is_color=False)
    print('~~~~~~~~~~~ [Matting] FINISHED! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ matted.avi has been created! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ alpha.avi has been created! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ unstabilized_alpha.avi has been created! ~~~~~~~~~~~')
    my_logger.info('Finished Matting')



CONFIG = {
    'ID_1': 302828991,
    'ID_2': 316524800,
    'MAX_CORNERS': 500,
    'QUALITY_LEVEL': 0.01,
    'MIN_DISTANCE': 30,
    'BLOCK_SIZE': 3,
    'SMOOTHING_RADIUS': 5,

}
##### TODO: Remove before assign
# input_video = '../Outputs/stabilized_302828991_316524800.avi'
# binary_video = '../Outputs/binary.avi'
# background_image = cv2.imread('../Inputs/background.jpg')
# # subtruct_background_by_median(input_video)
# # subtruct_background_by_MOG2(input_video)
# video_matting(input_video, binary_video, background_image)
#
# # Destroy all windows
# cv2.destroyAllWindows()
