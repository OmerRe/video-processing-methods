import os
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
from Code.utils import extract_video_parameters, release_video, load_video, write_video, fixBorder, convert_to_gray, \
    apply_mask, scale_matrix_0_to_255, geodesic_distance_2d
from scipy.stats import gaussian_kde

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

def video_matting(binary_video: cv2.VideoCapture, input_video: cv2.VideoCapture, config: dict):
    video_params = extract_video_parameters(input_video)
    n_frames = video_params['n_frames']
    HSV_frames = load_video(input_video, 'hsv')
    mask_frames = load_video(binary_video, 'gray')
    num_of_points = 200
    foreground_values = np.zeros((num_of_points, 1))
    background_values = np.zeros((num_of_points, 1))
    trimap = np.zeros((n_frames, video_params['h'], video_params['w'])).astype(np.uint8)

    for frame_index, frame in enumerate(mask_frames):
        print(f"[VM] - Frame: {frame_index + 1} / {n_frames}")
        foreground_scribbles = create_scribbles(frame, 'foreground', num_of_points)
        background_scribbles = create_scribbles(frame, 'background', num_of_points)
        frame_v_channel = HSV_frames[frame_index][:, :, 2]
        frame_v_channel = cv2.equalizeHist(frame_v_channel)
        foreground_values = frame_v_channel[np.uint32(foreground_scribbles[:, 0]), np.uint32(foreground_scribbles[:, 1])]
        background_values = frame_v_channel[np.uint32(background_scribbles[:, 0]), np.uint32(background_scribbles[:, 1])]
        colors = (np.reshape(frame_v_channel, (frame_v_channel.shape[0]*frame_v_channel.shape[1])))
        foreground_pdf = kde_scipy(foreground_values, colors)
        background_pdf = kde_scipy(background_values, colors)
        posterior_foreground_map = np.reshape((foreground_pdf / (foreground_pdf + background_pdf)), (frame.shape[0], frame.shape[1]))
        posterior_background_map = np.reshape((background_pdf / (foreground_pdf + background_pdf)), (frame.shape[0], frame.shape[1]))

        foreground_seeds = np.zeros(frame.shape)
        background_seeds = np.zeros(frame.shape)
        foreground_seeds[np.uint32(foreground_scribbles[:, 0]), np.uint32(foreground_scribbles[:, 1])] = 1
        background_seeds[np.uint32(background_scribbles[:, 0]), np.uint32(background_scribbles[:, 1])] = 1
        foreground_distance_map = geodesic_distance_2d(np.float32(posterior_foreground_map), np.uint8(foreground_seeds), 1.0, 1)
        background_distance_map = geodesic_distance_2d(np.float32(posterior_background_map),  np.uint8(background_seeds), 1.0, 1)
        trimap_frame = np.uint8((foreground_distance_map <= background_distance_map))

        # focusing on narrow band
        narrow_band_mask = (np.abs(foreground_distance_map - background_distance_map) < 0.99).astype(np.uint8)
        narrow_band_mask_indices = np.where(narrow_band_mask == 1)

        smaller_decided_foreground_mask = (foreground_distance_map < background_distance_map - 0.99).astype(np.uint8)
        smaller_decided_background_mask = (background_distance_map >= foreground_distance_map - 0.99).astype(np.uint8)

        # smaller_foreground_scribbles = create_scribbles(smaller_decided_foreground_mask, 'foreground', 200)
        #         # smaller_background_scribbles = create_scribbles(smaller_decided_background_mask, 'background', 200)
        #         # smaller_foreground_values = frame_v_channel[np.uint32(smaller_foreground_scribbles[:, 0]), np.uint32(smaller_foreground_scribbles[:, 1])]
        #         # smaller_background_values = frame_v_channel[np.uint32(smaller_background_scribbles[:, 0]), np.uint32(smaller_background_scribbles[:, 1])]
        #         #
        #         # colors_narrow_band = frame_v_channel[narrow_band_mask_indices]
        #         # colors_narrow_band = np.reshape(colors_narrow_band, (colors_narrow_band.shape[0]*colors_narrow_band.shape[1]))
        #         #
        #         # foreground_pdf = kde_scipy(smaller_foreground_values, colors_narrow_band, 1)
        #         # background_pdf = kde_scipy(smaller_background_values, colors_narrow_band, 1)
        #         #
        #         # posterior_foreground_map = np.reshape((foreground_pdf / (foreground_pdf + background_pdf)), (frame.shape[0], frame.shape[1]))
        #         # posterior_background_map = np.reshape((background_pdf / (foreground_pdf + background_pdf)), (frame.shape[0], frame.shape[1]))

        w_f = np.power(foreground_distance_map[narrow_band_mask_indices], -2) * posterior_foreground_map[narrow_band_mask_indices]
        w_b = np.power(background_distance_map[narrow_band_mask_indices], -2) * posterior_background_map[narrow_band_mask_indices]
        alpha_narrow_band = w_f / (w_f + w_b)

        alpha_map = np.uint8((foreground_distance_map <= background_distance_map))
        alpha_map[narrow_band_mask_indices] = alpha_narrow_band


        trimap[frame_index] = trimap_frame*255

    write_video(f'../Outputs/trimap_{config["ID_1"]}_{config["ID_2"]}.avi', trimap,
                video_params['fps'], (video_params['w'], video_params['h']), is_color=False)




