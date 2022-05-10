import cv2
import numpy as np
import matplotlib.pyplot as plt

from Code.utils import extract_video_parameters, write_video, load_video, apply_mask_on_color_frame


def subtruct_background_by_median(input_frames):
    params = extract_video_parameters(input_frames)
    number_of_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    selected_frames = np.arange(0, number_of_frames, int(number_of_frames / 5))
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
    # create the kernel that will be used to remove the noise in the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    frames = []
    for frameOI in selected_frames:
        input_frames.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
        ret, frame = input_frames.read()
        frames.append(frame)

    grayMedianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    row, col, ch = params['h'], params['w'], 3
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss_noise = gauss.reshape(row, col, ch)
    input_frames.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = input_frames.read()
        if not ret:
            break
        grayMedianFrame_with_noise = cv2.blur(grayMedianFrame, (5, 33), 0)
        frame_with_noise = cv2.blur(frame, (5, 33), 0)
        dframe = cv2.absdiff(frame_with_noise, grayMedianFrame_with_noise)
        # th, dframe = cv2.threshold(dframe, 50, 255, cv2.THRESH_BINARY)
        dframe = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)

        # get the foreground mask
        foregroundMask = bgSubtractor.apply(frame)

        # remove some of noise
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)

        # dframe_hsv = cv2.cvtColor(dframe, cv2.COLOR_BGR2YUV)
        # mask = cv2.inRange(dframe_hsv, (120, 0, 0), (240, 190, 255))
        # bak = dframe.copy()
        #
        # # replace with black
        # bak[mask > 0] = (0, 0, 0)
        # Display image
        aggregated_frame = cv2.bitwise_and(dframe, foregroundMask)
        # aggregated_frame = (aggregated_frame > 200).astype(np.uint8)
        th, aggregated_frame = cv2.threshold(aggregated_frame, 37, 255, cv2.THRESH_BINARY)
        # aggregated_frame = cv2.morphologyEx(aggregated_frame, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame', aggregated_frame)
        cv2.waitKey(1)

def subtruct_background_by_MOG2(input_frames):
    # creation of a videoCapture object with the opening of a video file or a capture device
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
    # create the kernel that will be used to remove the noise in the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        # get the next picture
        ret, frame = input_frames.read()

        if ret:
            # get the foreground mask
            foregroundMask = bgSubtractor.apply(frame)

            # remove some of noise
            foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)

            cv2.imshow('subtraction', foregroundMask)

            if cv2.waitKey(30) == ord('e'):
                break

def background_subtraction_ver15(input_video: cv2.VideoCapture, config: dict):
    print("Starting Background Subtraction...")
    video_params = extract_video_parameters(input_video)
    video_frames_bgr = load_video(input_video, color_space='bgr')
    video_frames_hsv = load_video(input_video, color_space='hsv')
    n_frames = len(video_frames_bgr)

    backSub = cv2.createBackgroundSubtractorKNN()
    mask_list = np.zeros((n_frames, video_params['h'], video_params['w'])).astype(np.uint8)
    masks = np.zeros((n_frames, video_params['h'], video_params['w'])).astype(np.uint8)
    after_mask = np.zeros((n_frames, video_params['h'], video_params['w'], 3)).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    print(f"[BS] - BackgroundSubtractorKNN Studying Frames history")
    for j in range(5):
        print(f"[BS] - BackgroundSubtractorKNN {j + 1} / 8 pass")
        for index_frame, frame in enumerate(video_frames_hsv):
            frame = cv2.GaussianBlur(frame, (9, 9), 0)
            frame_sv = frame[:, :, 1:]
            fgMask = backSub.apply(frame_sv)
            fgMask = (fgMask > 200).astype(np.uint8)
            mask_list[index_frame] = fgMask
    print(f"[BS] - BackgroundSubtractorKNN Finished")

    '''Collecting colors for building body & shoes KDEs'''
    for frame_index, frame in enumerate(video_frames_bgr):
        print(f"[BS] - Collecting colors for building body & shoes KDEs , Frame: {frame_index + 1} / {n_frames}")
        blue_frame, _, _ = cv2.split(frame)
        mask_for_frame = mask_list[frame_index].astype(np.uint8)
        # mask_for_frame = cv2.GaussianBlur(mask_for_frame, (7, 7), 0)
        mask_for_frame = cv2.medianBlur(mask_for_frame, 7)
        mask_for_frame = cv2.morphologyEx(mask_for_frame, cv2.MORPH_CLOSE, kernel)
        mask_for_frame = cv2.morphologyEx(mask_for_frame, cv2.MORPH_OPEN, kernel)
        mask_for_frame = cv2.medianBlur(mask_for_frame, 7)

        gradient = cv2.morphologyEx(mask_for_frame, cv2.MORPH_GRADIENT, kernel)
        # gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=cv2.contourArea, reverse=True)
        mask = np.zeros(mask_for_frame.shape)
        mask = cv2.fillPoly(mask, pts=[contours[0]], color=1)
        # mask = cv2.drawContours(mask, [contours[0]], contourIdx=-1, color=255, thickness=cv2.FILLED)
        blue_mask = (blue_frame < 140).astype(np.uint8)
        mask = (mask * blue_mask).astype(np.uint8)

        # masks[frame_index] = scale_matrix_0_to_255(mask)
        frame_after_mask = apply_mask_on_color_frame(frame, mask)
        masks[frame_index] = mask * 255
        after_mask[frame_index] = frame_after_mask

    write_video(f'../Outputs/binary_{config["ID_1"]}_{config["ID_2"]}.avi', masks,
                video_params['fps'], (video_params['w'], video_params['h']), is_color=False)
    write_video(f'../Outputs/extracted_{config["ID_1"]}_{config["ID_2"]}.avi', after_mask,
                video_params['fps'], (video_params['w'], video_params['h']), is_color=True)


CONFIG = {
    'ID_1': 302828991,
    'ID_2': 316524800,
    'MAX_CORNERS': 500,
    'QUALITY_LEVEL': 0.01,
    'MIN_DISTANCE': 30,
    'BLOCK_SIZE': 3,
    'SMOOTHING_RADIUS': 5,

}
input_video = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')
# subtruct_background_by_median(input_video)
# subtruct_background_by_MOG2(input_video)
background_subtraction_ver15(input_video, CONFIG)
# Release video object
input_video.release()

# Destroy all windows
cv2.destroyAllWindows()
