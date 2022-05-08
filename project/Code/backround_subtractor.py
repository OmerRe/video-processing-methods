import cv2
import numpy as np
import matplotlib.pyplot as plt

from Code.utils import extract_video_parameters


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


input_video = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')
subtruct_background_by_median(input_video)
# subtruct_background_by_MOG2(input_video)
# Release video object
input_video.release()

# Destroy all windows
cv2.destroyAllWindows()
