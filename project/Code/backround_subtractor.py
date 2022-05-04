import cv2
import numpy as np
import matplotlib.pyplot as plt

from Code.utils import extract_video_parameters


def subtruct_background(input_frames):
    params = extract_video_parameters(input_frames)
    number_of_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    selected_frames = np.arange(0, number_of_frames, int(number_of_frames / 5))

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
        th, dframe = cv2.threshold(dframe, 50, 255, cv2.THRESH_BINARY)
        dframe_hsv = cv2.cvtColor(dframe, cv2.COLOR_BGR2YUV)
        mask = cv2.inRange(dframe_hsv, (120, 0, 0), (240, 190, 255))
        bak = dframe.copy()

        # replace with black
        bak[mask > 0] = (0, 0, 0)
        # Display image
        cv2.imshow('frame', bak)
        cv2.waitKey(1)

input_video = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')
subtruct_background(input_video)
# Release video object
input_video.release()

# Destroy all windows
cv2.destroyAllWindows()
