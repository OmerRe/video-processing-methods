import os
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image

from Code.backround_subtractor import subtruct_background
from Code.video_matting import video_matting
from Code.video_stabilizer import stabilize_video
from Code.video_tracker import track_object

CONFIG = {
    'ID_1': 302828991,
    'ID_2': 316524800,
    'MAX_CORNERS': 500,
    'QUALITY_LEVEL': 0.01,
    'MIN_DISTANCE': 30,
    'BLOCK_SIZE': 3,
    'SMOOTHING_RADIUS': 5,

}

RUNNING_TIME = {}

def main(running_time, config):
    # import video
    input_video = cv2.VideoCapture('../Inputs/INPUT.mp4')
    # video stabilization
    start_time = time.time()
    stabilized_frames = stabilize_video(input_video, config)
    RUNNING_TIME['time_to_stabilize'] = time.time() - start_time
    stabilized_video = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')

    # video background subtraction
    start_time = time.time()
    binary_frames, extracted_frames = subtruct_background(stabilized_video, config)
    RUNNING_TIME['time_to_binary'] = time.time() - start_time

    # # video matting
    # #TODO: Refactor
    # start_time = time.time()
    # input_video = '../Outputs/stabilized_302828991_316524800.avi'
    # binary_video = '../Outputs/binary_302828991_316524800.avi'
    # background_image = '../Inputs/background.jpg'
    # matted_video_frames = video_matting(input_video, binary_video, background_image)
    # RUNNING_TIME['matting + alpha'] = time.time() - start_time

    # video tacking
    # TODO: Refactor
    start_time = time.time()
    matted_video_frames = track_object(extracted_frames)

    return


main(running_time=RUNNING_TIME, config=CONFIG)