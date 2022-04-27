import os
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
from Code.video_stabilizer import stabilize_video


CONFIG = {
    'ID_1': 302828991,
    'ID_2': 316524800,
    'SMOOTHING_RADIUS': 50
}

RUNNING_TIME = {}

def main(running_time, config):
    # import video
    input_video = cv2.VideoCapture('../Inputs/INPUT.mp4')

    # video stabilization
    start_time = time.time()
    stabilize_video(input_video, config)
    RUNNING_TIME['time_to_stabilize'] = time.time() - start_time

    # video background subtraction
    # start_time = time.time()
    # RUNNING_TIME['time_to_binary'] = time.time() - start_time

    return


main(running_time=RUNNING_TIME, config=CONFIG)