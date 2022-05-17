import cv2
import time
import json
from Code.backround_subtractor import subtruct_background
from Code.utils import RUNNING_TIME, CONFIG, extract_video_parameters, load_video, release_video, write_video
from Code.video_matting import video_matting
from Code.video_stabilizer import stabilize_video
from Code.video_tracker import track_object


def main(running_time=RUNNING_TIME, config=CONFIG):
    # import video
    input_video = cv2.VideoCapture('../Inputs/INPUT.mp4')
    video_params = extract_video_parameters(input_video)
    video_frames = load_video(input_video)
    config['video_params'] = video_params

    # video stabilization
    start_time = time.time()
    stabilized_frames = stabilize_video(video_frames, config)
    running_time['time_to_stabilize'] = time.time() - start_time
    release_video(input_video)
    write_video('stabilize', stabilized_frames, is_color=True)

    # video background subtraction
    start_time = time.time()
    binary_frames, extracted_frames = subtruct_background(stabilized_frames, config)
    running_time['time_to_binary'] = time.time() - start_time
    write_video('binary', binary_frames, is_color=False)
    write_video('extracted', extracted_frames, is_color=True)

    # # video matting
    start_time = time.time()
    background_image = cv2.imread(config['BACKGROUND_IMAGE_PATH'])
    matted_video_frames, alpha_frames = video_matting(stabilized_frames, binary_frames, background_image, config)
    running_time['matting + alpha'] = time.time() - start_time  #Todo insert real time
    write_video('matted', matted_video_frames, is_color=True)
    write_video('alpha', alpha_frames, is_color=False)

    # video tacking
    start_time = time.time()
    output_video_frames, list_bb = track_object(extracted_frames, matted_video_frames)
    running_time['time_to_output'] = time.time() - start_time

    with open('../Outputs/tracking.json', 'w') as f:
        json.dump(list_bb, f, indent=4)
    with open('../Outputs/timing.json', 'w') as f:
        json.dump(running_time, f, indent=4)
    write_video('OUTPUT', output_video_frames, is_color=True)

    return


main(running_time=RUNNING_TIME)