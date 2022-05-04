import cv2
import numpy as np
from scipy.stats import gaussian_kde

def extract_video_parameters(input_video: cv2.VideoCapture) -> dict:
    fourcc = int(input_video.get(cv2.CAP_PROP_FOURCC))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "h": h, "w": w,
            "n_frames": n_frames}

def load_video(video: cv2.VideoCapture, color_space: str = 'bgr'):
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        success, curr = video.read()
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
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames)

def convert_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def read_first_frame(input_video: cv2.VideoCapture):
    _, prev = input_video.read()
    return convert_to_gray(prev)

def release_video(video: cv2.VideoCapture) -> None:
    video.release()
    cv2.destroyAllWindows()

def write_video(output_path: str, frames: list, fps: int, out_size: tuple, is_color: bool) -> None:
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_out = cv2.VideoWriter(output_path, fourcc, fps, out_size, isColor=is_color)
    for frame in frames:
        video_out.write(frame)
    video_out.release()
    cv2.destroyAllWindows()

def fixBorder(frame):
    (h, w, channels) = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame
