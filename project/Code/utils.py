import cv2
import numpy as np
import GeodisTK

CONFIG = {
    'ID_1': 302828991,
    'ID_2': 316524800,
    'BACKGROUND_IMAGE_PATH': '../Inputs/background.jpg',
    'MAX_CORNERS': 500,
    'QUALITY_LEVEL': 0.01,
    'MIN_DISTANCE': 30,
    'BLOCK_SIZE': 3,
    'SMOOTHING_RADIUS': 5,
}

RUNNING_TIME = {}

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
        elif color_space == 'gray':
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

def write_video(output_path: str, frames: list, is_color: bool) -> None:
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_out = cv2.VideoWriter(f'../Outputs/{output_path}_{CONFIG["ID_1"]}_{CONFIG["ID_2"]}.avi',
                                fourcc, CONFIG['video_params']['fps'],
                                (CONFIG['video_params']['w'], CONFIG['video_params']['h']), isColor=is_color)
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

def scale_matrix_0_to_255(input_matrix):
    if input_matrix.dtype == np.bool:
        input_matrix = np.uint8(input_matrix)
    input_matrix = input_matrix.astype(np.uint8)
    scaled = 255 * (input_matrix - np.min(input_matrix)) / np.ptp(input_matrix)
    return np.uint8(scaled)

def apply_mask_on_color_frame(frame, mask):
    frame_after_mask = np.copy(frame)
    frame_after_mask[:, :, 0] = frame_after_mask[:, :, 0] * mask
    frame_after_mask[:, :, 1] = frame_after_mask[:, :, 1] * mask
    frame_after_mask[:, :, 2] = frame_after_mask[:, :, 2] * mask
    return frame_after_mask

def kernel(size: int, shape: str):
    if shape == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def geodesic_distance_2d(I, S, lamb, iter):
    '''
    get 2d geodesic disntance by raser scanning.
    I: input image, can have multiple channels. Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds. Type should be np.uint8.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic2d_raster_scan(I, S, lamb, iter)

def convert_frames_color(frames: list, color_space: int):
    converted_frames = []
    for frame in frames:
            converted_frames.append(cv2.cvtColor(frame, color_space))

    return converted_frames