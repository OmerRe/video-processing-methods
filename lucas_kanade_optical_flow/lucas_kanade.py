import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata

# FILL IN YOUR ID
ID1 = 302828991
ID2 = 316524800


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    for level in range(num_levels):
        filtered_image = signal.convolve2d(pyramid[-1], PYRAMID_FILTER, mode='same', boundary='symm')
        decimated_image = filtered_image[::2, ::2]
        pyramid.append(decimated_image)
    return pyramid

def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    boundary_idx = window_size // 2
    h, w = I1.shape

    for i in range(boundary_idx, h - boundary_idx):
        for j in range(boundary_idx, w - boundary_idx):
            Ix_window = (Ix[i - boundary_idx:i + boundary_idx + 1, j - boundary_idx:j + boundary_idx + 1]).reshape(
                                   window_size*window_size, 1)
            Iy_window = (Iy[i - boundary_idx:i + boundary_idx + 1, j - boundary_idx:j + boundary_idx + 1]).reshape(
                window_size*window_size, 1)
            It_window = (It[i - boundary_idx:i + boundary_idx + 1, j - boundary_idx:j + boundary_idx + 1]).reshape(
                window_size*window_size, 1)
            A = np.append(Ix_window, Iy_window, axis=1)
            b = It_window

            try:
                du[i, j], dv[i, j] = -np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)),
                                                         np.transpose(A)), b)
            except np.linalg.LinAlgError:
                pass
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    h, w = image.shape

    if u.shape != image.shape:
        normalize_factor_u = w / u.shape[1]
        normalize_factor_v = h / v.shape[0]
        u = cv2.resize(u, (w, h)) * normalize_factor_u
        v = cv2.resize(v, (w, h)) * normalize_factor_v

    x, y = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1), indexing='xy')

    warped_image = griddata((x.flatten(), y.flatten()), image.flatten(),
                            (x.flatten() + u.flatten(), y.flatten() + v.flatten()), fill_value=np.nan)
    indices = np.isnan(warped_image)
    if len(indices) > 0:
        warped_image[indices] = image.flatten()[indices]
    return warped_image.reshape(image.shape)

def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.T.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.T.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyramid_I2[-1].shape)
    v = np.zeros(pyramid_I2[-1].shape)
    for pyramid_level in (range(num_levels, -1, -1)):
        I2_warp = warp_image(pyramid_I2[pyramid_level], u, v)
        for step in range(max_iter):
            du, dv = lucas_kanade_step(pyramid_I1[pyramid_level], I2_warp, window_size)
            u = u + du
            v = v + dv
            I2_warp = warp_image(pyramid_I2[pyramid_level], u, v)
        if pyramid_level > 0:
            h_resize, w_resize = pyramid_I2[pyramid_level - 1].shape
            u = cv2.resize(u, (w_resize, h_resize)) * 2
            v = cv2.resize(v, (w_resize, h_resize)) * 2

    return u, v

def generator():
    while True:
        yield

def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    input_video = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(input_video)
    # creating VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, parameters["fps"],
                                   (parameters["width"], parameters["height"]), isColor=False)
    video_shape = (parameters["width"], parameters["height"])

    # reading first frame
    ret, first_frame = input_video.read()
    # converting to gray-scale
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # writing the first frame to output video
    output_video.write(first_frame_gray)
    # resizing the first gray scale frame
    h_factor = int(np.ceil(first_frame.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(first_frame.shape[1] / (2 ** (num_levels - 1 + 1))))
    image_size = (w_factor * (2 ** (num_levels - 1 + 1)), h_factor * (2 ** (num_levels - 1 + 1)))
    first_frame_gray = cv2.resize(first_frame_gray, image_size)

    # initialize u and v maps
    u = np.zeros(first_frame_gray.shape)
    v = np.zeros(first_frame_gray.shape)

    # looping over the frames in the input video
    prev_frame = first_frame_gray
    boundary_idx = window_size // 2

    for idx, _ in enumerate(tqdm(generator())):  # extracting the frames
        ret, frame = input_video.read()
        if ret:
            # converting to gray-scale and resizing
            gray_current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_current_frame = cv2.resize(gray_current_frame, image_size)

            # calculating u and v maps of the current frame
            u_current_frame, v_current_frame = lucas_kanade_optical_flow(prev_frame, gray_current_frame, window_size,
                                                                         max_iter, num_levels)

            # finding the mean values of u and v maps of the current frame
            u_current_frame_mean = np.mean(u_current_frame[boundary_idx:int(u_current_frame.shape[0]) - boundary_idx,
                                   boundary_idx: int(u_current_frame.shape[1]) - boundary_idx])
            v_current_frame_mean = np.mean(v_current_frame[boundary_idx:int(v_current_frame.shape[0]) - boundary_idx,
                                   boundary_idx: int(v_current_frame.shape[1]) - boundary_idx])

            # updating u and v maps
            u[boundary_idx:int(u_current_frame.shape[0]) - boundary_idx, boundary_idx: int(u_current_frame.shape[1]) - boundary_idx] += u_current_frame_mean
            v[boundary_idx:int(v_current_frame.shape[0]) - boundary_idx, boundary_idx: int(v_current_frame.shape[1]) - boundary_idx] += v_current_frame_mean

            # u += u_current_frame_mean * np.ones(u.shape)
            # v += v_current_frame_mean * np.ones(v.shape)

            warped_frame = warp_image(gray_current_frame, u, v)
            warped_frame = cv2.resize(warped_frame, video_shape)
            output_video.write(warped_frame.astype('uint8'))

            prev_frame = gray_current_frame

            plt.imshow(warped_frame, cmap='gray')  # TODO: remove
            plt.savefig(f'./stabilization_results/frame_num_{idx}')  # TODO: remove
        else:
            break

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """
    boundary_idx = window_size // 2
    h, w = I1.shape
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    corners_image = np.zeros(I1.shape)

    if I1.size < 26000:
        du, dv = lucas_kanade_step(I1, I2, window_size)
    else:
        R = cv2.cornerHarris(I2.astype('float32'), 2, 3, 0.05)
        # R = np.pad(R, (2,), constant_values=0)

        # Threshold for an optimal value, it may vary depending on the image.
        corners_image[R > 0.00005 * R.max()] = 1
        x_corners, y_corners = np.nonzero(corners_image)
        corners_indices = list(zip(x_corners, y_corners))
        Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
        Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
        It = I2 - I1

        for x_idx, y_idx in corners_indices:
            if (boundary_idx <= x_idx < h - boundary_idx) and (boundary_idx <= y_idx < w - boundary_idx):
                Ix_window = (Ix[x_idx - boundary_idx:x_idx + boundary_idx + 1, y_idx - boundary_idx:y_idx + boundary_idx + 1]).reshape(
                    window_size * window_size, 1)
                Iy_window = (Iy[x_idx - boundary_idx:x_idx + boundary_idx + 1, y_idx - boundary_idx:y_idx + boundary_idx + 1]).reshape(
                    window_size * window_size, 1)
                It_window = (It[x_idx - boundary_idx:x_idx + boundary_idx + 1, y_idx - boundary_idx:y_idx + boundary_idx + 1]).reshape(
                    window_size * window_size, 1)
                A = np.append(Ix_window, Iy_window, axis=1)
                b = It_window

                try:
                    du[x_idx, y_idx], dv[x_idx, y_idx] = -np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)),
                                                              np.transpose(A)), b)
                except np.linalg.LinAlgError:
                        pass
            else:
                pass

    return du, dv

def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.T.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.T.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyramid_I2[-1].shape)
    v = np.zeros(pyramid_I2[-1].shape)
    for pyramid_level in reversed(range(num_levels+1)):
        I2_warp = warp_image(pyramid_I2[pyramid_level], u, v)
        for step in range(max_iter):
            du, dv = faster_lucas_kanade_step(pyramid_I1[pyramid_level], I2_warp, window_size)
            u = u + du
            v = v + dv
            I2_warp = warp_image(pyramid_I2[pyramid_level], u, v)
        if pyramid_level > 0:
            h_resize, w_resize = pyramid_I2[pyramid_level - 1].shape
            u = cv2.resize(u, (w_resize, h_resize)) * 2
            v = cv2.resize(v, (w_resize, h_resize)) * 2

    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    input_video = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(input_video)
    # creating VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, parameters["fps"],
                                   (parameters["width"], parameters["height"]), isColor=False)
    video_shape = (parameters["width"], parameters["height"])

    # reading first frame
    ret, first_frame = input_video.read()
    # converting to gray-scale
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # writing the first frame to output video
    output_video.write(first_frame_gray)
    # resizing the first gray scale frame
    h_factor = int(np.ceil(first_frame.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(first_frame.shape[1] / (2 ** (num_levels - 1 + 1))))
    image_size = (w_factor * (2 ** (num_levels - 1 + 1)), h_factor * (2 ** (num_levels - 1 + 1)))
    first_frame_gray = cv2.resize(first_frame_gray, image_size)

    # initialize u and v maps
    u = np.zeros(first_frame_gray.shape)
    v = np.zeros(first_frame_gray.shape)

    # looping over the frames in the input video
    prev_frame = first_frame_gray
    boundary_idx = window_size // 2

    for idx, _ in enumerate(tqdm(generator())):  # extracting the frames
        ret, frame = input_video.read()
        if ret:
            # converting to gray-scale and resizing
            gray_current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_current_frame = cv2.resize(gray_current_frame, image_size)

            # calculating u and v maps of the current frame
            u_current_frame, v_current_frame = faster_lucas_kanade_optical_flow(prev_frame, gray_current_frame,
                                                                                window_size, max_iter, num_levels)

            # finding the mean values of u and v maps of the current frame
            u_current_frame_mean = np.mean(u_current_frame[boundary_idx:int(u_current_frame.shape[0]) - boundary_idx,
                                   boundary_idx: int(u_current_frame.shape[1]) - boundary_idx])
            v_current_frame_mean = np.mean(v_current_frame[boundary_idx:int(v_current_frame.shape[0]) - boundary_idx,
                                   boundary_idx: int(v_current_frame.shape[1]) - boundary_idx])

            # updating u and v maps
            u[boundary_idx:int(u_current_frame.shape[0]) - boundary_idx, boundary_idx: int(u_current_frame.shape[1]) - boundary_idx] += u_current_frame_mean
            v[boundary_idx:int(v_current_frame.shape[0]) - boundary_idx, boundary_idx: int(v_current_frame.shape[1]) - boundary_idx] += v_current_frame_mean

            # u += u_current_frame_mean * np.ones(u.shape)
            # v += v_current_frame_mean * np.ones(v.shape)

            warped_frame = warp_image(gray_current_frame, u, v)
            warped_frame = cv2.resize(warped_frame, video_shape)
            output_video.write(warped_frame.astype('uint8'))

            prev_frame = gray_current_frame
        else:
            break

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    input_video = cv2.VideoCapture(input_video_path)
    parameters = get_video_parameters(input_video)
    # creating VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, parameters["fps"],
                                   (parameters["width"], parameters["height"]), isColor=False)
    video_shape = (parameters["width"], parameters["height"])

    # reading first frame
    ret, first_frame = input_video.read()
    # converting to gray-scale
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # writing the first frame to output video
    output_video.write(first_frame_gray)
    # resizing the first gray scale frame
    h_factor = int(np.ceil(first_frame.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(first_frame.shape[1] / (2 ** (num_levels - 1 + 1))))
    image_size = (w_factor * (2 ** (num_levels - 1 + 1)), h_factor * (2 ** (num_levels - 1 + 1)))
    first_frame_gray = cv2.resize(first_frame_gray, image_size)

    # initialize u and v maps
    u = np.zeros(first_frame_gray.shape)
    v = np.zeros(first_frame_gray.shape)

    # looping over the frames in the input video
    prev_frame = first_frame_gray
    boundary_idx = window_size // 2

    for idx, _ in enumerate(tqdm(generator())):  # extracting the frames
        ret, frame = input_video.read()
        if ret:
            # converting to gray-scale and resizing
            gray_current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_current_frame = cv2.resize(gray_current_frame, image_size)

            # calculating u and v maps of the current frame
            u_current_frame, v_current_frame = faster_lucas_kanade_optical_flow(prev_frame, gray_current_frame,
                                                                                window_size, max_iter, num_levels)

            # finding the mean values of u and v maps of the current frame
            u_current_frame_mean = np.mean(u_current_frame[boundary_idx:int(u_current_frame.shape[0]) - boundary_idx,
                                   boundary_idx: int(u_current_frame.shape[1]) - boundary_idx])
            v_current_frame_mean = np.mean(v_current_frame[boundary_idx:int(v_current_frame.shape[0]) - boundary_idx,
                                   boundary_idx: int(v_current_frame.shape[1]) - boundary_idx])

            # updating u and v maps
            u[boundary_idx:int(u_current_frame.shape[0]) - boundary_idx, boundary_idx: int(u_current_frame.shape[1]) - boundary_idx] += u_current_frame_mean
            v[boundary_idx:int(v_current_frame.shape[0]) - boundary_idx, boundary_idx: int(v_current_frame.shape[1]) - boundary_idx] += v_current_frame_mean

            # u += u_current_frame_mean * np.ones(u.shape)
            # v += v_current_frame_mean * np.ones(v.shape)

            warped_frame = warp_image(gray_current_frame[start_rows:gray_current_frame.shape[0] - end_rows,
                                      start_cols:gray_current_frame.shape[1] - end_cols], u, v)

            warped_frame = cv2.resize(warped_frame, video_shape)
            output_video.write(warped_frame.astype('uint8'))

            prev_frame = gray_current_frame
        else:
            break

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()
    pass


