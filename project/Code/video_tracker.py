import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

N = 80
resize_scale = 4
# Initial Settings
s_initial = [250/resize_scale,    # x center
             600/resize_scale,    # y center
              150/resize_scale,    # half width  #Todo calc dynamic BB
              400/resize_scale,    # half height
               0,    # velocity x
               0]    # velocity y

def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior

    # Gaussian noise parameters
    mean = 0
    std_x_loc = 2
    std_y_loc = 0.3
    std_x_velocity = 0.8
    std_y_velocity = 0.2

    # Physical model + noise
    state_drifted[0, :] = s_prior[0, :] + s_prior[4, :] + np.round(np.random.normal(mean, std_x_loc, size=(1, N)))
    state_drifted[1, :] = s_prior[1, :] + s_prior[5, :] + np.round(np.random.normal(mean, std_y_loc, size=(1, N)))
    state_drifted[2, :] = s_prior[2, :] + np.round(np.random.normal(mean, std_x_velocity, size=(1, N)))
    state_drifted[3, :] = s_prior[3, :] + np.round(np.random.normal(mean, std_x_velocity, size=(1, N)))
    state_drifted[4, :] = s_prior[4, :] + np.round(np.random.normal(mean, std_x_velocity, size=(1, N)))
    state_drifted[5, :] = s_prior[5, :] + np.round(np.random.normal(mean, std_y_velocity, size=(1, N)))

    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((8, 8, 8))
    x, y, half_width, half_height, x_velocity, y_velocity = state
    image_sub_portion = image[max(0,y-half_height):min(image.shape[0],y+half_height), max(0,x-half_width): min(image.shape[1], x+half_width), :]
    image_sub_portion_quantized = np.floor(image_sub_portion*(7/255))
    image_sub_portion_quantized = image_sub_portion_quantized.astype(int)

    for i in range(image_sub_portion_quantized.shape[0]):
        for j in range(image_sub_portion_quantized.shape[1]):
            R_val = image_sub_portion_quantized[i, j, 0]
            G_val = image_sub_portion_quantized[i, j, 1]
            B_val = image_sub_portion_quantized[i, j, 2]
            hist[B_val, G_val, R_val] += 1

    hist = np.reshape(hist, 8 * 8 * 8)

    # normalize
    hist = hist/np.sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = np.zeros(previous_state.shape)
    for n in range(previous_state.shape[1]):
        r = np.random.uniform(0, 1)
        j = np.argmax(cdf >= r)
        S_next[:, n] = previous_state[:, j]
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    distance = np.exp(20*np.sum(np.sqrt(np.multiply(p, np.conjugate(q)))))
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict, frame_index_agg_states:dict
                  ) -> (np.ndarray, dict):
    state = state * resize_scale
    fig, ax = plt.subplots(1)
    image = image[:, :, ::-1]
    plt.imshow(image)
    plt.title(" - Frame mumber = " + str(frame_index))

    # Avg particle box
    S_avg = np.floor(np.average(state, 1, weights=W))
    x_avg = S_avg[0] - S_avg[2]
    y_avg = S_avg[1] - S_avg[3]
    w_avg = S_avg[2] * 2 # the width does not change from its initial value
    h_avg = S_avg[3] * 2  # the height does not change from its initial value
    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    max_idx = np.argmax(W)
    x_max = state[0, max_idx] - state[2, max_idx]
    y_max = state[1, max_idx] - state[3, max_idx]
    w_max = state[2, max_idx] * 2  # the width does not change from its initial value
    h_max = state[3, max_idx] * 2  # the height does not change from its initial value
    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    x_agg = (x_max + x_avg)/2
    y_agg = (y_max + y_avg)/2
    w_agg = (w_max + w_avg)/2 # the width does not change from its initial value
    h_agg = (h_max + h_avg)/2 # the height does not change from its initial value
    rect = patches.Rectangle((x_agg, y_agg), w_agg, h_agg, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    plt.show(block=False)
    fig.savefig(os.path.join('../Temp/', str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    frame_index_agg_states[frame_index] = [float(x) for x in [x_agg, y_agg, w_agg, h_agg]]
    image_copy = image.copy()
    frame = cv2.rectangle(image_copy,(int(x_agg), int(y_agg)), (int(x_agg+w_agg), int(y_agg+h_agg)),(0,255,0), 3)
    return (frame, frame_index_agg_states)

def track_object(extracted_frames, matted_frames):
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = extracted_frames[0]
    original_shape = image.shape
    image = cv2.resize(image, (int(original_shape[1]/resize_scale), int(original_shape[0]/resize_scale)))

    # COMPUTE NORMALIZED HISTOGRAM - Template
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = np.zeros(N)
    for i in range(N):
        # Compute normalized histogram for each particle
        p = compute_normalized_histogram(image, S[:, i])
        # Compute weights using bhattacharyya distance
        W[i] = bhattacharyya_distance(p, q)
    W = W/np.sum(W)
    C = np.array([np.sum(W[0:i]) for i in range(1, N+1)])

    images_processed = 1
    output_frames = []
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    frame_index_agg_states = {}
    for idx, frame in enumerate(extracted_frames[1:]):
        frame = cv2.resize(frame, (int(original_shape[1] / resize_scale), int(original_shape[0] / resize_scale)))
        S_prev = S

        # LOAD NEW IMAGE FRAME
        current_image = frame

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        for i in range(N):
            # Compute normalized histogram for each particle
            p = compute_normalized_histogram(current_image, S[:, i])
            # Compute weights using bhattacharyya distance
            W[i] = bhattacharyya_distance(p, q)
        W = W / np.sum(W)
        C = np.array([np.sum(W[0:i]) for i in range(1, N + 1)])

        # CREATE DETECTOR PLOTS
        images_processed += 1
        frame_with_bb, list_bb = show_particles(matted_frames[idx], S, W, images_processed, frame_index_to_avg_state, frame_index_to_max_state, frame_index_agg_states)
        output_frames.append(frame_with_bb)

        return output_frames, list_bb


# ##### TODO: Remove before assign
# CONFIG = {
#     'ID_1': 302828991,
#     'ID_2': 316524800,
#     'MAX_CORNERS': 500,
#     'QUALITY_LEVEL': 0.01,
#     'MIN_DISTANCE': 30,
#     'BLOCK_SIZE': 3,
#     'SMOOTHING_RADIUS': 5,
#
# }
# input_video = cv2.VideoCapture('../Temp/extracted_302828991_316524800.avi')
# video_params = extract_video_parameters(input_video)
# video_frames = load_video(input_video)
#
# track_object(video_frames)
# # Release video object
# input_video.release()
#
# # Destroy all windows
# cv2.destroyAllWindows()
