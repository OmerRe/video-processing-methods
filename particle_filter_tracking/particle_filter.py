import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# change IDs to your IDs.
ID1 = "302828991"
ID2 = "316524800"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
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
    std_y_loc = 1
    std_x_velocity = 0.8
    std_y_velocity = 0.6

    # Physical model + noise
    state_drifted[0, :] = s_prior[0, :] + s_prior[4, :] + np.round(np.random.normal(mean, std_x_loc, size=(1, 100)))
    state_drifted[1, :] = s_prior[1, :] + s_prior[5, :] + np.round(np.random.normal(mean, std_y_loc, size=(1, 100)))
    state_drifted[4, :] = s_prior[4, :] + np.round(np.random.normal(mean, std_x_velocity, size=(1, 100)))
    state_drifted[5, :] = s_prior[5, :] + np.round(np.random.normal(mean, std_y_velocity, size=(1, 100)))

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
    hist = np.zeros((16, 16, 16))
    x, y, half_width, half_height, x_velocity, y_velocity = state
    image_sub_portion = image[y-half_height:y+half_height, x-half_width:x+half_width, :]
    image_sub_portion_quantized = np.floor(image_sub_portion*(15/255))
    image_sub_portion_quantized = image_sub_portion_quantized.astype(int)

    for i in range(image_sub_portion_quantized.shape[0]):
        for j in range(image_sub_portion_quantized.shape[1]):
            R_val = image_sub_portion_quantized[i, j, 0]
            G_val = image_sub_portion_quantized[i, j, 1]
            B_val = image_sub_portion_quantized[i, j, 2]
            hist[B_val, G_val, R_val] += 1

    hist = np.reshape(hist, 16 * 16 * 16)

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


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:, :, ::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box
    S_avg = np.floor(np.average(state, 1, weights=W))
    x_avg = S_avg[0] - S_avg[2]
    y_avg = S_avg[1] - S_avg[3]
    w_avg = s_initial[2] * 2  # the width does not change from its initial value
    h_avg = s_initial[3] * 2  # the height does not change from its initial value
    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    max_idx = np.argmax(W)
    x_max = state[0, max_idx] - state[2, max_idx]
    y_max = state[1, max_idx] - state[3, max_idx]
    w_max = s_initial[2] * 2  # the width does not change from its initial value
    h_max = s_initial[3] * 2  # the height does not change from its initial value
    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show(block=False)
    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state

def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

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

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

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
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()