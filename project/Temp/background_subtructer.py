import cv2
import numpy as np


def background_subtraction():
    # Open Video
    cap = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    #
    # # Iterate over every file in the directory
    #
    # while (True):
    #     ret, frame = cap.read()
    #
    #     if frame is None:
    #         break
    #
    #     # Apply the background subraction
    #     history = 15  # previous frames to compare with
    #     foreground_mask = fgbg.apply(frame, learningRate=1.0/history)
    #
    #     # Remove some noise
    #     foreground_mask = cv2.medianBlur(foreground_mask, 9)
    #     # foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    #     # foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_ERODE, kernel, iterations=5)
    #     # foreground_mask = cv2.dilate(foreground_mask, kernel,iterations = 1)
    #     # foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_GRADIENT, kernel, iterations=1)
    #
    #     # foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    #
    #     contours, hierarchy = cv2.findContours(foreground_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    #     # Draw contours onto image
    #     # cv2.drawContours(frame,contours,-1,255)
    #
    #     # Draw elllipses onto image
    #     # for cnt in contours:
    #     #     try:
    #     #         ellipse = cv2.fitEllipse(cnt)  # (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
    #     #
    #     #         # Simple size constraints on the ellipses to eliminate some noise
    #     #         # Somewhat of a quick and dirty hack, but it works...
    #     #         if ellipse[1][0] * ellipse[1][1] < 1000 or ellipse[1][0] > 300 or ellipse[1][1] > 300:
    #     #             continue
    #     #
    #     #         cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
    #     #     except:
    #     #         pass
    #
    #     result = cv2.bitwise_and(frame, frame, mask=foreground_mask)  # Apply mask to RGB image
    #
    #     # Stack two images side by side in one image for display
    #     cv2.imshow('frame', np.hstack([frame, result]))
    #
    #     k = cv2.waitKey(1) & 0xff
    #     if k == ord('n'):
    #         # Go to next video
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()

    # frames_bgr = load_video(cap, color_space='bgr')
    # frames_hsv = load_video(cap, color_space='hsv')
    # n_frames = len(frames_bgr)
    # parameters = extract_video_parameters(cap)
    # backSub = cv2.createBackgroundSubtractorKNN()
    # mask_list = np.zeros((n_frames, parameters["h"], w)).astype(np.uint8)
    # print(f"[BS] - BackgroundSubtractorKNN Studying Frames history")
    # for j in range(8):
    #     print(f"[BS] - BackgroundSubtractorKNN {j + 1} / 8 pass")
    #     for index_frame, frame in enumerate(frames_hsv):
    #         frame_sv = frame[:, :, 1:]
    #         fgMask = backSub.apply(frame_sv)
    #         fgMask = (fgMask > 200).astype(np.uint8)
    #         mask_list[index_frame] = fgMask
    #
    #
    # FOI = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
    #
    # frames = []
    # for frameOI in FOI:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
    #     ret, frame = cap.read()
    #     frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     frame_HSV = cv2.GaussianBlur(frame_HSV, (3, 3), 0)
    #     frames.append(frame_HSV)
    # medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    #
    # # Display median frame
    # cv2.imshow('frame', medianFrame)
    # cv2.waitKey(0)
    #
    # # Reset frame number to 0
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #
    # # Convert background to grayscale
    # grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_HSV2BGR)
    # grayMedianFrame = cv2.cvtColor(grayMedianFrame, cv2.COLOR_BGR2GRAY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # # Loop over all frames
    # ret = True
    # while(ret):
    #       # Read frame
    #       ret, frame = cap.read()
    #       if not ret:
    #           break
    #       # Convert current frame to grayscale
    #       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #       # frame = cv2.medianBlur(frame, 3)
    #       # Calculate absolute difference of current frame and
    #       # the median frame
    #       dframe = cv2.absdiff(frame, grayMedianFrame)
    #       # Treshold to binarize
    #       th, dframe = cv2.threshold(dframe, 50, 255, cv2.THRESH_BINARY)
    #
    #       # dframe = cv2.morphologyEx(dframe, cv2.MORPH_CLOSE, kernel, iterations=5)
    #       # Display image
    #       cv2.imshow('frame', dframe)
    #       cv2.waitKey(20)
    #
    # # Release video object
    # cap.release()
    #
    # # Destroy all windows
    # cv2.destroyAllWindows()

def method3():
    cap = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')
    ret, mean = cap.read()
    mean = cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY)
    (col, row) = mean.shape[:2]

    var = np.ones((col, row), np.uint8)
    var[:col, :row] = 20
    count = 0
    while True:

        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        alpha = 0.25

        new_mean = (1 - alpha) * mean + alpha * frame_gray
        new_mean = new_mean.astype(np.uint8)
        new_var = (alpha) * (cv2.subtract(frame_gray, mean) ** 2) + (1 - alpha) * (var)

        value = cv2.absdiff(frame_gray, mean)
        value = value / np.sqrt(var)

        mean = np.where(value < 2.5, new_mean, mean)
        var = np.where(value < 2.5, new_var, var)
        a = np.uint8([255])
        b = np.uint8([0])
        background = np.where(value < 2.5, frame_gray, 0)
        forground = np.where(value >= 2.5, frame_gray, b)
        cv2.imshow('background', background)
        kernel = np.ones((3, 3), np.uint8)

        # erode = cv2.erode(forground, kernel, iterations=2)
        # erode = cv2.absdiff(forground,background)

        cv2.imshow('forground', forground)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def method4():

    # creating video element
    cap = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')

    # images from which Background to be estimated
    images = []

    # taking 13 frames to estimate the background
    for i in range(25):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        images.append(frame)

    # getting shape of the frame to create background
    row, col = frame.shape
    background = np.zeros([row, col], np.uint8)
    background = np.median(images, axis=0)

    # by median openration data type of background changes so again change it to uint8
    background = background.astype(np.uint8)
    res = np.zeros([row, col], np.uint8)

    # converting interger 0 and 255 to type uint8
    a = np.uint8([255])
    b = np.uint8([0])

    # initialising i so that we can replace frames from images to get new frames
    i = 0

    # creating different kernels for erode and dilate openration. bigger for erode and smaller for dilate
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
    while cap.isOpened():
        _, frame = cap.read()
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        images[i % 25] = frame1
        background = np.median(images, axis=0)
        background = background.astype(np.uint8)

        # taking absolute difference otherwise having trouble in setting a particular value of threshold used in np.where
        res = cv2.absdiff(frame1, background)
        foreground = res
        foreground[res >= 50] = 255
        foreground[res < 50] = 0
        # foreground = cv2.morphologyEx(foreground, cv2.MORPH_ERODE, kernel2)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_DILATE, kernel1)

        # to get the colored part of the generated mask res
        res = cv2.bitwise_and(frame, frame, mask=foreground)
        cv2.imshow('median', foreground)
        cv2.imshow('background', background)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        i = i + 1
    cap.release()
    cv2.destroyAllWindows()

def method5():
    cap = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')

    _, frame = cap.read()

    # getting shape of the frame
    row, col, channel = frame.shape

    # initialising background and foreground
    background = np.zeros([row, col], np.uint8)
    foreground = np.zeros([row, col], np.uint8)

    # converting data type of intergers 0 and 255 to uint8 type
    a = np.uint8([255])
    b = np.uint8([0])

    # creating kernel for removing noise
    kernel = np.ones([3, 3], np.uint8)

    while cap.isOpened():
        _, frame1 = cap.read()
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # applying algorithm of median approximation method to get estimated background
        background = np.where(frame > background, background + 1, background - 1)

        # using cv2.absdiff instead of background - frame, because 1 - 2 will give 255 which is not expected
        foreground = cv2.absdiff(background, frame)

        # setting a threshold value for removing noise and getting foreground
        foreground = np.where(foreground > 40, a, b)

        # removing noise
        foreground = cv2.erode(foreground, kernel)
        foreground = cv2.dilate(foreground, kernel)
        # using bitwise and to get colored foreground
        foreground = cv2.bitwise_and(frame1, frame1, mask=foreground)
        cv2.imshow('background', background)
        cv2.imshow('foreground', foreground)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def norm_pdf(x, mean, sigma):
    return (1 / (np.sqrt(2 * 3.14) * sigma)) * (np.exp(-0.5 * (((x - mean) / sigma) ** 2)))

def method6():

    # 3'rd gaussian is most probable and 1'st gaussian is least probable

    cap = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # getting shape of the frame
    row, col = frame.shape

    # initialising mean,variance,omega and omega by sigma
    mean = np.zeros([3, row, col], np.float64)
    mean[1, :, :] = frame

    variance = np.zeros([3, row, col], np.float64)
    variance[:, :, :] = 400

    omega = np.zeros([3, row, col], np.float64)
    omega[0, :, :], omega[1, :, :], omega[2, :, :] = 0, 0, 1

    omega_by_sigma = np.zeros([3, row, col], np.float64)

    # initialising foreground and background
    foreground = np.zeros([row, col], np.uint8)
    background = np.zeros([row, col], np.uint8)

    # initialising T and alpha
    alpha = 0.3
    T = 0.5

    # converting data type of integers 0 and 255 to uint8 type
    a = np.uint8([255])
    b = np.uint8([0])

    while cap.isOpened():
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # converting data type of frame_gray so that different operation with it can be performed
        frame_gray = frame_gray.astype(np.float64)

        # Because variance becomes negative after some time because of norm_pdf function so we are converting those indices
        # values which are near zero to some higher values according to their preferences
        variance[0][np.where(variance[0] < 1)] = 10
        variance[1][np.where(variance[1] < 1)] = 5
        variance[2][np.where(variance[2] < 1)] = 1

        # calulating standard deviation
        sigma1 = np.sqrt(variance[0])
        sigma2 = np.sqrt(variance[1])
        sigma3 = np.sqrt(variance[2])

        # getting values for the inequality test to get indexes of fitting indexes
        compare_val_1 = cv2.absdiff(frame_gray, mean[0])
        compare_val_2 = cv2.absdiff(frame_gray, mean[1])
        compare_val_3 = cv2.absdiff(frame_gray, mean[2])

        value1 = 2.5 * sigma1
        value2 = 2.5 * sigma2
        value3 = 2.5 * sigma3

        # finding those indexes where values of T are less than most probable gaussian and those where sum of most probale
        # and medium probable is greater than T and most probable is less than T
        fore_index1 = np.where(omega[2] > T)
        fore_index2 = np.where(((omega[2] + omega[1]) > T) & (omega[2] < T))

        # Finding those indices where a particular pixel values fits at least one of the gaussian
        gauss_fit_index1 = np.where(compare_val_1 <= value1)
        gauss_not_fit_index1 = np.where(compare_val_1 > value1)

        gauss_fit_index2 = np.where(compare_val_2 <= value2)
        gauss_not_fit_index2 = np.where(compare_val_2 > value2)

        gauss_fit_index3 = np.where(compare_val_3 <= value3)
        gauss_not_fit_index3 = np.where(compare_val_3 > value3)

        # finding common indices for those indices which satisfy line 70 and 80
        temp = np.zeros([row, col])
        temp[fore_index1] = 1
        temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
        index3 = np.where(temp == 2)

        # finding com
        temp = np.zeros([row, col])
        temp[fore_index2] = 1
        index = np.where((compare_val_3 <= value3) | (compare_val_2 <= value2))
        temp[index] = temp[index] + 1
        index2 = np.where(temp == 2)

        match_index = np.zeros([row, col])
        match_index[gauss_fit_index1] = 1
        match_index[gauss_fit_index2] = 1
        match_index[gauss_fit_index3] = 1
        not_match_index = np.where(match_index == 0)

        # updating variance and mean value of the matched indices of all three gaussians
        rho = alpha * norm_pdf(frame_gray[gauss_fit_index1], mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])
        constant = rho * ((frame_gray[gauss_fit_index1] - mean[0][gauss_fit_index1]) ** 2)
        mean[0][gauss_fit_index1] = (1 - rho) * mean[0][gauss_fit_index1] + rho * frame_gray[gauss_fit_index1]
        variance[0][gauss_fit_index1] = (1 - rho) * variance[0][gauss_fit_index1] + constant
        omega[0][gauss_fit_index1] = (1 - alpha) * omega[0][gauss_fit_index1] + alpha
        omega[0][gauss_not_fit_index1] = (1 - alpha) * omega[0][gauss_not_fit_index1]

        rho = alpha * norm_pdf(frame_gray[gauss_fit_index2], mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])
        constant = rho * ((frame_gray[gauss_fit_index2] - mean[1][gauss_fit_index2]) ** 2)
        mean[1][gauss_fit_index2] = (1 - rho) * mean[1][gauss_fit_index2] + rho * frame_gray[gauss_fit_index2]
        variance[1][gauss_fit_index2] = (1 - rho) * variance[1][gauss_fit_index2] + rho * constant
        omega[1][gauss_fit_index2] = (1 - alpha) * omega[1][gauss_fit_index2] + alpha
        omega[1][gauss_not_fit_index2] = (1 - alpha) * omega[1][gauss_not_fit_index2]

        rho = alpha * norm_pdf(frame_gray[gauss_fit_index3], mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])
        constant = rho * ((frame_gray[gauss_fit_index3] - mean[2][gauss_fit_index3]) ** 2)
        mean[2][gauss_fit_index3] = (1 - rho) * mean[2][gauss_fit_index3] + rho * frame_gray[gauss_fit_index3]
        variance[2][gauss_fit_index3] = (1 - rho) * variance[2][gauss_fit_index3] + constant
        omega[2][gauss_fit_index3] = (1 - alpha) * omega[2][gauss_fit_index3] + alpha
        omega[2][gauss_not_fit_index3] = (1 - alpha) * omega[2][gauss_not_fit_index3]

        # updating least probable gaussian for those pixel values which do not match any of the gaussian
        mean[0][not_match_index] = frame_gray[not_match_index]
        variance[0][not_match_index] = 200
        omega[0][not_match_index] = 0.1

        # normalise omega
        sum = np.sum(omega, axis=0)
        omega = omega / sum

        # finding omega by sigma for ordering of the gaussian
        omega_by_sigma[0] = omega[0] / sigma1
        omega_by_sigma[1] = omega[1] / sigma2
        omega_by_sigma[2] = omega[2] / sigma3

        # getting index order for sorting omega by sigma
        index = np.argsort(omega_by_sigma, axis=0)

        # from that index(line 139) sorting mean,variance and omega
        mean = np.take_along_axis(mean, index, axis=0)
        variance = np.take_along_axis(variance, index, axis=0)
        omega = np.take_along_axis(omega, index, axis=0)

        # converting data type of frame_gray so that we can use it to perform operations for displaying the image
        frame_gray = frame_gray.astype(np.uint8)

        # getting background from the index2 and index3
        background[index2] = frame_gray[index2]
        background[index3] = frame_gray[index3]
        cv2.imshow('frame', cv2.subtract(frame_gray, background))
        cv2.imshow('frame_gray', frame_gray)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # function for trackbar
def nothing(x):
    pass

def method7():

    # naming the trackbar
    cv2.namedWindow('diff')

    # creating trackbar for values to be subtracted
    cv2.createTrackbar('min_val', 'diff', 0, 255, nothing)
    cv2.createTrackbar('max_val', 'diff', 0, 255, nothing)

    # creating video element
    cap = cv2.VideoCapture('../Outputs/stabilized_302828991_316524800.avi')
    _, frame = cap.read()
    # converting the image into grayscale image
    image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # getting the shape of the frame capture which will be used for creating an array of resultant image which will store the diff
    row, col = image1.shape
    res = np.zeros([row, col, 1], np.uint8)

    # converting data type integers 255 and 0 to uint8 type
    a = np.uint8([255])
    b = np.uint8([0])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    while True:
        ret, image2 = cap.read()
        if not ret:
            break
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # getting threshold values from trackbar according to the lightning condition
        min_val = cv2.getTrackbarPos('min_val', 'diff')
        max_val = cv2.getTrackbarPos('max_val', 'diff')

        # using cv2.absdiff instead of image1 - image2 because 1 - 2 will give 255 which is not expected
        res = cv2.absdiff(image1, image2)
        cv2.imshow('image', res)

        # creating mask
        res = np.where((min_val < res) & (max_val > res), a, b)
        res = cv2.bitwise_and(image2, image2, mask=res)
        # res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=5)
        cv2.imshow('diff', res)

        # assigning new new to the previous frame
        image1 = image2

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()