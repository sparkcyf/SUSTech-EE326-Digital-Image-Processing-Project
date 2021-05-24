import numpy as np
import cv2
import ld_module_transform


def get_bin_img(img, kernel_size=3, sobel_dirn='X', sobel_thresh=(0, 255), r_thresh=(0, 255),
                s_thresh=(0, 255), b_thresh=(0, 255), g_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if sobel_dirn == 'X':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    combined = np.zeros_like(sbinary)
    combined[(sbinary == 1)] = 1

    # Threshold R color channel
    r_binary = get_rgb_thresh_img(img, thresh=r_thresh)

    # Threshhold G color channel
    g_binary = get_rgb_thresh_img(img, thresh=g_thresh, channel='G')

    # Threshhold B in LAB
    b_binary = get_lab_bthresh_img(img, thresh=b_thresh)

    # Threshold color channel
    s_binary = get_hls_sthresh_img(img, thresh=s_thresh)

    # If two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(combined)
    combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 1

    return combined_binary


def get_hls_sthresh_img(img, thresh=(0, 255)):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls_img[:, :, 2]

    binary_output = np.zeros_like(S).astype(np.uint8)
    binary_output[(S >= thresh[0]) & (S < thresh[1])] = 1

    return binary_output


def get_lab_bthresh_img(img, thresh=(0, 255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    B = lab_img[:, :, 2]

    bin_op = np.zeros_like(B).astype(np.uint8)
    bin_op[(B >= thresh[0]) & (B < thresh[1])] = 1

    return bin_op


def get_rgb_thresh_img(img, channel='R', thresh=(0, 255)):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channel == 'R':
        bin_img = img1[:, :, 0]
    if channel == 'G':
        bin_img = img1[:, :, 1]
    if channel == 'B':
        bin_img = img1[:, :, 2]

    binary_img = np.zeros_like(bin_img).astype(np.uint8)
    binary_img[(bin_img >= thresh[0]) & (bin_img < thresh[1])] = 1

    return binary_img


def abs_thresh(img, sobel_kernel=9, mag_thresh=(0, 255), return_grad=False, direction='y'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = None
    scaled_sobel = None

    # Sobel x
    if direction.lower() == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
    # Sobel y
    else:
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in y

    if return_grad == True:
        return grad

    abs_sobel = np.absolute(grad)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1

    return grad_binary
