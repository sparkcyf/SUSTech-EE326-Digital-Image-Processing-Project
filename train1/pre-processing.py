import numpy as np
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

input_img = cv2.imread('1492627187263118217/1.jpg')


def transform(input_img, num):
    rows, cols, ch = input_img.shape
    print(input_img.shape)

    # pts1 = np.float32([[629,250],[0,445],[705,250],[1279,430]])
    pts1 = np.float32([[450, 300], [-851, 720], [870, 300], [2057, 720]])
    pts2 = np.float32([[0, 0], [0, 600], [600, 0], [600, 600]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(input_img, M, (600, 600))
    dst_extract = abs_thresh(dst, sobel_kernel=3, mag_thresh=(50, 210), direction='x')
    # dst_extract = np.where(dst_extract>0.5,1,0)
    # histogram = np.sum(dst_extract[dst_extract.shape[0] // 2:, :], axis=0)

    ##add line mark
    dst_mark = add_line_mark(dst_extract)

    plt.subplot(131), plt.imshow(input_img), plt.title('Input')
    plt.subplot(132), plt.imshow(dst), plt.title('Output')
    plt.subplot(133), plt.imshow(dst_extract), plt.title('op_extracted')
    # plt.show()
    plt.savefig("op/op_reg" + str(num) + ".png", dpi=300)


def abs_thresh(img, sobel_kernel=9, mag_thresh=(0, 255), return_grad=False, direction='x'):
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


def add_line_mark(input_img):
    # determind start point
    input_img[450:599, 0:150] = 0
    input_img[450:599, 400:599] = 0
    # fill blank
    input_img_hist_shape = input_img[400:599, 0:599]
    # plt.plot(histogram)
    # plt.show()

    # use window to find points (windows size: H30 W50)

    ##LEFT
    histogram = np.sum(input_img_hist_shape[input_img_hist_shape.shape[0] // 2:, :], axis=0)
    peak_L = find_peaks(histogram[0:299], distance=150)[0]

    def gen_windows(bottom_H, midpoint_W, H, W):
        window = input_img[bottom_H - 30:bottom_H, midpoint_W - 25:midpoint_W + 25]
        H = 30
        W = 50
        # window_arr = input_img[]
        histogram_window = np.sum(window[window.shape[0] // 2:, :], axis=0)
        window_peak = find_peaks(histogram_window, distance=50)[0]
        return_mid = 0
        if window_peak.size > 0:
            if window_peak > 5:
                return_mid = midpoint_W - (W / 2) + window_peak
        else:
            return_mid = midpoint_W
        # plt.plot(histogram_window)
        # plt.show()
        return bottom_H - H / 2, return_mid

    lane_list_L = []
    # gen_windows(299,peak[0],30,50)
    plt.imshow(input_img)
    # iteraton(L)
    # print(peak_L)
    midpoint_tmp_L = peak_L[0]
    for bottom in range(599, 29, -30):
        window_H, window_WM = gen_windows(bottom, midpoint_tmp_L, 30, 50)
        midpoint_tmp_L = int(window_WM)
        plt.scatter(midpoint_tmp_L, bottom)

    ##RIGHT
    histogram = np.sum(input_img_hist_shape[input_img_hist_shape.shape[0] // 2:, :], axis=0)
    peak_R = find_peaks(histogram[300:599], distance=150)[0][0]+299
    if peak_R.size > 0 and histogram[peak_R] > 50:
        pass
    else:
        peak_R = 400
    # iteraton(L)
    print(peak_R)
    midpoint_tmp_R = peak_R
    for bottom in range(599, 29, -30):
        window_H, window_WM = gen_windows(bottom, midpoint_tmp_R, 30, 50)
        midpoint_tmp_R = int(window_WM)
        plt.scatter(midpoint_tmp_R, bottom)

    plt.show()

    return 0


for i in range(19,20):
    input_img = cv2.imread('1492626805094402903/' + str(i) + '.jpg')
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    transform(input_img, i)
