import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import circle_fit
import os
import json
from skimage.filters import threshold_otsu

input_img = cv2.imread('1492627187263118217/1.jpg')


def transform(input_img, num):
    # rows, cols, ch = input_img.shape
    # print(input_img.shape)

    # pts1 = np.float32([[629,250],[0,445],[705,250],[1279,430]])
    pts1 = np.float32([[450, 300], [-851, 720], [870, 300], [2057, 720]])
    pts2 = np.float32([[0, 0], [0, 600], [600, 0], [600, 600]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(input_img, M, (600, 600))
    dst_extract = abs_thresh(dst, sobel_kernel=3, mag_thresh=(35, 210), direction='x')
    # dst_extract = abs_otsu(dst)

    # dst_extract = np.where(dst_extract>0.5,1,0)
    # histogram = np.sum(dst_extract[dst_extract.shape[0] // 2:, :], axis=0)

    # plt.imshow(dst), plt.title(foldername + filename), plt.savefig("ware_dataset/op/" + foldername + "_" + filename + "_transform.png", dpi=300)

    ##add line mark
    dst_mark = add_line_mark(dst_extract,dst)

    # plt.subplot(131), plt.imshow(input_img), plt.title('Input')
    # plt.subplot(132), plt.imshow(dst), plt.title('Output')
    # plt.subplot(133), plt.imshow(dst_extract), plt.title('op_extracted')
    # plt.show()



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

def abs_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    return binary

def add_line_mark(input_img,dst):
    # determined start point

    # delete triangle
    mult_matrix = np.tri(600, 600, 420).T
    input_img = input_img * mult_matrix * np.flip(np.tri(600, 600, 440).T, 1)
    # input_img[450:599, 0:150] = 0
    # input_img[450:599, 450:599] = 0

    # fill blank
    input_img_hist_shape = input_img[400:599, 0:599]



    def gen_windows(bottom_H, midpoint_W, H, W):

        find_indicator = 1
        window = input_img[bottom_H - 30:bottom_H, midpoint_W - 50:midpoint_W + 50]
        H = 30
        W = 100
        # window_arr = input_img[]
        histogram_window = np.sum(window[window.shape[0] // 2:, :], axis=0)
        window_peak = find_peaks(histogram_window, distance=100)[0]

        return_mid = 0

        if window_peak.size > 0:
            if histogram_window[window_peak] > 5 and np.abs(window_peak - (W / 2)) < 50:
                return_mid = midpoint_W - (W / 2) + window_peak
            else:
                return_mid = midpoint_W
        else:
            return_mid = midpoint_W

            find_indicator = 0
        # plt.plot(histogram_window)
        # plt.show()
        # print(return_mid)
        return bottom_H - H / 2, return_mid, find_indicator

    lane_list_L = []
    # gen_windows(299,peak[0],30,50)

    plt.imshow(input_img)
    #plt.imshow(dst)


    input_img_hist_all = np.sum(input_img[input_img.shape[0] // 2:, :], axis=0)
    peak_hist_all = find_peaks(input_img_hist_all, distance=150)
    print(peak_hist_all)

    # iteraton(L)
    # print(peak_L)

    def find_hist_iteration(baseline=500, start_W=0):

        input_img_hist_iter = input_img[baseline:baseline + 149, start_W:start_W + 300]
        find_hist_iteration_iter = np.sum(input_img_hist_iter[input_img_hist_iter.shape[0] // 2:, :], axis=0)

        peak_hist = find_peaks(find_hist_iteration_iter, distance=150)[0] or [0]

        if (find_hist_iteration_iter[peak_hist[0]] < 10) or peak_hist.size == 0:
            # print(find_peaks(find_hist_iteration_iter, distance=180))
            # print(baseline)
            find_hist_iteration(baseline - 50, start_W)
        else:
            pass

        return (peak_hist[0] + start_W), baseline + 50

    # find_hist_iteration()
    ##RIGHT

    # histogram = np.sum(input_img_hist_shape[input_img_hist_shape.shape[0] // 2:, :], axis=0)
    # peak_L = find_peaks(histogram[300:599], distance=150)[0]
    # midpoint_tmp_L = peak_L[0] + offset

    def find_point(start_W):
        midpoint_arr_R = a = np.zeros((0, 2)).astype(int)
        # print(midpoint_arr_R)
        midpoint_tmp_R, bottom_R = find_hist_iteration(450, start_W)
        # print(midpoint_tmp_R)
        while bottom_R > 0:
            window_H, window_WM, find_indicator = gen_windows(bottom_R, midpoint_tmp_R, 30, 100)
            midpoint_tmp_R = int(window_WM)

            if find_indicator == 1:
                tmp_list = np.array([[midpoint_tmp_R, bottom_R]]).astype(int)
                midpoint_arr_R = np.vstack([midpoint_arr_R, tmp_list])
                # plt.scatter(midpoint_tmp_R, bottom_R)
            bottom_R = bottom_R - 30
        return midpoint_arr_R

    try:
        midpoint_arr_L = find_point(0)

        print(midpoint_arr_L.T)
        plt.scatter(x=midpoint_arr_L.T[0], y=midpoint_arr_L.T[1])
        ycl, xcl, rcl, _ = circle_fit.hyper_fit(midpoint_arr_L)
        plt.gca().add_artist(plt.Circle((ycl, xcl), rcl, color='r', fill=False))
    except Exception:
        pass

    try:
        midpoint_arr_R = find_point(300)
        print(midpoint_arr_R.T)
        plt.scatter(x=midpoint_arr_R.T[0], y=midpoint_arr_R.T[1])
        ycl, xcl, rcl, _ = circle_fit.hyper_fit(midpoint_arr_R)
        if rcl<300:
            accuracy=0
        plt.gca().add_artist(plt.Circle((ycl, xcl), rcl, color='b', fill=False))
    except Exception:
        accuracy = 0
        pass

    # np.polyfit

    plt.xlim(right=599)  # xmax is your value
    plt.xlim(left=0)  # xmin is your value
    plt.ylim(top=599)  # ymax is your value
    plt.ylim(bottom=0)  # ymin is your value
    plt.gca().invert_yaxis()
    # plt.title(str(i))
    # plt.savefig("op/op_reg" + str(i) + ".png", dpi=300)
    plt.title(foldername + filename)
    # plt.savefig("op/" + foldername + "_" + filename + ".png", dpi=300)
    plt.savefig("ware_dataset/op/" + foldername + "_" + filename + ".png", dpi=300)
    plt.close()

    return 0


# rootdir = 'D:/train_set/clips/0531'
rootdir = 'D:/train_set/clips/examine'
foldername = "null"

for subdir, dirs, files in os.walk(rootdir):

    for file in files:
        impath = os.path.join(subdir, file)
        input_img = cv2.imread(impath)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img[:,:,0]
        foldername = subdir[-19:]
        filename = file.split(".")[0]
        i = 0
        transform(input_img, i)
        # print(accuracy)


# jsonString = json.dumps(accuracy)
# with open('accuracy.json', 'w') as outfile:
#     json.dump(jsonString, outfile)

# for i in range(1, 20):
#     input_img = cv2.imread('1492626805094402903/' + str(i) + '.jpg')
#     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#     transform(input_img, i)
