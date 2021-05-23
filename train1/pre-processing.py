import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import circle_fit
import os
import json
from skimage.filters import threshold_otsu
import time

start_time = time.time()



# input_img = cv2.imread('1492627187263118217/1.jpg')


def transform(input_img, original_img, num):
    # rows, cols, ch = input_img.shape
    # print(input_img.shape)

    dst, dst_extract = transform_bare(input_img, num)

    # for i in range(1,4):
    #     dst_tmp,dst_extract

    # plt.imshow(dst), plt.title(foldername + filename), plt.savefig("ware_dataset/op/" + foldername + "_" + filename + "_transform.png", dpi=300)

    ##add line mark
    dst_mark = add_line_mark(dst_extract, dst, original_img)

    # plt.subplot(131), plt.imshow(input_img), plt.title('Input')
    # plt.subplot(132), plt.imshow(dst), plt.title('Output')
    # plt.subplot(133), plt.imshow(dst_extract), plt.title('op_extracted')
    # plt.show()


def transform_bare(input_img, num):
    # rows, cols, ch = input_img.shape
    # print(input_img.shape)

    # pts1 = np.float32([[629,250],[0,445],[705,250],[1279,430]])
    pts1 = np.float32([[450, 300], [-851, 720], [870, 300], [2057, 720]])
    pts2 = np.float32([[0, 0], [0, 600], [600, 0], [600, 600]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(input_img, M, (600, 600))
    # dst_extract = abs_thresh(dst, sobel_kernel=3, mag_thresh=(35, 210), direction='x')

    # tresh
    kernel_size = 7
    mag_thresh = (30, 150)
    r_thresh = (235, 255)
    s_thresh = (165, 255)
    b_thresh = (160, 255)
    g_thresh = (210, 255)
    dst_extract = get_bin_img(dst, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh,
                              s_thresh=s_thresh, b_thresh=b_thresh, g_thresh=g_thresh) + abs_thresh(dst, sobel_kernel=3, mag_thresh=(35, 210), direction='x')
    # dst_extract = dst

    return dst, dst_extract


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


def abs_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    return binary


def add_line_mark(input_img, dst, original_img):
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
        window = input_img[bottom_H - 15:bottom_H, midpoint_W - 50:midpoint_W + 50]
        H = 15
        W = 100
        # window_arr = input_img[]
        histogram_window = np.sum(window[window.shape[0] // 2:, :], axis=0)
        window_peak = find_peaks(histogram_window, distance=100)[0]

        return_mid = 0

        if window_peak.size > 0:
            if histogram_window[window_peak] > 5 and np.abs(window_peak - (W / 2)) < 40:
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
    # plt.subplot(131)
    # plt.imshow(input_img)
    # plt.imshow(dst)

    input_img_hist_all = np.sum(input_img[input_img.shape[0] // 2:, :], axis=0)
    peak_hist_all = find_peaks(input_img_hist_all, distance=150)
    print(peak_hist_all)

    # iteraton(L)
    # print(peak_L)

    def find_hist_iteration(baseline=500, start_W=0):
        # print("startWis" + str(start_W)+ "baseline" + str(baseline))

        input_img_hist_iter = input_img[baseline:baseline + 149, start_W:start_W + 200]
        find_hist_iteration_iter = np.sum(input_img_hist_iter[input_img_hist_iter.shape[0] // 2:, :], axis=0)
        # print(np.max(find_hist_iteration_iter))
        # print("startWis" + str(start_W) + "peakhist" + str(find_peaks(find_hist_iteration_iter, distance=250)[0]))
        peak_hist = find_peaks(find_hist_iteration_iter, distance=250)[0] or [0]

        if (find_hist_iteration_iter[peak_hist[0]] < 20) or peak_hist.size == 0:
            # print(find_peaks(find_hist_iteration_iter, distance=180))
            # print(baseline)
            # print(start_W)
            find_hist_iteration(baseline - 50, start_W)
        else:
            pass

        # print(peak_hist[0])
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
            window_H, window_WM, find_indicator = gen_windows(bottom_R, midpoint_tmp_R, 15, 100)
            midpoint_tmp_R = int(window_WM)

            if find_indicator == 1:
                tmp_list = np.array([[midpoint_tmp_R, bottom_R]]).astype(int)
                midpoint_arr_R = np.vstack([midpoint_arr_R, tmp_list])
                # plt.scatter(midpoint_tmp_R, bottom_R)
            bottom_R = bottom_R - 15
        return midpoint_arr_R

    def fit_lane(midpoint):
        midpoint_arr = np.zeros((0, 2)).astype(int)
        xcl = 0
        ycl = 0
        rcl = 0
        try:
            midpoint_arr = find_point(midpoint - 100)
            # print(midpoint_arr.T)
            # plt.scatter(x=midpoint_arr.T[0], y=midpoint_arr.T[1], s=8)
            ycl, xcl, rcl, _ = circle_fit.hyper_fit(midpoint_arr)
            # print(rcl)
            # if (rcl > 1000):
            #     plt.gca().add_artist(plt.Circle((ycl, xcl), rcl, color='r', fill=False))
        except Exception:
            pass
        return midpoint_arr, ycl, xcl, rcl

    mark_layer = np.zeros((600, 600, 3), np.uint8)
    for j in peak_hist_all[0]:
        point_array,ycl,xcl,rcl = fit_lane(j)

        if point_array.size > 0 and rcl > 800 and rcl < 2147483647:
            # print(point_array)
            for point in point_array:
                cv2.circle(mark_layer,tuple(point),10,(0,0,255),thickness=-1)
                cv2.circle(mark_layer,tuple([int(ycl),int(xcl)]),int(rcl),(0,255,0),thickness=10)

    pts2 = np.float32([[450, 300], [-851, 720], [870, 300], [2057, 720]])
    pts1 = np.float32([[0, 0], [0, 600], [600, 0], [600, 600]])

    M1 = cv2.getPerspectiveTransform(pts1, pts2)

    mark_layer_trans = cv2.warpPerspective(mark_layer, M1, (1280,720))
    mark_layer_mask = np.where(mark_layer_trans > 0,0,1)
    original_img = original_img*mark_layer_mask + mark_layer_trans
    plt.imshow(original_img)
    # mark_layer_trans = cv2.addWeighted(mark_layer_trans, 1, original_img, 1, 0.0)



    # try:
    #     midpoint_arr_L = find_point(0)
    #
    #     print(midpoint_arr_L.T)
    #     plt.scatter(x=midpoint_arr_L.T[0], y=midpoint_arr_L.T[1])
    #     ycl, xcl, rcl, _ = circle_fit.hyper_fit(midpoint_arr_L)
    #     plt.gca().add_artist(plt.Circle((ycl, xcl), rcl, color='r', fill=False))
    # except Exception:
    #     pass
    #
    # try:
    #     midpoint_arr_R = find_point(300)
    #     print(midpoint_arr_R.T)
    #     plt.scatter(x=midpoint_arr_R.T[0], y=midpoint_arr_R.T[1])
    #     ycl, xcl, rcl, _ = circle_fit.hyper_fit(midpoint_arr_R)
    #     plt.gca().add_artist(plt.Circle((ycl, xcl), rcl, color='b', fill=False))
    # except Exception:
    #     accuracy = 0
    #     pass

    # np.polyfit

    # plt.xlim(right=599)  # xmax is your value
    # plt.xlim(left=0)  # xmin is your value
    # plt.ylim(top=599)  # ymax is your value
    # plt.ylim(bottom=0)  # ymin is your value
    # plt.gca().invert_yaxis()
    # # plt.title(str(i))
    # # plt.savefig("op/op_reg" + str(i) + ".png", dpi=300)
    # plt.title(foldername + filename)
    # # plt.savefig("op/" + foldername + "_" + filename + ".png", dpi=300)
    # plt.subplot(132)
    # plt.imshow(dst)
    # plt.savefig("op/" + foldername + "_" + filename + ".png", dpi=300)
    # # print("ware_dataset/op/" + foldername + "_" + filename + ".png")

    # plt.xlim(right=1279)  # xmax is your value
    # plt.xlim(left=0)  # xmin is your value
    # plt.ylim(top=719)  # ymax is your value
    # plt.ylim(bottom=0)  # ymin is your value
    # plt.gca().invert_yaxis()
    # # plt.title(str(i))
    # # plt.savefig("op/op_reg" + str(i) + ".png", dpi=300)
    # plt.title(foldername + "_" + filename)
    # # plt.savefig("op/" + foldername + "_" + filename + ".png", dpi=300)
    # # plt.subplot(132)
    # # plt.imshow(dst)
    # plt.savefig("op/" + foldername + "_" + filename + ".png", dpi=300)
    # # print("ware_dataset/op/" + foldername + "_" + filename + ".png")

    # plt.close()
    cv2.imwrite("op/" + foldername + "_" + filename + ".png", original_img)

    return 0


rootdir = 'D:/train_set/clips/0531/'
# rootdir = 'D:/train_set/clips/examine/'
foldername = "null"

for subdir, dirs, files in os.walk(rootdir):
    # print(dirs)
    for d in dirs:
        # for file in files:
        #     impath = os.path.join(subdir, file)
        # print(subdir)
        # read into seq
        # buffer first three image

        img_buffer_array = np.zeros([4, 720, 1280, 3]).astype('uint8')
        for k1 in range(1, 4):
            print(subdir + d + "/" + str(k1) + ".jpg")
            input_img = cv2.imread(subdir + d + "/" + str(k1) + ".jpg")
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            img_buffer_array[k1 - 1] = input_img
            print(k1)
        rotate_num = 3

        for k2 in range(4, 21):
            input_img = cv2.imread(subdir + d + "/" + str(k2) + ".jpg")
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            # yellow filter
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            original_img = input_img
            img_buffer_array[rotate_num] = input_img
            input_img = (img_buffer_array[(rotate_num) % 4] * 0.25 + img_buffer_array[(rotate_num - 1) % 4] * 0.25 +
                         img_buffer_array[
                             (rotate_num - 2) % 4] * 0.25 + img_buffer_array[(rotate_num - 3) % 4] * 0.25).astype(
                'uint8')
            # input_img = np.maximum.reduce([img_buffer_array[(rotate_num) % 4], img_buffer_array[(rotate_num - 1) % 4],
            #                                img_buffer_array[(rotate_num - 2) % 4],
            #                                img_buffer_array[(rotate_num - 3) % 4]]).astype('uint8')
            # input_img = input_img[:,:,0]
            # foldername = subdir[-19:]
            foldername = d
            filename = str(k2)
            i = 0
            transform(input_img,original_img, i)
            # print(accuracy)

# jsonString = json.dumps(accuracy)
# with open('accuracy.json', 'w') as outfile:
#     json.dump(jsonString, outfile)

# for i in range(1, 20):
#     input_img = cv2.imread('1492626805094402903/' + str(i) + '.jpg')
#     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#     transform(input_img, i)


print("--- %s seconds ---" % (time.time() - start_time))