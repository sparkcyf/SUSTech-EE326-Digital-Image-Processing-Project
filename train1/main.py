import cv2
import os
import time
from numba import njit, prange

import matplotlib.pyplot as plt

import ld_module_transform
import ld_module_binary
import numpy as np
import ld_module_detection

start_time = time.time()

# rootdir = 'D:/train_set/clips/0531/'
rootdir = 'D:/train_set/clips/submit/test/'
foldername = "null"


def lane_detection(input_img, original_img, i):
    # 1 perspective transformation
    transformed_img = ld_module_transform.transform_to(input_img)

    # 2 input for binary operation
    kernel_size = 7
    mag_thresh = (30, 150)
    r_thresh = (235, 255)
    s_thresh = (165, 255)
    b_thresh = (160, 255)
    g_thresh = (210, 255)
    extracted_img = ld_module_binary.get_bin_img(transformed_img, kernel_size=kernel_size,
                                                 sobel_thresh=mag_thresh,
                                                 r_thresh=r_thresh,
                                                 s_thresh=s_thresh, b_thresh=b_thresh,
                                                 g_thresh=g_thresh) \
        # + ld_module_binary.abs_thresh(transformed_img,
    #                               sobel_kernel=3,
    #                               mag_thresh=(
    #                                   35, 210),
    #                               direction='x')

    # 3 add line mark
    # 3.1 determine the start point of the mark
    # 3.1.1 eliminate the lower L/R traingle
    input_img = extracted_img
    tri_matrix = np.tri(600, 600, 420).T
    input_img = (input_img * tri_matrix * np.flip(np.tri(600, 600, 440).T, 1)).astype("uint8")

    # 3.1.1.5 hough transform
    # gaussian blur
    blur_kernel = np.ones((3, 3), np.float32) / 9
    input_img = cv2.filter2D(input_img, -1, blur_kernel)

    # hough transform(could not use)
    # hough_lines = cv2.HoughLinesP(input_img, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=20, maxLineGap=200)
    # line_nos = hough_lines.shape[0]
    # hough_line_layer = np.zeros((600, 600, 1), np.uint8)
    # for i in range(line_nos):
    #     x_1 = hough_lines[i][0][0]
    #     y_1 = hough_lines[i][0][1]
    #     x_2 = hough_lines[i][0][2]
    #     y_2 = hough_lines[i][0][3]
    #     if np.abs((y_2-y_1)/(x_2-x_1))>5:
    #         cv2.line(hough_line_layer, pt1=(x_1, y_1), pt2=(x_2, y_2), color=255, thickness=2)
    #
    # plt.subplot(121)
    # plt.imshow(hough_line_layer)

    # flow1 plot
    # plt.rcParams["font.size"] = "2.5"
    # plt.subplot(151)
    # plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    # plt.title("Raw Input")
    # plt.subplot(152)
    # plt.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    # plt.title("Perspective Transformed")
    # plt.subplot(153)
    # plt.imshow(ld_module_binary.get_hls_sthresh_img(transformed_img, thresh=s_thresh))
    # plt.title("LAB Colour Space (Yellow Lane)",wrap=True)
    # plt.subplot(154)
    # plt.imshow(ld_module_binary.get_rgb_thresh_img(transformed_img, thresh=r_thresh)+ld_module_binary.get_rgb_thresh_img(transformed_img, thresh=g_thresh, channel='G'))
    # plt.title("RGB Colour Space (White Lane)",wrap=True)
    # plt.subplot(155)
    # plt.imshow(input_img)
    # plt.title("Binarization")
    # plt.savefig("flow12.svg", format='svg', dpi=600)

    # 3.1.2 find peak (line) number
    img_peak_histogram = ld_module_detection.intensity_map(input_img, 150) \
 \
    # plt.subplot(121)

    # 3.1.3 find lane via sliding window
    plt.imshow(input_img)
    mark_layer = np.zeros((600, 600, 3), np.uint8)
    for midpoint in img_peak_histogram:
        # midpoint (width)

        if 100 < midpoint < 500:
            point_array, ycl, xcl, rcl = ld_module_detection.fit_lane(input_img, midpoint)

            # plt.scatter(x=point_array.T[0], y=point_array.T[1], s=16)

            # 3.1.3.1 draw line over the original image
            if point_array.size > 0 and rcl > 800 and rcl < 2147483647:
                # and rcl > 800 and rcl < 2147483647
                for point in point_array:
                    # print(i)
                    cv2.circle(mark_layer, tuple(point), 10, (0, 0, 255), thickness=-1)
                cv2.circle(mark_layer, tuple([int(ycl), int(xcl)]), int(rcl), (0, 255, 0), thickness=10)
                # plt.gca().add_artist(plt.Circle((ycl, xcl), rcl, color='r', fill=False,lw=8))
    # 3.1.3.2 transform the mark layer back
    pts2 = np.float32([[450, 300], [-851, 720], [870, 300], [2057, 720]])
    pts1 = np.float32([[0, 0], [0, 600], [600, 0], [600, 600]])

    M1 = cv2.getPerspectiveTransform(pts1, pts2)

    mark_layer_trans = cv2.warpPerspective(mark_layer, M1, (1280, 720))
    mark_layer_mask = np.where(mark_layer_trans > 0, 0, 1)
    original_img = original_img * mark_layer_mask + mark_layer_trans

    # plt.xlim(right=599)  # xmax is your value
    # plt.xlim(left=0)  # xmin is your value
    # plt.ylim(top=599)  # ymax is your value
    # plt.ylim(bottom=0)  # ymin is your value
    # plt.gca().invert_yaxis()

    # plt.subplot(122)
    # plt.imshow(original_img)

    # plt.show()

    # 4 save image
    # simulate not save image
    cv2.imwrite("op_module/" + foldername + "_" + filename + ".png", original_img)

    plt.imshow(cv2.cvtColor(original_img.astype("uint8"), cv2.COLOR_BGR2RGB))
    plt.savefig("output.svg", format='svg', dpi=600)


for subdir, dirs, files in os.walk(rootdir):
    for d in dirs:
        # for file in files:
        #     impath = os.path.join(subdir, file)
        # read into seq
        # buffer first three image

        # img_buffer_array = np.zeros([4, 720, 1280, 3]).astype('uint8')
        # for k1 in range(1, 4):
        #     print(subdir + d + "/" + str(k1) + ".jpg")
        #     input_img = cv2.imread(subdir + d + "/" + str(k1) + ".jpg")
        #     input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        #     img_buffer_array[k1 - 1] = input_img
        #     print(k1)
        # rotate_num = 3

        for k2 in range(1, 2):
            input_img = cv2.imread(subdir + d + "/" + str(k2) + ".jpg")
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            # yellow filter
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            original_img = input_img
            # img_buffer_array[rotate_num] = input_img
            # input_img = (img_buffer_array[(rotate_num) % 4] * 0.25 + img_buffer_array[(rotate_num - 1) % 4] * 0.25 +
            #              img_buffer_array[
            #                  (rotate_num - 2) % 4] * 0.25 + img_buffer_array[(rotate_num - 3) % 4] * 0.25).astype(
            #     'uint8')
            # input_img = np.maximum.reduce([img_buffer_array[(rotate_num) % 4], img_buffer_array[(rotate_num - 1) % 4],
            #                                img_buffer_array[(rotate_num - 2) % 4],
            #                                img_buffer_array[(rotate_num - 3) % 4]]).astype('uint8')
            # input_img = input_img[:,:,0]
            # foldername = subdir[-19:]
            foldername = d
            filename = str(k2)
            i = 0

            lane_detection(input_img, original_img, k2)

print("--- %s seconds ---" % (time.time() - start_time))
