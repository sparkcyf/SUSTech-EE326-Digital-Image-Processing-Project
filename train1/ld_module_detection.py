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


def intensity_map(input_array, distance):
    histogram = np.sum(input_array[input_array.shape[0] // 2:, :], axis=0)
    peak_array = find_peaks(histogram, distance=distance)[0]
    return peak_array


def find_point(input_img, start_W):

    midpoint_arr_R = a = np.zeros((0, 2)).astype(int)
    # print(midpoint_arr_R)
    midpoint_tmp_R, bottom_R = find_hist_iteration(input_img, 450, start_W)
    # print(midpoint_tmp_R)

    while bottom_R > 0:
        window_H, window_WM, find_indicator = gen_windows(input_img,bottom_R, midpoint_tmp_R, 15, 100)
        midpoint_tmp_R = int(window_WM)


        if find_indicator == 1:
            tmp_list = np.array([[midpoint_tmp_R, bottom_R]]).astype(int)
            midpoint_arr_R = np.vstack([midpoint_arr_R, tmp_list])
            # plt.scatter(midpoint_tmp_R, bottom_R)
        bottom_R = bottom_R - 15
    return midpoint_arr_R


def find_hist_iteration(input_img, baseline=500, start_W=0):
    input_img_hist_iter = input_img[baseline:baseline + 149, start_W:start_W + 200]
    find_hist_iteration_iter = np.sum(input_img_hist_iter[input_img_hist_iter.shape[0] // 2:, :], axis=0)
    peak_hist = find_peaks(find_hist_iteration_iter, distance=250)[0] or [0]
    if (find_hist_iteration_iter[peak_hist[0]] < 20) or peak_hist.size == 0:
        find_hist_iteration(baseline - 50, start_W)
    else:
        pass
    return (peak_hist[0] + start_W), baseline + 50


def gen_windows(input_img, bottom_H, midpoint_W, H, W):
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
    return bottom_H - H / 2, return_mid, find_indicator

def fit_lane(input_img,midpoint):
    midpoint_arr = np.zeros((0, 2)).astype(int)
    xcl = 0
    ycl = 0
    rcl = 0

    try:
        midpoint_arr = find_point(input_img,midpoint - 100)
        # plt.scatter(x=midpoint_arr.T[0], y=midpoint_arr.T[1], s=8)
        ycl, xcl, rcl, _ = circle_fit.hyper_fit(midpoint_arr)
        # print(rcl)
        # if (rcl > 1000):
        #     plt.gca().add_artist(plt.Circle((ycl, xcl), rcl, color='r', fill=False))
    except Exception:
        pass
    return midpoint_arr, ycl, xcl, rcl