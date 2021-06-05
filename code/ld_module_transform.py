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
import ld_module_binary
from numba import njit, prange


def transform_to(input_img):

    pts1 = np.float32([[450, 300], [-851, 720], [870, 300], [2057, 720]])
    pts2 = np.float32([[0, 0], [0, 600], [600, 0], [600, 600]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(input_img, M, (600, 600))

    return dst