from typing import Callable

import cv2
import numpy as np

cropped_image_shape = (27, 27, 1)
network_input_shape = (7,)
vein_pics_to_no_vein_ratio = 0.5
input_num = np.prod(network_input_shape)
select_only_pics_containing_vein = True
number_of_samples = 4000
ratio_training_to_test = 0.8


def calculate_hu_moments(img: np.ndarray):
    return cv2.HuMoments(cv2.moments(img)).flatten()


def select_channel(data: np.ndarray):
    return data[:, :, 1]


def hist_equalize(data: np.ndarray):
    return cv2.equalizeHist(data)


preprocessing = [select_channel, hist_equalize, calculate_hu_moments]


def preprocess(data: np.ndarray):  # function composition
    for func in preprocessing:
        data = func(data)
    return data
