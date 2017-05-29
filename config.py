import random

import cv2
import numpy as np

cropped_image_shape = (15, 15, 1)
network_input_shape = (7,)
vein_pics_to_no_vein_ratio = 0.5
input_num = np.prod(network_input_shape)
pics_containing_vein_ratio = 0.
number_of_samples = 20000
ratio_training_to_test = 0.8
save_files_as_images = False
predic_woth_keras = True

def calculate_hu_moments(img: np.ndarray):
    hu_moments = cv2.HuMoments(cv2.moments(img)).flatten()
    return hu_moments


def select_channel(data: np.ndarray):
    return data[:, :, 1]


def hist_equalize(data: np.ndarray):
    return cv2.equalizeHist(data)


def invert(data: np.ndarray):
    return 255 - data


def reshape(data: np.ndarray):
    return np.reshape(data, input_num)


def divide(data: np.ndarray):
    return np.divide(data, 255)


def calculate_mean_std_dev(data: np.ndarray):
    (means, stds) = cv2.meanStdDev(data)
    return np.concatenate([means, stds]).flatten()


def calculate_mean_std_and_hu(data: np.ndarray):
    return np.append(calculate_hu_moments(data), calculate_hu_moments(data[:, :, 1]))


preprocessing = [select_channel, invert, calculate_hu_moments]


def preprocess(data: np.ndarray):  # function composition
    for func in preprocessing:
        data = func(data)
    return data


if __name__ == '__main__':
    zeros = np.zeros((27, 27))
    hu = calculate_hu_moments(zeros)
    print(hu)
