import random

import cv2
import numpy as np

cropped_image_shape = (21, 21, 1)
network_input_shape = (441,)
vein_pics_to_no_vein_ratio = 0.25
input_num = np.prod(network_input_shape)
pics_containing_vein_ratio = 0.25
number_of_samples = 250000
ratio_training_to_test = 0.8
predict_with_keras = True
save_files_as_images = False
pack_to_pickle = True
epochs = 50


def calculate_hu_moments(img: np.ndarray):
    img = img.astype(np.float)
    hu_moments = cv2.HuMoments(cv2.moments(img)).flatten()
    hu_moments[hu_moments == 0] = 1
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
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
    return np.append(calculate_hu_moments(data), calculate_hu_moments(data))


def normalize_by_feature(data: np.ndarray, min_vec: np.ndarray = None, max_vec: np.ndarray = None):
    data = data.T
    if min_vec is None or max_vec is None:
        min_vec = np.empty(data.shape[1])
        max_vec = np.empty(data.shape[1])
        for idx, row in enumerate(data):
            min_value = np.min(row)
            max_value = np.max(row)
            print((max_value - min_value))
            data[idx] = (row - min_value) / (max_value - min_value)
            min_vec[idx] = min_value
            max_vec[idx] = max_value
    else:
        for idx, row in enumerate(data):
            print((max_vec[idx] - min_vec[idx]))
            data[idx] = (row - min_vec[idx]) / (max_vec[idx] - min_vec[idx])
    print(min_vec, max_vec)
    return data.T


def blur(data: np.ndarray):
    return cv2.blur(data, (5, 5))


def select_graysacle(data: np.ndarray):
    return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)


full_image_preprocess = [select_channel, hist_equalize, invert, divide]
cropped_preprocessing = [reshape]


def preprocess(data: np.ndarray, functions):  # function composition
    for func in functions:
        data = func(data)
    return data


if __name__ == '__main__':
    data = np.array([[5,2],[1,9],[3,2],[3,3]])
    print(np.argmax(data, axis=1))

