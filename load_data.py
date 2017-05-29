import glob
import pickle
import random
import sys
from os.path import basename

import cv2
import numpy as np
from etaprogress.progress import ProgressBar

from config import ratio_training_to_test, input_num


def split_data(data, results):
    num_train = round(len(data) * ratio_training_to_test)
    X_train, X_test = np.split(data, [num_train])
    y_train, y_test = np.split(results, [num_train])
    print('Loaded data')
    return (X_train, y_train), (X_test, y_test), (7,)


def load_retinopathy_data():
    random.seed(77)
    positive = glob.glob('test_files/cropped_images/*1.00.jpg')[:4000]
    negative = glob.glob('test_files/cropped_images/*0.00.jpg')[:4000]
    all_images = positive + negative
    random.shuffle(all_images)
    data = np.empty((len(all_images), input_num))
    results = np.empty(len(all_images))
    print('Loading data')
    bar = ProgressBar(len(all_images))
    for idx, file_path in enumerate(all_images):
        file_name = basename(file_path)
        num, score = file_name.replace(r'.jpg', '').split('_')
        score = round(float(score))
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, input_num).astype('float32')
        data[idx] = img
        results[idx] = score
        if idx % 1000 == 0:
            bar.numerator = idx
            print(bar)
            sys.stdout.write('\r')
            sys.stdout.flush()
        return split_data(data, results)


def load_from_pickle():
    dataset = pickle.load(open("pdataset.p", "rb"))
    return split_data(dataset['X'], dataset['y'])


if __name__ == '__main__':
    data = load_from_pickle()
