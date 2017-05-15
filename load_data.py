import glob
from os.path import basename, splitext

import sys
from etaprogress.progress import ProgressBar
import cv2
import numpy as np

ratio_taining_to_test = 0.8


def load_retinopathy_data():
    all_images = glob.glob('test_files/cropped_images/*.jpg')
    img = cv2.imread(all_images[0])
    num_pixels = np.prod(img.shape)
    shape = img.shape
    data = np.empty((len(all_images), num_pixels))
    results = np.empty(len(all_images))
    print('Loading data')
    bar = ProgressBar(len(all_images))
    for idx, file_path in enumerate(all_images):
        file_name = basename(file_path)
        num, score = file_name.replace(r'.jpg', '').split('_')
        # score = round(float(score))
        img = cv2.imread(file_path)
        # img[:,:,1] = cv2.equalizeHist(img[:,:,1])
        img = np.divide(img, 255)
        img = np.reshape(img, num_pixels).astype('float32')
        data[idx] = img
        results[idx] = score
        if idx % 1000 == 0:
            bar.numerator = idx
            print(bar)
            sys.stdout.write('\r')
            sys.stdout.flush()
    num_train = round(len(data) * ratio_taining_to_test)
    X_train, X_test = np.split(data, [num_train])
    y_train, y_test = np.split(results, [num_train])
    print('Loaded data')
    return (X_train, y_train), (X_test, y_test), shape


if __name__ == '__main__':
    load_retinopathy_data()
