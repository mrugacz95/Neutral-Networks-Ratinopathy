import datetime
import glob
import random
from array import array
from os.path import basename, splitext

import sys
from etaprogress.progress import ProgressBar
import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt

from load_data import load_retinopathy_data


def predict(input_data, loaded_model):
    value = loaded_model.predict_proba(input_data, verbose=1)
    return value


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print("Loaded model from disk")
    return loaded_model


def process_image(filepath: str, output: str):
    loaded_model = load_model()
    network_input_shape = loaded_model.inputs[0].shape
    input_shape = (18, 18, 3)
    if np.prod(input_shape) != np.prod(network_input_shape):
        print('Wrong input shape', input_shape, '!=', network_input_shape)
        return
    input_w, input_h,d = input_shape
    num_pixels = np.prod(input_shape)
    image = cv2.imread(filepath)
    # image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    # image = cv2.equalizeHist(image)
    # image = image[400:700, 400:700]
    predict_date = datetime.datetime.now().strftime("%Y %m %d %H %M %S")
    source_file_path = output + predict_date + '_source.jpg'
    cv2.imwrite(source_file_path, image)
    image = np.divide(image,255)
    result = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    w, h,d = image.shape
    output_shape = (w - input_w) * (h - input_h)
    network_input = np.empty((output_shape, num_pixels))
    index = 0
    bar = ProgressBar(output_shape)
    for x in range(0, w - input_w):
        for y in range(0, h - input_h):
            cropped_img = image[x:x + input_w, y:y + input_h]
            cropped_img = np.reshape(cropped_img, num_pixels)
            network_input[index] = cropped_img
            if index % 1000 == 0:
                #print(index, ' out of ', output_shape)
                bar.numerator = index
                print(bar, end='\r')
                sys.stdout.flush()
            index += 1
    values = predict(network_input, loaded_model)
    values = np.reshape(values, (w - input_w, h - input_h))
    cv2.imshow('result', values)
    values *= 255
    cv2.imwrite(output + predict_date + '.jpg', values)
    #th1 = cv2.adaptiveThreshold(values.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret, th1 = cv2.threshold(values, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output + predict_date + '_thresholding.jpg', th1)
    print('\nFinished')
    cv2.waitKey()
    return result


def main():
    (x_train, y_train), (X_test, y_test), shape = load_retinopathy_data()  # mnist.load_data()
    loaded_model = load_model()

    # num_pixels = X_test.shape[1] * X_test.shape[2]
    print(X_test.shape)
    number_of_test = 10
    random_offset = random.randint(0, X_test.shape[0] - number_of_test)
    for i in range(random_offset, random_offset + number_of_test):
        input_data = X_test[i:i + 1]
        img = input_data[0].copy()
        img = np.reshape(img, shape)
        plt.imshow(img, cmap='gray')
        result = predict(input_data, loaded_model)
        print(result, 'expected:', y_test[i])
        plt.show()


if __name__ == '__main__':
    print('Loaded tesor flow')
    #test_file_path = glob.glob('test_files/full_images/Image_01L*')[0]
    test_file_path = glob.glob('test_files/test1*')[0]
    test_file_name = splitext(basename(test_file_path))[0]
    result = process_image(test_file_path, 'test_files/output/')
