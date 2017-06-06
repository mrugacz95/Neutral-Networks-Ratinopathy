import datetime
import glob
import random
from array import array
from os.path import basename, splitext

import sys
from etaprogress.progress import ProgressBar
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
from config import calculate_hu_moments, cropped_image_shape, hist_equalize
from implementation.network import Network
from load_data import load_retinopathy_data

from sklearn.preprocessing import normalize, scale


def predict(input_data, loaded_model):
    value = loaded_model.predict_proba(input_data, verbose=1)
    return value


def load_model():
    from keras.models import model_from_json
    print('Load model')
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

def cert(data):
    return np.divide(np.add((data[:, 0] - data[:, 1]), 1), 2)

def process_image(filepath: str, output: str):
    input_w, input_h, d = cropped_image_shape
    image = cv2.imread(filepath)
    predict_date = datetime.datetime.now().strftime("%Y %m %d %H %M %S")
    source_file_path = output + predict_date + '_source.jpg'
    cv2.imwrite(source_file_path, image)
    w, h, d = image.shape
    result = np.zeros((w, h), np.uint8)
    image = config.preprocess(image, config.full_image_preprocess)
    output_shape = (w - input_w) * (h - input_h)
    network_input = np.empty((output_shape, config.input_num))
    index = 0
    bar = ProgressBar(output_shape)
    for x in range(0, w - input_w):
        for y in range(0, h - input_h):
            cropped_img = image[x:x + input_w, y:y + input_h]
            cropped_img = config.preprocess(cropped_img, config.cropped_preprocessing)
            network_input[index] = cropped_img
            if index % 1000 == 0:
                bar.numerator = index
                print(bar)
                sys.stdout.flush()
            index += 1

    network_input = scale(network_input, axis=1)
    print('Predict with', 'Keras' if config.predict_with_keras else 'Own implementation')
    if config.predict_with_keras:
        values = predict(network_input, load_model())
        certainty = cert(values)
        values = np.argmax(values, axis=1).astype(np.uint8)
        certainty = np.multiply(certainty, 255)
        certainty = np.reshape(certainty, (w - input_w, h - input_h)).astype(np.uint8)

        cv2.imwrite(output + predict_date + '_certainty.jpg', certainty)
        plt.imshow(certainty)
        plt.show()
    else:
        network = Network.load_model()
        values = network.predict(network_input)
    values *= 255
    values = np.reshape(values, (w - input_w, h - input_h))
    cv2.imshow('result', values)
    cv2.imwrite(output + predict_date + '.jpg', values)
    cv2.imwrite(output + predict_date + '_thresholding.jpg', values)
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
    test_file_path = glob.glob('test_files/test2*')[0]
    test_file_name = splitext(basename(test_file_path))[0]
    result = process_image(test_file_path, 'test_files/output/')
    # d = np.array([ [0,1] , [0.1,0.7], [0.3, 0.5],[0.5,0.502], [1,0.0]])
    # print(cert(d))
