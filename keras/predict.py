import glob
import random
from array import array
from os.path import basename, splitext

import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt

from load_data import load_retinopathy_data


def predict(input_data, loaded_model):
    return loaded_model.predict(input_data, verbose=0)


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model


def processImage(filepath: str, output: str):
    loaded_model = load_model()
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = image[400:600, 400:600]
    cv2.imshow('input',image)
    cv2.imwrite(output + 'source.jpg', image)
    result = np.zeros((image.shape[0], image.shape[1]),np.uint8)
    for x in range(28, image.shape[0] - 28):
        for y in range(28, image.shape[1] - 28):
            cropped_img = image[x:x + 28, y:y + 28]
            cropped_img = np.reshape(cropped_img, (28 * 28))
            value = predict(np.array([cropped_img, ]), loaded_model)
            value *= 255
            print(x, y, 'value', value)
            result[x][y] = value
    cv2.imwrite(output, result)
    cv2.imshow('result', result)
    cv2.waitKey()
    return result


def main():
    (x_train, y_train), (X_test, y_test), num_pixels = load_retinopathy_data()  # mnist.load_data()
    loaded_model = load_model()
    print("Loaded model from disk")

    # num_pixels = X_test.shape[1] * X_test.shape[2]
    print(X_test.shape)
    number_of_test = 10
    random_offset = random.randint(0, X_test.shape[0] - number_of_test)
    for i in range(random_offset, random_offset + number_of_test):
        input_data = X_test[i:i + 1]
        img = input_data[0].copy()
        img = np.reshape(img, (28, 28))
        plt.imshow(img, cmap='gray')
        # input = input.reshape(1, num_pixels).astype('float32')
        result = predict(input_data)
        print(result, 'expected:', y_test[i])
        plt.show()


if __name__ == '__main__':
    test_file_path = glob.glob('test_files/full_images/*.jpg')[0]
    test_file_name = splitext(basename(test_file_path))[0]
    result = processImage(test_file_path, './test_files/results_images/' + test_file_name + '.jpg')
