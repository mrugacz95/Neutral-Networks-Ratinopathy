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
    return loaded_model.predict(input_data, verbose=1)


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


def processImage(filepath: str, output: str):
    loaded_model = load_model()
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #image = image[400:600, 400:600]
    cv2.imwrite(output + 'source.jpg', image)
    result = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    w, h = image.shape
    output_shape = (w - 28) * (h - 28)
    network_input = np.empty(((w - 28) * (h - 28), 28 * 28))
    index = 0
    for x in range(0, image.shape[0] - 28):
        for y in range(0, image.shape[1] - 28):
            cropped_img = image[x:x + 28, y:y + 28]
            cropped_img = np.reshape(cropped_img, (28 * 28))
            network_input[index] = cropped_img
            if index % 1000 == 0:
                print(index, ' out of ', network_input.shape[1])
            index += 1
    values = predict(network_input, loaded_model)
    values *= 255
    values = np.reshape(values, (w - 28, h - 28))
    cv2.imwrite(output, values)
    cv2.imshow('result', values)
    cv2.waitKey()
    return result


def main():
    (x_train, y_train), (X_test, y_test), num_pixels = load_retinopathy_data()  # mnist.load_data()
    loaded_model = load_model()

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
    print('Loaded tesor flow')
    test_file_path = glob.glob('test_files/full_images/*.jpg')[0]
    test_file_name = splitext(basename(test_file_path))[0]
    result = processImage(test_file_path, 'test_files/output/' + test_file_name + '.jpg')
