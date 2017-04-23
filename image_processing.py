import glob
from os.path import basename

from matplotlib import pyplot as plt
import cv2
import numpy as np


def gamma_correction(img, correction):
    img = img / 255.0
    img = cv2.pow(img, correction)
    return np.uint8(img * 255)


def processImage(image: np.ndarray) -> np.ndarray:
    rows, cols, channels = image.shape
    result = image.copy()
    print(result.shape)
    result[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    result[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    result[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    blur = cv2.GaussianBlur(result, (5, 5), 0)
    gray_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    inv_image = cv2.bitwise_not(gray_image)
    mask = np.zeros((rows, cols), np.uint8)
    mask = cv2.circle(mask, (int(rows / 2), int(cols / 2)), int(image.shape[0] / 2), (255, 255, 255), thickness=-1)
    #mask = cv2.rectangle(mask,(500,400),(600,800), (255,255,255), thickness=-1)
    gama_cor = gamma_correction(inv_image, 2.0)
    result = cv2.bitwise_and(mask, gama_cor, mask=mask)
    return result


if __name__ == '__main__':
    test_file_path = glob.glob('test_files/images/*.jpg')[0]
    test_file = basename(test_file_path)[:-4]
    expected_file = glob.glob('test_files/results/Image_01*.png')
    image = cv2.imread(test_file_path)
    expected = cv2.imread(expected_file[0])
    result = processImage(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    expected = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
    difference = np.subtract(result, expected)

    difference = cv2.equalizeHist(difference)
    plt.subplot(141), plt.imshow(image), plt.title('ORIGINAL')
    plt.subplot(142), plt.imshow(expected, 'gray'), plt.title('EXPECTED')
    plt.subplot(143), plt.imshow(result, 'gray'), plt.title('RESULT')
    plt.subplot(144), plt.imshow(difference, 'gray'), plt.title('DIFFERENCE')

    plt.show()
    cv2.waitKey()
