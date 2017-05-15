import glob
from os.path import basename, splitext

from matplotlib import pyplot as plt
import cv2
import numpy as np


def processImage(filepath: str, output: str):
    image = cv2.imread(filepath)
    result = simpleImageProcessing(image)
    cv2.imwrite(output, result)


def gamma_correction(img, correction):
    img = img / 255.0
    img = cv2.pow(img, correction)
    return np.uint8(img * 255)


def simpleImageProcessingEqHist(image: np.ndarray) -> np.ndarray:
    rows, cols, channels = image.shape
    result = image.copy()
    result[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    result[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    result[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    blur = cv2.GaussianBlur(result, (5, 5), 0)
    gray_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    inv_image = cv2.bitwise_not(gray_image)
    mask = np.zeros((rows, cols), np.uint8)
    mask = cv2.circle(mask, (int(rows / 2), int(cols / 2)), int(image.shape[0] / 2), (255, 255, 255), thickness=-1)
    gama_cor = gamma_correction(inv_image, 2.0)
    result = cv2.bitwise_and(mask, gama_cor, mask=mask)
    return result


def simpleImageProcessing(image: np.ndarray) -> np.ndarray:
    #image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    #image = cv2.equalizeHist(image[:, :, 1])
    #image = image[:, :, 1]
    #image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.equalizeHist(image[:, :, 1])
    image = cv2.medianBlur(image, 9)
    image = gamma_correction(image, 1.2)
    #ret, th1 = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 2)
    th3 = 255 - th3
    return th3


if __name__ == '__main__':
    test_file_path = glob.glob('test_files/full_images/*')[0]
    test_file = splitext(basename(test_file_path))[0]
    expected_file = glob.glob('test_files/full_results/*')
    image = cv2.imread(test_file_path)
    expected = cv2.imread(expected_file[0])
    result = simpleImageProcessing(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    expected = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
    difference = result - expected
    plt.subplot(141), plt.imshow(image), plt.title('ORIGINAL')
    plt.subplot(142), plt.imshow(expected, 'gray'), plt.title('EXPECTED')
    plt.subplot(143), plt.imshow(result, 'gray'), plt.title('RESULT')
    plt.subplot(144), plt.imshow(difference, 'gray'), plt.title('DIFFERENCE')

    plt.show()
    cv2.waitKey()
