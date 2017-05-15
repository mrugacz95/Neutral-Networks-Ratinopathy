import cv2
import numpy as np


def processImage(filepath: str, output: str):
    image = cv2.imread(filepath)
    cv2.imwrite(output, image)

def main():
    pass

if __name__ == '__main__':
    main()
