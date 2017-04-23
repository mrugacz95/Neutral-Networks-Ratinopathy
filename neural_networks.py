import cv2


def processImage(filepath: str, output: str):
    image = cv2.imread(filepath)
    cv2.imwrite(output, image)

if __name__ == '__main__':
    pass