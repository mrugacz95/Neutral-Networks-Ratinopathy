import cv2
import numpy as np


def processImage(filepath: str, output: str):
    image = cv2.imread(filepath)
    cv2.imwrite(output, image)

# data from http://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    dict = unpickle('./neural_files/data_batch_1')
    print(dict[b'filenames'])
    print(dict[b'data'])
    image = dict[b'data'][1]
    print(image.shape)
    data = np.zeros((32, 32, 3), dtype=np.uint8)
    data[:,:,0] = image[:1024].reshape(32,32)
    data[:,:,1] = image[1024:2048].reshape(32,32)
    data[:,:,2] = image[2048:].reshape(32,32)
    print(image.shape)
    data = cv2.resize(data, (200, 200))
    cv2.imshow('image', data)
    cv2.waitKey()



if __name__ == '__main__':
    main()
