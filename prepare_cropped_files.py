import argparse
import glob
import os
import random

import numpy as np
import cv2
import math


def crop_image(source_image: np.ndarray, shape: tuple) -> np.ndarray:
    w, h, channels = shape
    cropped_image = np.zeros(shape, np.uint8)
    x_offset = random.randint(0, source_image.shape[0] - w)
    y_offset = random.randint(0, source_image.shape[1] - h)
    cropped_image = source_image[x_offset: x_offset + w, y_offset: y_offset + h]
    return cropped_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--source',
                        type=str,
                        help='Path to full images',
                        default='test_files/full_images/*.jpg')
    parser.add_argument('-n',
                        '--number',
                        type=int,
                        help='Number of generated files',
                        default=1000)
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help='Path to output files',
                        default='test_files/cropped_images/')
    parser.add_argument('-i',
                        '--image-size',
                        type=int,
                        help='Cropped image size ',
                        default=25)
    args = parser.parse_args()
    source_files = glob.glob(args.source)
    crop_per_file = int(math.ceil(int(args.number) / len(source_files)))
    w = h = args.image_size
    crop_shape = (w, h, 3)
    count = 0
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_path in source_files:
        image = cv2.imread(file_path)
        for photo in range(crop_per_file):
            cropped_image = crop_image(image, crop_shape)
            output_path = output_dir + str(count) + '.jpg'
            cv2.imwrite(output_path, cropped_image)
            count += 1
            print(count, 'out of', args.number, 'path:',output_path)
            if count == args.number:
                break
        if count == args.number:
            break


if __name__ == '__main__':
    main()
