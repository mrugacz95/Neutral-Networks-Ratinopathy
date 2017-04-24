import argparse
import glob
from importlib import import_module

import cv2
import numpy as np
from os.path import basename
from image_processing import processImage as image_processing
from image_processing import processImage as neural_networks


def differece(img1: np.ndarray, img2: np.ndarray) -> float:
    return np.sum(img1 - img2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--method',
                        required=True,
                        help='Method for testing full_images (image_processing; neural_networks)')
    parser.add_argument('-s',
                        '--source-files',
                        required=True,
                        help='Files for testing method')
    parser.add_argument('-t',
                        '--test-files',
                        required=True,
                        help='Files for comparing full_results')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='Dir for output full_images')
    args = parser.parse_args()

    expected_files = glob.glob(args.test_files)
    if args.method == 'image_processing':
        processImage = image_processing
    else:
        processImage = neural_networks
    source_files = glob.glob(args.source_files)
    i = 1
    for file_path in source_files:
        file_name = basename(file_path)
        expected_file = [file for file in expected_files if file_name[:-4] in file][0]
        output_file = args.output + '/' + file_name
        processImage(file_path, output_file)
        expected = cv2.imread(expected_file)
        result = cv2.imread(output_file)
        print(i, 'out of', len(source_files), 'difference', differece(expected, result))
        i += 1


if __name__ == '__main__':
    main()
