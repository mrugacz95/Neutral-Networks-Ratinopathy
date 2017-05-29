import argparse
import glob
import math
import os
import pickle
import random
import sys
from os.path import basename, splitext

import cv2
import numpy as np
from etaprogress.progress import ProgressBar

import config


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--source',
                        type=str,
                        help='Path to full images',
                        default='test_files/full_images/*')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help='Path to output files',
                        default='test_files/cropped_images/')
    parser.add_argument('-r',
                        '--save-result-images',
                        type=bool,
                        help='Write True to save also black and white cropped result images',
                        default=False)
    args = parser.parse_args()
    return args


def crop_image(source_image: np.ndarray, result_image: np.ndarray, picture_of_vein: bool) -> (
        np.ndarray, np.ndarray, bool):
    w, h, channels = config.cropped_image_shape
    while True:
        x_offset = random.randint(0, source_image.shape[0] - w)
        y_offset = random.randint(0, source_image.shape[1] - h)
        cropped_result_image = result_image[x_offset: x_offset + w, y_offset: y_offset + h]
        if config.pics_containing_vein_ratio > random.random() and not cropped_result_image[cropped_result_image > 0].any():
            continue
        is_vein = np.round(result_image[x_offset + math.ceil(w / 2), y_offset + math.ceil(h / 2)] / 255)
        if picture_of_vein != is_vein:
            continue
        cropped_image = np.array(source_image[x_offset: x_offset + w, y_offset: y_offset + h])
        cropped_image = config.preprocess(cropped_image)
        return cropped_image, cropped_result_image, is_vein


def print_progressbar(count, bar):
    if count % 100 == 0:
        bar.numerator = count
        print(bar, end='\r')
        sys.stdout.flush()


def main():
    args = load_config()
    save_result_images = args.save_result_images
    source_files = glob.glob(args.source)
    crop_per_file = int(math.ceil(int(config.number_of_samples) / len(source_files)))
    print('Cropped samples from one image: ', crop_per_file)
    count = 0
    output_dir = args.output
    vein_images_count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_path in glob.glob(output_dir + "*.jpg"):
        os.remove(file_path)

    bar = ProgressBar(config.number_of_samples)
    pickle_output = { 'X' :np.empty((config.number_of_samples, config.input_num), dtype=float),
                      'y':np.empty(config.number_of_samples, dtype=int)
                      }
    for file_path in source_files:
        full_image = cv2.imread(file_path)
        output_image_filepath = glob.glob('test_files/full_results/' + splitext(basename(file_path))[0] + '*')[0]
        output_image = cv2.imread(output_image_filepath, cv2.IMREAD_GRAYSCALE)
        for photo in range(crop_per_file):
            cropped_image, cropped_output_image, is_vein = crop_image(full_image, output_image,
                                                                      count * config.vein_pics_to_no_vein_ratio > vein_images_count)
            # save
            if config.save_files_as_images:
                output_path = output_dir + str(count) + '_' + '%.2f' % is_vein + '.jpg'
                cv2.imwrite(output_path, cropped_image)
            pickle_output['X'][count] = cropped_image
            pickle_output['y'][count] = is_vein
            if save_result_images:
                output_path = output_dir + str(count) + '_' + 'output' + '_' + '%.2f' % is_vein + '.jpg'
                cv2.imwrite(output_path, cropped_output_image)
            if is_vein:
                vein_images_count += 1
            print_progressbar(count, bar)
            count += 1
            if count == config.number_of_samples:
                break
        if count == config.number_of_samples:
            break
    pickle.dump(pickle_output, open('pdataset.p', 'wb'))
    for i in range(0,10):
        print(pickle_output['X'][i])
    print('vein samples: ', vein_images_count, ', not vein samples: ', count - vein_images_count, ', all: ', count)


if __name__ == '__main__':
    main()
