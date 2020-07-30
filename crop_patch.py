import argparse
import importlib
import json
import os
import time

import torch

import dataloaders
import models
from utils import image_utils

import numpy as np
import cv2 as cv
import skimage.color
import skimage.measure


def _image_to_uint8(image):
    return np.clip(np.round(image), a_min=0, a_max=255).astype(np.uint8)


def crop_patch(image, x1, x2, y1, y2):
    return image[y1:y2, x1:x2, :]


def _save_image(image, path):
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(path, image)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', type=str, default='4',
                        help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')

    args, remaining_args = parser.parse_known_args()

    # scale
    scale_list = list(map(lambda x: int(x), args.scales.split(',')))
    scale = scale_list[0]

    # check remaining args
    if (len(remaining_args) > 0):
        print('WARNING: found unhandled arguments: %s' % (remaining_args))

    # datasets
    truth_root_path = 'C:/aim2020/data/test_HR/Urban100'
    output_root_path = 'C:/aim2020/data/test_SR/Urban100'
    result_root_path = 'C:/aim2020/data/test_patch/Urban100'
    os.makedirs(result_root_path, exist_ok=True)
    models = ['VDSR', 'LapSRN', 'MemNet', 'CARN', 'msrr', 'IMDN', 'LarvaNet']
    # images = ['img_011.png', 'img_067.png', 'img_084.png', 'img_092.png', 'img_095.png', 'img_096.png']
    # coordinates = [[45,85,700,740], [315,370,95,150], [685,720,365,400],
    #                [580,630,165,215], [275,363,400,504], [360,405,105,150]]
    images = ['img_095.png']
    coordinates = [[280, 380, 400, 500]]

    # test
    print('begin crop')
    for image, coord in zip(images, coordinates):
        truth_path = os.path.join(truth_root_path)
        image_truth_path = os.path.join(truth_path, image)
        result_path = os.path.join(result_root_path, image)
        os.makedirs(result_path, exist_ok=True)

        # truth image form [H, W, C]
        truth_image = cv.imread(image_truth_path)
        truth_image = cv.cvtColor(truth_image, cv.COLOR_BGR2RGB)
        truth_patch = crop_patch(truth_image, *coord)

        index = 1
        _save_image(truth_patch, os.path.join(result_path, f'({index}) HR.png'))

        for model in models:
            output_path = os.path.join(output_root_path, 'Urban100_' + model, 'x4')
            image_output_path = os.path.join(output_path, image)

            # output image form [H, W, C]
            output_image = cv.imread(image_output_path)
            output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)
            output_patch = crop_patch(output_image, *coord)

            index += 1
            _save_image(output_patch, os.path.join(result_path, f'({index}) {model}.png'))
    # finalize
    print('finished')


if __name__ == '__main__':
    main()