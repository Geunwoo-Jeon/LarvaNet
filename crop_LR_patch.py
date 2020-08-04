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


def crop_patch(image, x1, x2, y1, y2):
    return image[y1:y2, x1:x2, :]


def _save_image(image, path):
    cv.imwrite(path, image)


def main():
    # datasets
    input_root_path = 'C:/aim2020/data/test_LR/Urban100'
    result_root_path = 'C:/aim2020/data/test_patch/Urban100'
    os.makedirs(result_root_path, exist_ok=True)
    # images = ['img_011.png', 'img_067.png', 'img_084.png', 'img_092.png', 'img_095.png', 'img_096.png']
    # coordinates = [[45,85,700,740], [315,370,95,150], [685,720,365,400],
    #                [580,630,165,215], [275,363,400,504], [360,405,105,150]]
    images = ['img_067.png']
    coordinates = [[79, 92, 24, 37]]

    # test
    print('begin crop')
    for image, coord in zip(images, coordinates):
        image_input_path = os.path.join(input_root_path, image)
        result_path = os.path.join(result_root_path, image)
        os.makedirs(result_path, exist_ok=True)

        # truth image form [H, W, C]
        input_image = cv.imread(image_input_path)
        input_patch = crop_patch(input_image, *coord)
        _save_image(input_patch, os.path.join(result_path, 'LR.png'))
    # finalize
    print('finished')


if __name__ == '__main__':
    main()
