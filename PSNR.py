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


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def _fit_truth_image_size(output_image, truth_image):
    return truth_image[0:output_image.shape[0], 0:output_image.shape[1], :]


def shave(img, border):
    img = img[border:-border, border:-border, :]
    return img


def _image_psnr(im1, im2):
    return skimage.measure.compare_psnr(im1, im2)


def _image_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    return skimage.measure.compare_ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5,
                                        use_sample_covariance=False, multichannel=isRGB)


def _save_image(image, path):
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(path, image)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', type=str, default='4',
                        help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')

    parser.add_argument('--chop_forward', action='store_true', help='Employ chop-forward to reduce the memory usage.')
    parser.add_argument('--chop_overlap_size', type=int, default=20,
                        help='The overlapping size for the chop-forward process. Should be even.')

    args, remaining_args = parser.parse_known_args()

    # scale
    scale_list = list(map(lambda x: int(x), args.scales.split(',')))
    scale = scale_list[0]

    # check remaining args
    if (len(remaining_args) > 0):
        print('WARNING: found unhandled arguments: %s' % (remaining_args))

    # datasets
    truth_root_path = 'c:/aim2020/data/test_HR/Urban100'
    output_root_path = 'C:/aim2020/data/test_SR/Urban100'
    log = open(os.path.join(output_root_path, 'log.txt'), 'w')

    models = ['msrr', 'LarvaNet']

    # test
    print('begin test')
    psnr_total_list = []
    ssim_total_list = []
    duration_list = []
    for model in models:
        truth_path = os.path.join(truth_root_path)
        output_path = os.path.join(output_root_path, 'Urban100_' + model, 'x4')
        image_name_list = [f for f in os.listdir(truth_path) if f.lower().endswith('.png')]

        print(f'-------{model} is prepared---------')
        log.write(f'-------{model} is prepared---------\n')

        start_time = time.perf_counter()
        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for image_index, image_name in enumerate(image_name_list):
                image_truth_path = os.path.join(truth_path, image_name)
                image_output_path = os.path.join(output_path, os.path.splitext(image_name)[0] + '.png')

                # truth image form [H, W, C]
                truth_image = cv.imread(image_truth_path)
                truth_image = cv.cvtColor(truth_image, cv.COLOR_BGR2RGB)

                # output image form [H, W, C], uint8, and shave
                output_image = cv.imread(image_output_path)
                output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)
                output_image = _image_to_uint8(output_image)
                cropped_output_image = shave(output_image, 4)

                # truth image and shave
                truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)
                truth_image = _image_to_uint8(truth_image)
                cropped_truth_image = shave(truth_image, 4)

                # evaluate on YCbCr channels
                test_output_image = skimage.color.rgb2ycbcr(cropped_output_image)[:, :, 0]
                test_output_image = _image_to_uint8(test_output_image)
                test_truth_image = skimage.color.rgb2ycbcr(cropped_truth_image)[:, :, 0]
                test_truth_image = _image_to_uint8(test_truth_image)

                psnr = _image_psnr(test_output_image, test_truth_image)
                psnr_list.append(psnr)

                ssim = _image_ssim(test_output_image, test_truth_image)
                ssim_list.append(ssim)

                _save_image(output_image, image_output_path)

                print('x%d, %d/%d, psnr=%.4f, ssim=%.4f' % (scale, image_index + 1, len(image_name_list), psnr, ssim))
                log.write(
                    'x%d, %d/%d, psnr=%.4f, ssim=%.4f\n' % (scale, image_index + 1, len(image_name_list), psnr, ssim))

        psnr_total_list.append(psnr_list)
        ssim_total_list.append(ssim_list)

        average_psnr = np.mean(psnr_list)
        average_ssim = np.mean(ssim_list)

        print('x%d, %s model, psnr=%.4f, ssim=%.4f' % (
        scale, model, average_psnr, average_ssim))
        log.write('x%d, %s model, psnr=%.4f, ssim=%.4f\n' % (
        scale, model, average_psnr, average_ssim))

    for i, larva_psnr in enumerate(psnr_total_list[-1]):
        print(
            f'image_{i:3d}, larva_psnr={larva_psnr:.2f}, msrr_psnr={psnr_total_list[-3][i]:.2f}')
        print(f'diff = {larva_psnr-psnr_total_list[-3][i]:.3f}')
        log.write(
            f'image_{i:3d}, larva_psnr={larva_psnr:.2f}, msrr_psnr={psnr_total_list[-3][i]:.2f}\n')
        log.write(f'diff = {larva_psnr-psnr_total_list[-3][i]:.3f}\n')

    log.close()

    # finalize
    print('finished')


if __name__ == '__main__':
    main()