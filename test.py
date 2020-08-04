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
                                      use_sample_covariance=False, multichannel=False)

def _save_image(image, path):
  image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  cv.imwrite(path, image)


def main():
  # parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--model', type=str, default='edsr', help='Name of the model.')

  parser.add_argument('--scales', type=str, default='4', help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
  parser.add_argument('--cuda_device', type=str, default='-1', help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  parser.add_argument('--restore_path', type=str, required=True, help='Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')

  parser.add_argument('--chop_forward', action='store_true', help='Employ chop-forward to reduce the memory usage.')
  parser.add_argument('--chop_overlap_size', type=int, default=20, help='The overlapping size for the chop-forward process. Should be even.')

  args, remaining_args = parser.parse_known_args()


  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
  scale_list = list(map(lambda x: int(x), args.scales.split(',')))

  # model
  print('prepare model - %s' % (args.model))
  MODEL_MODULE = importlib.import_module('models.' + args.model)
  model = MODEL_MODULE.create_model()
  _, remaining_args = model.parse_args(remaining_args)
  model.prepare(is_training=False, scales=scale_list)
  scale = scale_list[0]

  # check remaining args
  if (len(remaining_args) > 0):
    print('WARNING: found unhandled arguments: %s' % (remaining_args))

  # model > restore
  model.restore(ckpt_path=args.restore_path)
  print('restored the model')

  # datasets
  input_root_path = 'C:/aim2020/data/test_LR'
  truth_root_path = 'c:/aim2020/data/test_HR'
  output_root_path = os.path.join('C:/aim2020/data/test_SR/', args.model)
  os.makedirs(output_root_path, exist_ok=True)
  log = open(os.path.join(output_root_path, 'log.txt'), 'w')

  datasets = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']

  # test
  print('begin test')
  average_psnr_list = []
  average_ssim_list = []
  duration_list = []
  # for dataset in datasets:
  for dataset in ['Urban100']:
    input_path = os.path.join(input_root_path, dataset)
    truth_path = os.path.join(truth_root_path, dataset)
    output_path = os.path.join(output_root_path, dataset)
    os.makedirs(output_path, exist_ok=True)
    image_name_list = [f for f in os.listdir(input_path) if f.lower().endswith('.png')]

    print(f'{dataset}: {len(image_name_list)} images are prepared')
    log.write(f'{dataset}: {len(image_name_list)} images are prepared\n')

    start_time = time.perf_counter()
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
      for image_index, image_name in enumerate(image_name_list):
        image_input_path = os.path.join(input_path, image_name)
        image_truth_path = os.path.join(truth_path, image_name)
        image_output_path = os.path.join(output_path, os.path.splitext(image_name)[0]+'.png')

        # load image as form [C, H, W]
        input_image = cv.imread(image_input_path)
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        input_image = np.transpose(input_image, [2, 0, 1])
        if args.model == 'msrr_test':
          input_image = torch.from_numpy(np.ascontiguousarray(input_image)).float().div(255.).unsqueeze(0)

        # truth image form [H, W, C]
        truth_image = cv.imread(image_truth_path)
        truth_image = cv.cvtColor(truth_image, cv.COLOR_BGR2RGB)

        if args.model == 'msrr_test':
          output_tensor = model.test(input_image)
        else:
          output_tensor = model.test(input_list=[input_image])

        # output image form [H, W, C], uint8, and shave
        if args.model == 'msrr_test':
          output_image = tensor2uint(output_tensor)
        else:
          output_image = output_tensor.detach().cpu().numpy()[0]
          output_image = _image_to_uint8(output_image)
          output_image = np.transpose(output_image, [1, 2, 0])

        cropped_output_image = shave(output_image, scale)

        # truth image and shave
        truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)
        truth_image = _image_to_uint8(truth_image)
        cropped_truth_image = shave(truth_image, scale)

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

        print('x%d, %d/%d, psnr=%.4f, ssim=%.4f' % (scale, image_index+1, len(image_name_list), psnr, ssim))
        log.write('x%d, %d/%d, psnr=%.4f, ssim=%.4f\n' % (scale, image_index+1, len(image_name_list), psnr, ssim))

    average_psnr = np.mean(psnr_list)
    average_psnr_list.append(average_psnr)

    average_ssim = np.mean(ssim_list)
    average_ssim_list.append(average_ssim)

    duration = time.perf_counter() - start_time
    duration_list.append(duration)

    print('x%d, %s dataset, psnr=%.4f, ssim=%.4f, duration=%.0f' % (scale, dataset, average_psnr, average_ssim, duration))

  for i, dataset in enumerate(datasets):
    print(f'{dataset}, psnr={average_psnr_list[i]:.4f}, ssim={average_ssim_list[i]:.4f}, duration={duration_list[i]}')
    log.write(f'{dataset}, psnr={average_psnr_list[i]:.4f}, ssim={average_ssim_list[i]:.4f}, duration={duration_list[i]}\n')
  log.close()
    
  # finalize
  print('finished')


if __name__ == '__main__':
  main()