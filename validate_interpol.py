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

import torch.nn.functional as F


def _image_to_uint8(image):
  return np.clip(np.round(image), a_min=0, a_max=255).astype(np.uint8)

def _fit_truth_image_size(output_image, truth_image):
  return truth_image[:, 0:output_image.shape[1], 0:output_image.shape[2]]

def _image_psnr(output_image, truth_image):
  diff = np.float32(truth_image) - np.float32(output_image)
  mse = np.mean(np.power(diff, 2))
  psnr = 10.0 * np.log10(255.0 ** 2 / mse)
  return psnr

def _save_image(image, path):
  image = np.transpose(image, [1, 2, 0])
  image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  cv.imwrite(path, image)


def main():
  # parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataloader', type=str, default='div2k_val_loader', help='Name of the data loader.')
  parser.add_argument('--interpolate', type=str, default='bilinear', help='Name of the interpolation method.')

  parser.add_argument('--scales', type=str, default='4', help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
  parser.add_argument('--cuda_device', type=str, default='-1', help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  parser.add_argument('--save_path', type=str, help='Base output path of the upscaled images. Specify this to save the upscaled images.')

  parser.add_argument('--chop_forward', action='store_true', help='Employ chop-forward to reduce the memory usage.')
  parser.add_argument('--chop_overlap_size', type=int, default=20, help='The overlapping size for the chop-forward process. Should be even.')

  args, remaining_args = parser.parse_known_args()


  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
  scale_list = list(map(lambda x: int(x), args.scales.split(',')))

  # data loader
  print('prepare data loader - %s' % (args.dataloader))
  DATALOADER_MODULE = importlib.import_module('dataloaders.' + args.dataloader)
  dataloader = DATALOADER_MODULE.create_loader()
  _, remaining_args = dataloader.parse_args(remaining_args)
  dataloader.prepare(scales=scale_list)

  # validate
  print('begin validation')
  num_images = dataloader.get_num_images()
  average_duration_dict = {}
  average_psnr_dict = {}
  for scale in scale_list:
    duration_list = []
    psnr_list = []

    for image_index in range(num_images):
      input_image, truth_image, image_name = dataloader.get_image_pair(image_index=image_index, scale=scale)

      input_tensor = torch.tensor([input_image], dtype=torch.float32, device='cpu')
      start_time = time.perf_counter()
      output_tensor = F.interpolate(input_tensor, scale_factor=scale, mode=args.interpolate, align_corners=False)[0]
      end_time = time.perf_counter()
      output_image = output_tensor.detach().cpu().numpy()

      duration = end_time - start_time
      duration_list.append(duration)

      truth_image = _image_to_uint8(truth_image)
      output_image = _image_to_uint8(output_image)

      if (args.save_path is not None):
        os.makedirs(os.path.join(args.save_path, 'x%d' % (scale)), exist_ok=True)
        output_image_path = os.path.join(args.save_path, 'x%d' % (scale), image_name+'.png')
        _save_image(output_image, output_image_path)

      truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)

      psnr = _image_psnr(output_image=output_image, truth_image=truth_image)

      psnr_list.append(psnr)
      print('x%d, %d/%d, psnr=%.2f, duration=%.4f' % (scale, image_index+1, num_images, psnr, duration))

    average_psnr = np.mean(psnr_list)
    average_psnr_dict[scale] = average_psnr
    average_duration = np.mean(duration_list)
    average_duration_dict[scale] = average_duration
    print('x%d, psnr=%.2f, duration=%.4f' % (scale, average_psnr, average_duration))

    
  # finalize
  print('finished')


if __name__ == '__main__':
  main()