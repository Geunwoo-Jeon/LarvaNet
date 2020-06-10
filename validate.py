import argparse
import importlib
import json
import os
import time

import dataloaders
import models
from utils import image_utils

import numpy as np
import cv2 as cv


def _clip_image(image):
  return np.clip(np.round(image), a_min=0, a_max=255)

def _fit_truth_image_size(output_image, truth_image):
  return truth_image[:, 0:output_image.shape[1], 0:output_image.shape[2]]

def _image_psnr(output_image, truth_image):
  diff = truth_image - output_image
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

  parser.add_argument('--dataloader', type=str, default='div2k_loader', help='Name of the data loader.')
  parser.add_argument('--model', type=str, default='edsr', help='Name of the model.')

  parser.add_argument('--scales', type=str, default='4', help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
  parser.add_argument('--cuda_device', type=str, default='-1', help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  parser.add_argument('--restore_path', type=str, required=True, help='Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
  parser.add_argument('--restore_target', type=str, help='Target of the restoration.')
  parser.add_argument('--restore_global_step', type=int, default=0, help='Global step of the restored model. Some models may require to specify this.')

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

  # model
  print('prepare model - %s' % (args.model))
  MODEL_MODULE = importlib.import_module('models.' + args.model)
  model = MODEL_MODULE.create_model()
  _, remaining_args = model.parse_args(remaining_args)
  model.prepare(is_training=False, scales=scale_list, global_step=args.restore_global_step)

  # check remaining args
  if (len(remaining_args) > 0):
    print('WARNING: found unhandled arguments: %s' % (remaining_args))

  # model > restore
  model.restore(ckpt_path=args.restore_path, target=args.restore_target)
  print('restored the model')

  # validate
  print('begin validation')
  num_images = dataloader.get_num_images()
  average_psnr_dict = {}
  for scale in scale_list:
    psnr_list = []

    for image_index in range(num_images):
      input_image, truth_image, image_name = dataloader.get_image_pair(image_index=image_index, scale=scale)

      if (args.chop_forward):
        output_image = image_utils.upscale_with_chop_forward(model=model, input_image=input_image, scale=scale, overlap_size=args.chop_overlap_size)
      else:
        output_image = model.upscale(input_list=[input_image], scale=scale)[0]

      if (args.save_path is not None):
        os.makedirs(os.path.join(args.save_path, 'x%d' % (scale)), exist_ok=True)
        output_image_path = os.path.join(args.save_path, 'x%d' % (scale), image_name+'.png')
        _save_image(output_image, output_image_path)

      truth_image = _clip_image(truth_image)
      output_image = _clip_image(output_image)

      truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)

      psnr = _image_psnr(output_image=output_image, truth_image=truth_image)

      psnr_list.append(psnr)
      print('x%d, %d/%d, psnr=%.2f' % (scale, image_index+1, num_images, psnr))

    average_psnr = np.mean(psnr_list)
    average_psnr_dict[scale] = average_psnr
    print('x%d, average psnr=%.2f' % (scale, average_psnr))

    
  # finalize
  print('finished')


if __name__ == '__main__':
  main()