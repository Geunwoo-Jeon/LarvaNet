import argparse
import importlib
import json
import os
import time

import models

import numpy as np
import cv2 as cv


def main():
  # parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--model', type=str, default='edsr', help='Name of the model.')

  parser.add_argument('--scale', type=int, default=4, help='Scale of the input images.')
  parser.add_argument('--cuda_device', type=str, default='-1', help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  parser.add_argument('--restore_path', type=str, required=True, help='Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
  parser.add_argument('--restore_target', type=str, help='Target of the restoration.')
  parser.add_argument('--restore_global_step', type=int, default=0, help='Global step of the restored model. Some models may require to specify this.')

  parser.add_argument('--input_path', type=str, default='LR', help='Base path of the input images.')
  parser.add_argument('--output_path', type=str, default='SR', help='Base path of the output images.')

  args, remaining_args = parser.parse_known_args()


  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
  scale_list = [args.scale]
  os.makedirs(args.output_path, exist_ok=True)

  # retrieve image name list
  image_name_list = [f for f in os.listdir(args.input_path) if f.lower().endswith('.png')]
  print('data: %d images are prepared' % (len(image_name_list)))

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


  # get outputs
  print('begin super-resolution')
  num_images = len(image_name_list)
  for image_index, image_name in enumerate(image_name_list):
    print('%d/%d, %s' % (image_index+1, num_images, image_name))

    image_input_path = os.path.join(args.input_path, image_name)
    image_output_path = os.path.join(args.output_path, os.path.splitext(image_name)[0]+'.png')

    input_image = cv.imread(image_input_path)
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = np.transpose(input_image, [2, 0, 1])

    output_image = model.upscale(input_list=[input_image], scale=args.scale)[0]
    output_image = np.clip(output_image, a_min=0, a_max=255)
    output_image = np.round(output_image).astype(np.uint8)
    output_image = np.transpose(output_image, [1, 2, 0])
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
    cv.imwrite(image_output_path, output_image)

    
  # finalize
  print('finished')


if __name__ == '__main__':
  main()