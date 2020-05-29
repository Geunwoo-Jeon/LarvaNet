import argparse
import copy
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

  parser.add_argument('--chop_forward', action='store_true', help='Employ chop-forward to reduce the memory usage.')
  parser.add_argument('--chop_overlap_size', type=int, default=20, help='The overlapping size for the chop-forward process. Should be even.')

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

    input_split_images = _split_image(input_image, chop=args.chop_forward, overlap_size=args.chop_overlap_size)
    output_split_images = []
    for input_split in input_split_images:
      output_split = model.upscale(input_list=[input_split], scale=args.scale)[0]
      output_split_images.append(output_split)
    
    output_image = _combine_images(output_split_images, input_image=input_image, scale=args.scale, chop=args.chop_forward, overlap_size=args.chop_overlap_size)
    
    output_image = np.clip(output_image, a_min=0, a_max=255)
    output_image = np.round(output_image).astype(np.uint8)
    output_image = np.transpose(output_image, [1, 2, 0])
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
    cv.imwrite(image_output_path, output_image)

    
  # finalize
  print('finished')


def _split_image(image, chop, overlap_size):
  if (not chop):
    return [image]
  
  _, height, width = image.shape
  split_height = height // 2
  split_width = width // 2
  half_overlap_size = overlap_size // 2

  images = []
  images.append(copy.deepcopy(image[:, :(split_height+half_overlap_size), :(split_width+half_overlap_size)]))
  images.append(copy.deepcopy(image[:, :(split_height+half_overlap_size), (split_width-half_overlap_size):]))
  images.append(copy.deepcopy(image[:, (split_height-half_overlap_size):, :(split_width+half_overlap_size)]))
  images.append(copy.deepcopy(image[:, (split_height-half_overlap_size):, (split_width-half_overlap_size):]))
  
  return images

def _combine_images(images, input_image, scale, chop, overlap_size):
  if (len(images) == 1):
    return images[0]
  
  _, height, width = input_image.shape
  split_height = height // 2
  split_width = width // 2
  new_height = height * scale
  new_width = width * scale
  new_split_height = split_height * scale
  new_split_width = split_width * scale
  new_half_overlap_size = (overlap_size // 2) * scale

  output_image = np.zeros([3, new_height, new_width])
  output_image[:, :new_split_height, :new_split_width] = images[0][:, :new_split_height, :new_split_width]
  output_image[:, :new_split_height, new_split_width:] = images[1][:, :new_split_height, new_half_overlap_size:]
  output_image[:, new_split_height:, :new_split_width] = images[2][:, new_half_overlap_size:, :new_split_width]
  output_image[:, new_split_height:, new_split_width:] = images[3][:, new_half_overlap_size:, new_half_overlap_size:]

  return output_image



if __name__ == '__main__':
  main()