import argparse
import importlib
import os
import time

import torch

import numpy as np


def _image_to_uint8(image):
  return np.clip(np.round(image), a_min=0, a_max=255).astype(np.uint8)

def _fit_truth_image_size(output_image, truth_image):
  return truth_image[:, 0:output_image.shape[1], 0:output_image.shape[2]]


def main():
  # parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataloader', type=str, default='div2k_val_loader', help='Name of the data loader.')
  parser.add_argument('--model', type=str, default='edsr', help='Name of the model.')

  parser.add_argument('--scales', type=str, default='4', help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
  parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

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
  model.prepare(is_training=False, scales=scale_list)

  # check remaining args
  if (len(remaining_args) > 0):
    print('WARNING: found unhandled arguments: %s' % (remaining_args))

  # runtime check
  print('begin runtime check')
  num_images = dataloader.get_num_images()
  for scale in scale_list:
    runtime_list = []

    for image_index in range(num_images):
      start_time = time.perf_counter()
      input_image, truth_image, image_name = dataloader.get_image_pair(image_index=image_index, scale=scale)
      im_load_time = time.perf_counter() - start_time

      start_time = time.perf_counter()
      input_tensor = torch.tensor([input_image], dtype=torch.float32, device='cuda')
      im_tf_time = time.perf_counter() - start_time

      start_time = time.perf_counter()
      output_tenser = model.fwd_runtime(input_tensor=input_tensor)
      runtime = time.perf_counter() - start_time

      runtime_list.append(runtime)
      print(f'{image_index+1}/{num_images}, image load time={im_load_time:.4f},'
            f'image transform time={im_tf_time:.4f}, runtime={runtime:.4f}')
    average_runtime = np.mean(runtime_list)
    print(f'runtime={average_runtime:.4f}')

    
  # finalize
  print('finished')


if __name__ == '__main__':
  main()