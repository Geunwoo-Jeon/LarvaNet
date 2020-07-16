import argparse
import copy
import importlib
import json
import os
import time

import models
from utils import torchsummaryX

import numpy as np
import torch


def main():
  # parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--model', type=str, default='edsr', help='Name of the model.')
  parser.add_argument('--scale', type=int, default=4, help='Scale of the input images.')
  parser.add_argument('--scales', type=str, default='4', help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')

  parser.add_argument('--input_width', type=int, default=224, help='Width of a dummy input.')
  parser.add_argument('--input_height', type=int, default=224, help='Height of a dummy input.')

  args, remaining_args = parser.parse_known_args()


  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  scale_list = list(map(lambda x: int(x), args.scales.split(',')))

  # model
  print('prepare model - %s' % (args.model))
  MODEL_MODULE = importlib.import_module('models.' + args.model)
  model = MODEL_MODULE.create_model()
  _, remaining_args = model.parse_args(remaining_args)
  model.prepare(is_training=False, scales=scale_list)

  # check remaining args
  if (len(remaining_args) > 0):
    print('WARNING: found unhandled arguments: %s' % (remaining_args))
  
  # get summary
  torchsummaryX.summary(model.get_model(), torch.zeros(1, 3, args.input_height, args.input_width))
    
  # finalize
  print('finished')



if __name__ == '__main__':
  main()