import argparse
import copy
import os

import numpy as np
import cv2 as cv

from .base import BaseLoader

# DIV2K dataset loader

def create_loader():
  return DIV2KLoader()

class DIV2KLoader(BaseLoader):
  def __init__(self):
    super().__init__()

  
  def parse_args(self, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input_path', type=str, default='c:/aim2020/data/DIV2K_train_LR_bicubic', help='Base path of the input images. For example, if you specify this argument to \'LR\', the downscaled images by a factor of 4 should be in \'LR/X4/\'.')
    parser.add_argument('--data_truth_path', type=str, default='c:/aim2020/data/DIV2K_train_HR', help='Base path of the ground-truth images.')
    parser.add_argument('--data_cached', action='store_true', help='If true, cache the data on the memory.')

    self.args, remaining_args = parser.parse_known_args(args=args)
    return copy.deepcopy(self.args), remaining_args


  def prepare(self, scales):
    self.scale_list = scales

    # retrieve image name list
    input_path = os.path.join(self.args.data_truth_path)
    self.image_name_list = [os.path.splitext(f)[0] for f in os.listdir(input_path) if f.lower().endswith('.png')]
    print('data: %d images are prepared (%s)' % (len(self.image_name_list), 'caching enabled' if self.args.data_cached else 'caching disabled'))
    
    # initialize cached list
    self.cached_input_image_list = {}
    for scale in self.scale_list:
      self.cached_input_image_list[scale] = {}
    self.cached_truth_image_list = {}
  

  def get_num_images(self):
    return len(self.image_name_list)
  
  
  def get_patch_batch(self, batch_size, scale, input_patch_size):
    input_list = []
    truth_list = []

    for _ in range(batch_size):
      input_patch, truth_patch = self.get_random_image_patch_pair(scale=scale, input_patch_size=input_patch_size)
      input_list.append(input_patch)
      truth_list.append(truth_patch)
    
    return input_list, truth_list
  

  def get_random_image_patch_pair(self, scale, input_patch_size):
    # select an image
    image_index = np.random.randint(self.get_num_images())

    # retrieve image
    input_patch, truth_patch = self.get_image_patch_pair(image_index=image_index, scale=scale, input_patch_size=input_patch_size)
    
    # finalize
    return input_patch, truth_patch


  def get_image_patch_pair(self, image_index, scale, input_patch_size):
    # retrieve image
    input_image, truth_image, _ = self.get_image_pair(image_index=image_index, scale=scale)

    # randomly crop
    truth_patch_size = input_patch_size * scale
    _, height, width = input_image.shape
    input_x = np.random.randint(width - input_patch_size)
    input_y = np.random.randint(height - input_patch_size)
    truth_x = input_x * scale
    truth_y = input_y * scale
    input_patch = input_image[:, input_y:(input_y+input_patch_size), input_x:(input_x+input_patch_size)]
    truth_patch = truth_image[:, truth_y:(truth_y+truth_patch_size), truth_x:(truth_x+truth_patch_size)]

    # randomly rotate
    rot90_k = np.random.randint(4)+1
    input_patch = np.rot90(input_patch, k=rot90_k, axes=(1, 2))
    truth_patch = np.rot90(truth_patch, k=rot90_k, axes=(1, 2))

    # randomly flip
    flip = (np.random.uniform() < 0.5)
    if (flip):
      input_patch = input_patch[:, :, ::-1]
      truth_patch = truth_patch[:, :, ::-1]
    
    # finalize
    return input_patch, truth_patch
  

  def get_image_pair(self, image_index, scale):
    # retrieve image
    image_name = self.image_name_list[image_index]
    input_image = self._get_input_image(scale, image_name)
    truth_image = self._get_truth_image(image_name)

    # finalize
    return input_image, truth_image, image_name


  def _get_input_image(self, scale, image_name):
    image = None
    has_cached = False
    if (self.args.data_cached):
      if (image_name in self.cached_input_image_list[scale]):
        image = self.cached_input_image_list[scale][image_name]
        has_cached = True
    
    if (image is None):
      image_path = os.path.join(self.args.data_input_path, ('X%d' % (scale)), ('%sx%d.png' % (image_name, scale)))
      image = self._load_image(image_path)
    
    if (self.args.data_cached and (not has_cached)):
      self.cached_input_image_list[scale][image_name] = image
    
    return image
  

  def _get_truth_image(self, image_name):
    image = None
    has_cached = False
    if (self.args.data_cached):
      if (image_name in self.cached_truth_image_list):
        image = self.cached_truth_image_list[image_name]
        has_cached = True
    
    if (image is None):
      image_path = os.path.join(self.args.data_truth_path, ('%s.png' % (image_name)))
      image = self._load_image(image_path)
    
    if (self.args.data_cached and (not has_cached)):
      self.cached_truth_image_list[image_name] = image
    
    return image
  

  def _load_image(self, path):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.transpose(image, [2, 0, 1])
    return image