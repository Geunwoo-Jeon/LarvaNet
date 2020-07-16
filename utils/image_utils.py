import copy

import numpy as np
import cv2 as cv


def upscale_with_chop_forward(model, input_image, scale, overlap_size):
  """
  Get an upscaled image with employing chopping forward.
  Args:
    model: Target model object.
    input_image: The input image.
    scale: Scale to be super-resolved.
    summary: Summary writer to write the current training state. Can be None to skip writing for current training step.
  Returns:
    output_image: The output image.
  """

  input_split_images = _split_image(input_image, chop=True, overlap_size=overlap_size)
  output_split_images = []

  for input_split in input_split_images:
    input_split_2_images = _split_image(input_split, chop=True, overlap_size=overlap_size)
    output_split_2_images = []
    for input_split_2 in input_split_2_images:
      input_split_3_images = _split_image(input_split_2, chop=True, overlap_size=overlap_size)
      output_split_3_images = []
      for input_split_3 in input_split_3_images:
        output_split_3 = model.upscale([input_split_3], scale=scale)[0]
        output_split_3_images.append(output_split_3)
      output_image_tmp_3 = _combine_images(output_split_3_images, input_image=input_split_2, scale=scale, chop=True, overlap_size=overlap_size)
      output_split_2_images.append(output_image_tmp_3)
    output_image_tmp = _combine_images(output_split_2_images, input_image=input_split, scale=scale, chop=True, overlap_size=overlap_size)
    # output_split = model.upscale(input_list=[input_split], scale=scale)[0]
    output_split_images.append(output_image_tmp)
  
  output_image = _combine_images(output_split_images, input_image=input_image, scale=scale, chop=True, overlap_size=overlap_size)

  return output_image


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