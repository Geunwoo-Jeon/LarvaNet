import copy

import numpy as np
import torch
import cv2 as cv


def extract_with_chop_forward(model, input_image, scale, overlap_size):
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
    output_split_images_upscaled = []
    for input_split in input_split_images:
        output_split = model.feature(input_list=[input_split], scale=scale)[0]
        output_split_images.append(output_split)

        for image in output_split_images:
            image = torch.as_tensor(image, dtype=torch.float32)
            output_split_images_upscaled.append(model.upscale(image, scale=scale)[0])

    output_image = _combine_images(output_split_images_upscaled, input_image=input_image, scale=scale, chop=True,
                                   overlap_size=overlap_size)
    output_image = torch.as_tensor(output_image, dtype=torch.float32)
    return output_image


def _split_image(image, chop, overlap_size):
    if (not chop):
        return [image]

    _, height, width = image.shape
    split_height = height // 2
    split_width = width // 2
    half_overlap_size = overlap_size // 2

    images = []
    images.append(copy.deepcopy(image[:, :(split_height + half_overlap_size), :(split_width + half_overlap_size)]))
    images.append(copy.deepcopy(image[:, :(split_height + half_overlap_size), (split_width - half_overlap_size):]))
    images.append(copy.deepcopy(image[:, (split_height - half_overlap_size):, :(split_width + half_overlap_size)]))
    images.append(copy.deepcopy(image[:, (split_height - half_overlap_size):, (split_width - half_overlap_size):]))

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

    output_image = np.zeros([images[0].shape[0], new_height, new_width])
    output_image[:, :new_split_height, :new_split_width] = images[0][:, :new_split_height, :new_split_width]
    output_image[:, :new_split_height, new_split_width:] = images[1][:, :new_split_height, new_half_overlap_size:]
    output_image[:, new_split_height:, :new_split_width] = images[2][:, new_half_overlap_size:, :new_split_width]
    output_image[:, new_split_height:, new_split_width:] = images[3][:, new_half_overlap_size:, new_half_overlap_size:]

    return output_image