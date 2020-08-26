import argparse
import copy
import importlib
import json
import os
import time
import torch

import models
# from utils import image_utils
from utils import image_utils_chop1_se
from numpy import zeros, newaxis

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
  parser.add_argument('--self_ensemble', action='store_true', help='Employ self_ensemble for improving quality.')
  parser.add_argument('--chop_overlap_size', type=int, default=20, help='The overlapping size for the chop-forward process. Should be even.')

  parser.add_argument('--isHsv', action='store_true',
                      help='Convert color space to hsv.')
  args, remaining_args = parser.parse_known_args()


  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # scale_list = [args.scale]
  scale_list = [2, 3, 4]
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
  duration_list = []

  for image_index, image_name in enumerate(image_name_list):
    image_input_path = os.path.join(args.input_path, image_name)
    image_output_path = os.path.join(args.output_path, os.path.splitext(image_name)[0]+'.png')

    input_image = cv.imread(image_input_path)
    if(args.isHsv):
      input_image = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)
    else:
      input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

    input_image = np.transpose(input_image, [2, 0, 1])

    start_time = time.perf_counter()
    if (args.chop_forward):
      if (args.self_ensemble):
          with torch.no_grad():
            tmp_img = torch.zeros([1, 64, input_image.shape[1], input_image.shape[2]])
            for i in range(2):
              if i == 0:
                flip_img = input_image[:, :, ::-1]
                for j in range(4):
                  rot_flip_img = np.rot90(flip_img, j, axes=(1, 2))
                  rot_flip_img = torch.from_numpy(np.flip(rot_flip_img,axis=0).copy())
                  out_img = model.feature(rot_flip_img.unsqueeze(0), args.scale)
                  tmp = torch.rot90(out_img, 4 - j, dims=(2, 3))
                  tmp = tmp.numpy()
                  tmp = np.array(tmp[:,:,::-1])
                  tmp = torch.as_tensor(tmp, dtype=torch.float32)
                  # print(tmp_img.shape)
                  # print(tmp.shape)
                  tmp_img += tmp
              else:
                for k in range(4):
                  rot_img = np.rot90(input_image, k, axes=(1, 2))
                  rot_img = torch.from_numpy(np.flip(rot_img,axis=0).copy())
                  out_img = model.feature(rot_img.unsqueeze(0), args.scale)
                  tmp_img += torch.rot90(out_img, 4 - k, dims=(2, 3))
            output_tensor = tmp_img / 8
            # output_tensor = output_tensor.unsqueeze(0)
            output_image = model.upscale(output_tensor, scale=args.scale)[0]

      else:
        with torch.no_grad():
          output_image = image_utils_chop1_se.extract_with_chop_forward(model=model, input_image=input_image, scale=args.scale, overlap_size=args.chop_overlap_size)
          output_image = model.upscale(output_image, scale=args.scale)

    else:
      output_image = model.upscale(input_list=[input_image], scale=args.scale)[0]
    end_time = time.perf_counter()

    duration = end_time - start_time
    duration_list.append(duration)
    

    if(args.isHsv):
      output_image_h = output_image[0, :, :]
      output_image_h = np.clip(output_image_h, a_min=0, a_max=179)
      output_image_h = output_image_h[newaxis, :, :]
      output_image_sv = output_image[1:, :, :]
      output_image_sv = np.clip(output_image_sv, a_min=0, a_max=255)

      output_image = np.concatenate([output_image_h, output_image_sv], axis = 0)
    else:
      output_image = np.clip(output_image, a_min=0, a_max=255)


    output_image = np.round(output_image).astype(np.uint8)
    output_image = np.transpose(output_image, [1, 2, 0])
    if(args.isHsv):
      output_image = cv.cvtColor(output_image, cv.COLOR_HSV2BGR)
    else:
      output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
    cv.imwrite(image_output_path, output_image)
    torch.cuda.empty_cache()

    print('%d/%d, %s, duration: %.4fs' % (image_index+1, num_images, image_name, duration))

    
  # finalize
  print('finished')
  print('- average duration: %.4fs' % (np.mean(duration_list)))



if __name__ == '__main__':
  main()