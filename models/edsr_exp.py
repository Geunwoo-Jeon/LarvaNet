import argparse
import copy
import math
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from models.base import BaseModel
import torch.nn.functional as F


def create_model():
  return EDSR()


class EDSR(BaseModel):
  def __init__(self):
    super().__init__()

  def parse_args(self, args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--edsr_train_patch_size', type=int, default=48,
                        help='Size of the input patch during training.')
    parser.add_argument('--edsr_conv_features', type=int, default=64, help='The number of convolutional features.')
    parser.add_argument('--edsr_res_blocks', type=int, default=8, help='The number of residual blocks.')
    parser.add_argument('--edsr_res_weight', type=float, default=1.0, help='The scaling factor.')

    parser.add_argument('--edsr_learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--edsr_learning_rate_decay', type=float, default=0.5, help='Learning rate decay factor.')
    parser.add_argument('--edsr_learning_rate_decay_steps', type=int, default=200000,
                        help='The number of training steps to perform learning rate decay.')

    self.args, remaining_args = parser.parse_known_args(args=args)
    return copy.deepcopy(self.args), remaining_args

  def prepare(self, is_training, scales, global_step=0):
    # config. parameters
    self.global_step = global_step

    self.scale_list = scales
    for scale in self.scale_list:
      if (not scale in [2, 3, 4]):
        raise ValueError('Unsupported scale is provided.')
    if len(self.scale_list) != 1:
      raise ValueError('Only one scale should be provided.')
    self.scale = self.scale_list[0]

    # configure device
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PyTorch model
    self.model = EDSRModule(args=self.args, scale=self.scale)
    if (is_training):
      self.optim = optim.Adam(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=self._get_learning_rate()
      )
      self.loss_fn_l1 = nn.L1Loss()
      self.loss_fn_ms_ssim = Loss(device=self.device, width=self.args.edsr_train_patch_size * self.scale)

    # configure device
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = self.model.to(self.device)

  def save(self, base_path):
    save_path = os.path.join(base_path, 'model_%d.pth' % (self.global_step))
    torch.save(self.model.state_dict(), save_path)

  def restore(self, ckpt_path, target=None):
    self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

  def get_next_train_scale(self):
    scale = self.scale_list[np.random.randint(len(self.scale_list))]
    return scale

  def train_step(self, input_list, scale, truth_list, summary=None):
    # numpy to torch
    input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)
    truth_tensor = torch.tensor(truth_list, dtype=torch.float32, device=self.device)
    truth_bicubic_tensor = F.interpolate(truth_tensor, scale_factor=1/scale, mode='bicubic', align_corners=False)

    # get SR and calculate loss
    output_tensor_1, output_tensor_2 = self.model(input_tensor)
    loss_1 = self.loss_fn_l1(output_tensor_1, truth_bicubic_tensor)
    loss_2 = self.loss_fn_ms_ssim(output_tensor_2, truth_tensor)
    loss = 0.5*(loss_1+loss_2)

    # adjust learning rate
    lr = self._get_learning_rate()
    for param_group in self.optim.param_groups:
      param_group['lr'] = lr

    # do back propagation
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

    # finalize
    self.global_step += 1

    # write summary
    if (summary is not None):
      summary.add_scalar('loss', loss, self.global_step)
      summary.add_scalar('lr', lr, self.global_step)

      output_tensor_uint8_1 = output_tensor_1.clamp(0, 255).byte()
      output_tensor_uint8_2 = output_tensor_2.clamp(0, 255).byte()
      for i in range(min(4, len(input_list))):
        summary.add_image('input/%d' % i, input_list[i], self.global_step)
        summary.add_image('output_1/%d' % i, output_tensor_uint8_1[i, :, :, :], self.global_step)
        summary.add_image('output_2/%d' % i, output_tensor_uint8_2[i, :, :, :], self.global_step)
        summary.add_image('truth/%d' % i, truth_list[i], self.global_step)

    return loss.item()

  def upscale(self, input_list, scale):
    # numpy to torch
    input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)

    # get SR
    output_tensor_1, output_tensor_2 = self.model(input_tensor)

    # finalize
    return output_tensor_2.detach().cpu().numpy()

  def _get_learning_rate(self):
    return self.args.edsr_learning_rate * (
              self.args.edsr_learning_rate_decay ** (self.global_step // self.args.edsr_learning_rate_decay_steps))

class Loss(nn.Module):
  def __init__(self, device, width):
    super(Loss, self).__init__()

    self.device = device
    self.sigma = 8

    gaussian = np.exp(-1.*np.arange(-(width/2), width/2)**2/(2*self.sigma**2))
    gaussian = np.outer(gaussian, gaussian.reshape((width, 1)))	# extend to 2D
    gaussian = gaussian/np.sum(gaussian)								# normailization
    gaussian = np.reshape(gaussian, (1, 1, width, width)) 	# reshape to 4D
    self.gaussian = torch.FloatTensor(gaussian).to(self.device)

  def forward(self, output_tensor, truth_tensor):
    ms_ssim_loss = 1 - ms_ssim(output_tensor, truth_tensor, data_range=255, size_average=True)

    l1_loss = torch.abs(truth_tensor - output_tensor)
    l1_loss_filtered = torch.mul(l1_loss, self.gaussian)
    l1_loss_filtered = F.conv2d(input=l1_loss, weight=self.gaussian, groups=3)

    return 0.16*torch.mean(ms_ssim_loss)+0.84*l1_loss_filtered

class MeanShift(nn.Conv2d):
  def __init__(self, rgb_mean, sign):
    super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
    self.weight_data = torch.eye(3).view(3, 3, 1, 1)
    self.bias_data = sign * torch.Tensor(rgb_mean)

    for params in self.parameters():
      params.requires_grad = False


class ResidualBlock(nn.Module):
  def __init__(self, num_channels, weight=1.0):
    super(ResidualBlock, self).__init__()

    self.body = nn.Sequential(
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
    )
    self.weight = weight

  def forward(self, x):
    res = self.body(x).mul(self.weight)
    output = torch.add(x, res)
    return output


class UpsampleBlock(nn.Module):
  def __init__(self, num_channels, scale):
    super(UpsampleBlock, self).__init__()

    layers = []
    if scale == 2 or scale == 4 or scale == 8:
      for _ in range(int(math.log(scale, 2))):
        layers.append(
          nn.Conv2d(in_channels=num_channels, out_channels=4 * num_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PixelShuffle(2))
    elif scale == 3:
      layers.append(
        nn.Conv2d(in_channels=num_channels, out_channels=9 * num_channels, kernel_size=3, stride=1, padding=1))
      layers.append(nn.PixelShuffle(3))

    self.body = nn.Sequential(*layers)

  def forward(self, x):
    output = self.body(x)
    return output


class EDSRModule(nn.Module):
  def __init__(self, args, scale):
    super(EDSRModule, self).__init__()

    self.mean_shift = MeanShift([114.4, 111.5, 103.0], sign=1.0)
    self.first_conv = nn.Conv2d(in_channels=3, out_channels=args.edsr_conv_features, kernel_size=3, stride=1, padding=1)

    res_block_layers_1 = []
    res_block_layers_2 = []
    for i in range(args.edsr_res_blocks):
      res_block_layers_1.append(ResidualBlock(num_channels=args.edsr_conv_features, weight=args.edsr_res_weight))
      res_block_layers_2.append(ResidualBlock(num_channels=args.edsr_conv_features, weight=args.edsr_res_weight))

    self.res_blocks1 = nn.Sequential(*res_block_layers_1)
    self.res_blocks2 = nn.Sequential(*res_block_layers_2)
    self.after_res_conv = nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=args.edsr_conv_features,
                                    kernel_size=3, stride=1, padding=1)

    self.upsample = UpsampleBlock(num_channels=args.edsr_conv_features, scale=scale)
    self.final_conv1 = nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.final_conv2 = nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=3, kernel_size=3, stride=1, padding=1)

    self.mean_inverse_shift = MeanShift([114.4, 111.5, 103.0], sign=-1.0)

  def forward(self, x):
      x = self.mean_shift(x)
      x = self.first_conv(x)
      tmp = self.res_blocks1(x)

      #route 1
      res1 = torch.add(x, tmp)
      res1 = self.final_conv1(res1)
      res1 = self.mean_inverse_shift(res1)

      #route 2
      res2 = self.res_blocks2(tmp)
      res2 = self.after_res_conv(res2)
      res2 = torch.add(x, res2)
      res2 = self.upsample(res2)
      res2 = self.final_conv2(res2)
      res2 = self.mean_inverse_shift(res2)

      return res1, res2

