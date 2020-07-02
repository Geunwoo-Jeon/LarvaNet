import argparse
import copy
import math
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.base import BaseModel


def create_model():
  return IMDN_MSRR()

class IMDN_MSRR(BaseModel):
  def __init__(self):
    super().__init__()

  def parse_args(self, args):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_filters', type=int, default=64, help='The number of filters.')
    parser.add_argument('--num_blocks', type=int, default=8, help='The number of modules.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='Learning rate decay factor.')
    parser.add_argument('--learning_rate_decay_steps', type=int, default=200000, help='The number of training steps to perform learning rate decay.')

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

    # PyTorch model
    self.model = IMDN_MSRR_Module(args=self.args, scale=self.scale)
    if (is_training):
      self.optim = optim.Adam(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=self._get_learning_rate()
      )
      self.loss_fn = nn.L1Loss()

    # configure device
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = self.model.to(self.device)
      
  def save(self, base_path):
    save_path = os.path.join(base_path, 'model_%d.pth' % (self.global_step))
    torch.save(self.model.state_dict(), save_path)

  def restore(self, ckpt_path, target=None):
    self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
  
  def get_model(self):
    return self.model
  
  def get_next_train_scale(self):
    scale = self.scale_list[np.random.randint(len(self.scale_list))]
    return scale

  def train_step(self, input_list, scale, truth_list, summary=None):
    # numpy to torch
    input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)
    truth_tensor = torch.tensor(truth_list, dtype=torch.float32, device=self.device)

    # get SR and calculate loss
    output_tensor = self.model(input_tensor)
    loss = self.loss_fn(output_tensor, truth_tensor)

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

      output_tensor_uint8 = output_tensor.clamp(0, 255).byte()
      for i in range(min(4, len(input_list))):
        summary.add_image('input/%d' % i, input_list[i], self.global_step)
        summary.add_image('output/%d' % i, output_tensor_uint8[i, :, :, :], self.global_step)
        summary.add_image('truth/%d' % i, truth_list[i], self.global_step)

    return loss.item()

  def upscale(self, input_list, scale):
    # numpy to torch
    input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)

    # get SR
    output_tensor = self.model(input_tensor)

    # finalize
    return output_tensor.detach().cpu().numpy()

  
  def _get_learning_rate(self):
    return self.args.learning_rate * (self.args.learning_rate_decay ** (self.global_step // self.args.learning_rate_decay_steps))
  


class MeanShift(nn.Conv2d):
  def __init__(self, rgb_mean, sign):
    super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
    self.weight_data = torch.eye(3).view(3, 3, 1, 1)
    self.bias_data = sign * torch.Tensor(rgb_mean)

    for params in self.parameters():
      params.requires_grad = False


class IMDBlock(nn.Module):
  def __init__(self, num_channels, distill_rate=0.25):
    super(IMDBlock, self).__init__()

    self.distilled_channels = int(num_channels * distill_rate)
    self.remaining_channels = int(num_channels - self.distilled_channels)

    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.05, inplace=True),
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(in_channels=self.remaining_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.05, inplace=True),
    ) 
    self.conv3 = nn.Sequential(
      nn.Conv2d(in_channels=self.remaining_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.05, inplace=True),
    ) 
    self.conv4 = nn.Sequential(
      nn.Conv2d(in_channels=self.remaining_channels, out_channels=self.distilled_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.05, inplace=True),
    )  
    self.conv5 = nn.Conv2d(in_channels=self.distilled_channels*4, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
  
  def forward(self, x):
    res1 = self.conv1(x)
    distilled_res1, remaining_res1 = torch.split(res1, (self.distilled_channels, self.remaining_channels), dim=1)
    res2 = self.conv2(remaining_res1)
    distilled_res2, remaining_res2 = torch.split(res2, (self.distilled_channels, self.remaining_channels), dim=1)
    res3 = self.conv3(remaining_res2)
    distilled_res3, remaining_res3 = torch.split(res3, (self.distilled_channels, self.remaining_channels), dim=1)
    res4 = self.conv4(remaining_res3)

    res = torch.cat([distilled_res1, distilled_res2, distilled_res3, res4], dim=1)
    res = self.conv5(res)

    output = torch.add(x, res)
    return output


class UpsampleBlock(nn.Module):
  def __init__(self, num_channels, out_channels, scale):
    super(UpsampleBlock, self).__init__()
  
    layers = []
    if scale == 2 or scale == 4 or scale == 8:
        for _ in range(int(math.log(scale, 2))):
            layers.append(
                nn.Conv2d(in_channels=num_channels, out_channels=4 * out_channels, kernel_size=3, stride=1,
                          padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
    elif scale == 3:
        layers.append(
            nn.Conv2d(in_channels=num_channels, out_channels=9 * out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PixelShuffle(3))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))


    self.body = nn.Sequential(
      nn.Conv2d(in_channels=num_channels, out_channels=4 * out_channels, kernel_size=3, stride=1, padding=1),
      nn.PixelShuffle(2),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),
      nn.Conv2d(in_channels=3, out_channels=4 * out_channels, kernel_size=3, stride=1, padding=1),
      nn.PixelShuffle(2),
      nn.LeakyReLU(negative_slope=0.1, inplace=True)
    ) 

  def forward(self, x):
    output = self.body(x)
    return output



class IMDN_MSRR_Module(nn.Module):
  def __init__(self, args, scale):
    super(IMDN_MSRR_Module, self).__init__()

    self.mean_shift = MeanShift([114.4, 111.5, 103.0], sign=1.0)
    self.first_conv = nn.Conv2d(in_channels=3, out_channels=args.num_filters, kernel_size=3, stride=1, padding=1)
    
    res_block_layers = []
    for i in range(args.num_blocks):
      res_block_layers.append(IMDBlock(num_channels=args.num_filters))
    self.res_blocks = nn.Sequential(*res_block_layers)
    self.after_res_conv = nn.Conv2d(in_channels=args.num_filters, out_channels=args.num_filters, kernel_size=3, stride=1, padding=1)

    self.upsample = UpsampleBlock(num_channels=args.num_filters, out_channels=3, scale=scale)

    self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    self.hr_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.hr_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.mean_inverse_shift = MeanShift([114.4, 111.5, 103.0], sign=-1.0)
  
  def forward(self, x):
    # tmp = self.mean_shift(x)
    tmp = self.first_conv(x)
    tmp = self.lrelu(tmp)

    res = self.res_blocks(tmp)
    res = self.after_res_conv(res)
    res = self.lrelu(res)
    tmp = torch.add(tmp, res)

    tmp = self.upsample(tmp)
    tmp = self.hr_conv2(self.lrelu(self.hr_conv1(tmp)))
    base = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
    tmp += base 
    # tmp = self.mean_inverse_shift(tmp)

    return tmp
    
