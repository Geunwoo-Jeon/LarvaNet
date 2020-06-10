import argparse
import copy
import math
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.base import BaseModel

def create_model():
  return MAMNet()

class MAMNet(BaseModel):
  def __init__(self):
    super().__init__()

  def parse_args(self, args):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mamnet_conv_features', type=int, default=64, help='The number of convolutional features.')
    parser.add_argument('--mamnet_res_blocks', type=int, default=16, help='The number of residual blocks.')
    parser.add_argument('--mamnet_res_weight', type=float, default=1.0, help='The scaling factor.')

    parser.add_argument('--mamnet_learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--mamnet_learning_rate_decay', type=float, default=0.5, help='Learning rate decay factor.')
    parser.add_argument('--mamnet_learning_rate_decay_steps', type=int, default=200000, help='The number of training steps to perform learning rate decay.')

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
    self.model = MAMNetModule(args=self.args, scale=self.scale)
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
    return self.args.mamnet_learning_rate * (self.args.mamnet_learning_rate_decay ** (self.global_step // self.args.mamnet_learning_rate_decay_steps))
  


class MeanShift(nn.Conv2d):
  def __init__(self, rgb_mean, sign):
    super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
    self.weight_data = torch.eye(3).view(3, 3, 1, 1)
    self.bias_data = sign * torch.Tensor(rgb_mean)

    for params in self.parameters():
      params.requires_grad = False


class MAMBlock(nn.Module):
  def __init__(self, num_channels, weight=1.0):
    super(MAMBlock, self).__init__()

    self.body = nn.Sequential(
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
      MAMLayer(num_channels=num_channels, reduction=16)
    )
    self.weight = weight
  
  def forward(self, x):
    res = self.body(x).mul(self.weight)
    output = torch.add(x, res)
    return output

class MAMLayer(nn.Module):
  def __init__(self, num_channels, reduction=16):
    super(MAMLayer, self).__init__()
    self.modulation_map_CSI = 0.0
    self.modulation_map_ICD = 0.0
    self.modulation_map_CSD = 0.0
    self.conv_du = nn.Sequential(
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels//reduction, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=num_channels//reduction, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
    ) 
    self.depthwise_conv2d = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, groups=num_channels)
    self.scaling = nn.Sigmoid()

  def forward(self, x):
    N, C, W, H = x.size()
    tmp_var = x.view(N, C, -1).var(dim=2, keepdim=True)
    self.modulation_map_CSI = tmp_var.unsqueeze(-1).repeat(1, 1, W, H)
    self.modulation_map_ICD = self.conv_du(tmp_var.unsqueeze(-1)).repeat(1, 1, W, H)
    self.modulation_map_CSD = self.depthwise_conv2d(x)
    y = self.scaling(self.modulation_map_CSI + self.modulation_map_ICD + self.modulation_map_CSD)
    return x * y

class UpsampleBlock(nn.Module):
  def __init__(self, num_channels, scale):
    super(UpsampleBlock, self).__init__()

    layers = []
    if scale == 2 or scale == 4 or scale == 8:
      for _ in range(int(math.log(scale, 2))):
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=4*num_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PixelShuffle(2))
    elif scale == 3:
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=9*num_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.PixelShuffle(3))

    self.body = nn.Sequential(*layers)
  
  def forward(self, x):
    output = self.body(x)
    return output



class MAMNetModule(nn.Module):
  def __init__(self, args, scale):
    super(MAMNetModule, self).__init__()

    self.mean_shift = MeanShift([114.4, 111.5, 103.0], sign=1.0)
    self.first_conv = nn.Conv2d(in_channels=3, out_channels=args.mamnet_conv_features, kernel_size=3, stride=1, padding=1)
    
    res_block_layers = []
    for i in range(args.mamnet_res_blocks):
      res_block_layers.append(MAMBlock(num_channels=args.mamnet_conv_features, weight=args.mamnet_res_weight))
    self.res_blocks = nn.Sequential(*res_block_layers)
    self.after_res_conv = nn.Conv2d(in_channels=args.mamnet_conv_features, out_channels=args.mamnet_conv_features, kernel_size=3, stride=1, padding=1)

    self.upsample = UpsampleBlock(num_channels=args.mamnet_conv_features, scale=scale)
    self.final_conv = nn.Conv2d(in_channels=args.mamnet_conv_features, out_channels=3, kernel_size=3, stride=1, padding=1)

    self.mean_inverse_shift = MeanShift([114.4, 111.5, 103.0], sign=-1.0)
  
  def forward(self, x):
    x = self.mean_shift(x)
    x = self.first_conv(x)

    res = self.res_blocks(x)
    res = self.after_res_conv(res)
    x = torch.add(x, res)

    x = self.upsample(x)
    x = self.final_conv(x)
    x = self.mean_inverse_shift(x)

    return x
    

