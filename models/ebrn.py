import torch
import torch.nn as nn
import argparse
import copy
import math
import os

import numpy as np
from models.base import BaseModel
import torch.optim as optim

def create_model():
    return EBRN()

class EBRN(BaseModel):
  def __init__(self):
    super().__init__()

  def parse_args(self, args):
    parser = argparse.ArgumentParser()

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
    self.model = EBRNModule()
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

  def get_next_train_scale(self):
    scale = self.scale_list[np.random.randint(len(self.scale_list))]
    return scale

  def train_step(self, input_list, scale, truth_list, summary=None):
    # numpy to torch
    input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)
    truth_tensor = torch.tensor(truth_list, dtype=torch.float32, device=self.device)
    # print(input_tensor.shape)
    # print(truth_tensor.shape)

    # get SR and calculate loss
    output_tensor= self.model(input_tensor)
    # print(output_tensor.shape)
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
        summary.add_image('output_1/%d' % i, output_tensor_uint8[i, :, :, :], self.global_step)
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
    return self.args.edsr_learning_rate * (
              self.args.edsr_learning_rate_decay ** (self.global_step // self.args.edsr_learning_rate_decay_steps))

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BRM(nn.Module):
    def __init__(self, channels, scale, bp = True):
        super(BRM, self).__init__()
        self. bp = bp
        scale=1
        kernel_size, stride, padding = {
            1: (5, 1, 2),
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2)
        }[scale]
        # 先进行上采样
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size, stride=stride, padding=padding)
        # 上采样特征进行重建
        self.sr_flow = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels)
        ])
        # 再进行下采样
        self.down = nn.Conv2d(channels, channels, kernel_size, stride = stride, padding = padding)
        # 残差进行重建
        self.bp_flow = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
            nn.PReLU(channels)
        ])
    def forward(self, x):
        up = self.up(x)
        ox = self.sr_flow(up)
        # 最后一层并没有back-projection flow
        if self.bp:
            down = self.down(up)
            sub = x - down
            ix = self.bp_flow(sub)
            ix += sub
            return ix, ox
        return ox

class EBRNModule(nn.Module):
    def __init__(self):
        super(EBRNModule, self).__init__()
        self.n_brms = 10
        n_colors = 3
        n_feats = 64
        kernel_size = 3
        scale = 1

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # feature shallow extraction
        self.head = nn.Sequential(*[
            nn.Conv2d(n_colors, n_feats * 4, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats * 4),
            nn.Conv2d(n_feats * 4, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats)
        ])

        # convolutional layer after fusion
        self.convs = nn.ModuleList()
        for i in range(self.n_brms - 1):
            self.convs.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding = kernel_size//2))

        # embedded block residual learning
        self.brms = nn.ModuleList()
        for i in range(self.n_brms - 1):
            self.brms.append(BRM(n_feats, scale, True))
        self.brms.append(BRM(n_feats, scale, False))

        # reconstruction
        self.tail = nn.Conv2d(n_feats * self.n_brms, n_colors, kernel_size, padding = kernel_size//2)
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        out = []
        sr_sets = []
        # 前面的self.n_brms-1层
        for i in range(self.n_brms - 1):
            x, sr = self.brms[i](x)
            sr_sets.append(sr)
        # 最后的第n_brms层
        sr = self.brms[self.n_brms - 1](x)
        out.append(sr)

        for i in range(self.n_brms - 1):
            sr = sr + sr_sets[self.n_brms - i - 2]
            sr = self.convs[i](sr)
            out.append(sr)
        x = self.tail(torch.cat(out, dim = 1))
        return x

# from torchstat import stat
# net = EBRN()
# stat(net, (3, 10, 10))
# Total params: 7,632,143