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
    return EBRN()


class EBRN(BaseModel):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_filters', type=int, default=64, help='The number of filters.')
        parser.add_argument('--num_brms', type=int, default=10, help='The number of modules.')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
        parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='Learning rate decay factor.')
        parser.add_argument('--learning_rate_decay_steps', type=int, default=200000,
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

        # PyTorch model
        self.model = EBRNModule(args=self.args, scale=self.scale)
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
        return self.args.learning_rate * (
                    self.args.learning_rate_decay ** (self.global_step // self.args.learning_rate_decay_steps))


class BRM(nn.Module):
    def __init__(self, num_channels, scale, back_project=True):
        super(BRM, self).__init__()
        self.back_project = back_project

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.body(x)
        output = torch.add(x, res)
        if self.back_project:
            return res, output
        return output

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
        self.weight_data = torch.eye(3).view(3, 3, 1, 1)
        self.bias_data = sign * torch.Tensor(rgb_mean)

        for params in self.parameters():
            params.requires_grad = False


class UpsampleBlock(nn.Module):
  def __init__(self, num_channels, out_channels, scale):
    super(UpsampleBlock, self).__init__()

    layers = []
    layers.append(nn.Conv2d(in_channels=num_channels, out_channels=out_channels*(scale**2), kernel_size=3, stride=1, padding=1))
    layers.append(nn.PixelShuffle(scale))
    self.body = nn.Sequential(*layers)
  
  def forward(self, x):
    output = self.body(x)
    return output


class EBRNModule(nn.Module):
    def __init__(self, args, scale):
        super(EBRNModule, self).__init__()
        num_filters = args.num_filters
        kernel_size = 3
        stride = 1
        padding = 1
        self.num_brms = args.num_brms

        self.mean_shift = MeanShift([114.4, 111.5, 103.0], sign=1.0)
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding)

        self.brms = nn.ModuleList()
        for _ in range(self.num_brms - 1):
            self.brms.append(BRM(num_channels=num_filters, scale=scale, back_project=True))
        self.brms.append(BRM(num_channels=num_filters, scale=scale, back_project=False))

        self.fusion_layers = nn.ModuleList()
        for _ in range(self.num_brms - 1):
            self.fusion_layers.append(
                nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride,
                          padding=padding))

        self.upsample = UpsampleBlock(num_channels=num_filters * self.num_brms, out_channels=3, scale=scale)
        self.mean_inverse_shift = MeanShift([114.4, 111.5, 103.0], sign=-1.0)

    def forward(self, x):
        fea = self.first_conv(x)

        out_list = []
        out_prime_list = []

        for i in range(self.num_brms - 1):
            fea, out = self.brms[i](fea)
            out_list.append(out)
        out = self.brms[-1](fea)
        out_prime_list.append(out)

        for i in range(self.num_brms - 1):
            out_prime = self.fusion_layers[i](out + out_list[-(i + 1)])
            out_prime_list.append(out_prime)

        sr = self.upsample(torch.cat(out_prime_list, dim=1))
        sr = sr + F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return sr
