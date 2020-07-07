import argparse
import copy
import math
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

from models.base import BaseModel


def create_model():
    return MSRR()


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class MSRR(BaseModel):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_lr_blocks', type=int, default=4, help='The number of residual blocks at LR domain.')
        parser.add_argument('--num_hr_blocks', type=int, default=4, help='The number of residual blocks at HR domain.')
        parser.add_argument('--num_hr_filters', type=int, default=12, help='The number of filters at HR domain.')
        parser.add_argument('--hr_filter_size', type=int, default=3, help='The size of filters at HR domain.')
        parser.add_argument('--interpolate', type=str, default='bilinear', help='Interpolation method.')

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
        self.model = MSRRModule(args=self.args, scale=self.scale)
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
        return self.args.learning_rate * (self.args.learning_rate_decay ** (
                self.global_step // self.args.learning_rate_decay_steps))

    def fwd_runtime(self, input_tensor):
        output_tensor = self.model(input_tensor)
        return output_tensor


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, filter_size=3):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=filter_size, stride=1,
                      padding=int((filter_size-1)/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=filter_size, stride=1,
                      padding=int((filter_size-1)/2))
        )
        # initialization
        initialize_weights(self.body, 0.1)

    def forward(self, x):
        res = self.body(x)
        output = torch.add(x, res)
        return output


class MSRRModule(nn.Module):
    def __init__(self, args, scale):
        super(MSRRModule, self).__init__()

        num_filters = 3 * (scale * scale)

        self.first_conv = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=3, stride=1,
                                    padding=1)

        lr_res_block_layers = []
        for i in range(args.num_lr_blocks):
            lr_res_block_layers.append(ResidualBlock(num_channels=num_filters))
        if args.num_lr_blocks > 0:
            self.lr_res_blocks = nn.Sequential(*lr_res_block_layers)

        self.upsample = nn.PixelShuffle(scale)
        self.middle_conv = nn.Conv2d(in_channels=3, out_channels=args.num_hr_filters, kernel_size=3, stride=1, padding=1)

        hr_res_block_layers = []
        for i in range(args.num_hr_blocks):
            hr_res_block_layers.append(ResidualBlock(num_channels=args.num_hr_filters, filter_size=args.hr_filter_size))
        if args.num_hr_blocks > 0:
            self.hr_res_blocks = nn.Sequential(*hr_res_block_layers)

        self.final_conv = nn.Conv2d(in_channels=args.num_hr_filters, out_channels=3, kernel_size=3, stride=1, padding=1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.first_conv, self.final_conv], 0.1)

        self.interpolate = args.interpolate

    def forward(self, x):
        out = self.lrelu(self.first_conv(x))

        if hasattr(self, 'lr_res_blocks'):
            out = self.lr_res_blocks(out)

        out = self.upsample(out)
        out = self.lrelu(self.middle_conv(out))

        if hasattr(self, 'hr_res_blocks'):
            out = self.hr_res_blocks(out)

        out = self.final_conv(self.lrelu(out))

        base = F.interpolate(x, scale_factor=4, mode=self.interpolate, align_corners=False)
        out += base
        return out
