import argparse
import copy
import math
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

from models.base import BaseModel

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

def create_model():
    return REGO()


class REGO(BaseModel):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_filters', type=int, default=64, help='The number of convolutional features.')
        parser.add_argument('--len_side', type=int, default=5, help='The number of residual blocks.')
        parser.add_argument('--num_regos', type=int, default=1, help='num of serial repeat of REGO-module')
        parser.add_argument('--weight_scale', type=float, default=0.1, help='The scaling factor.')
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
        self.model = REGOModule(args=self.args, scale=self.scale)
        if (is_training):
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self._get_learning_rate()
            )
            self.loss_fn = nn.L1Loss()

        # configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(next(self.model.parameters()).device)
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


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
        self.weight_data = torch.eye(3).view(3, 3, 1, 1)
        self.bias_data = sign * torch.Tensor(rgb_mean)

        for params in self.parameters():
            params.requires_grad = False


class RESBlock(nn.Module):
    def __init__(self, num_channels, weight=1.0):
        super(RESBlock, self).__init__()
        self.weight = weight
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        )
        initialize_weights(self.body, weight)

    def forward(self, x):
        res = self.body(x)
        output = torch.add(x, res)
        return res, output


class UpsampleBlock(nn.Module):
    def __init__(self, num_channels, out_channels, scale):
        super(UpsampleBlock, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(in_channels=num_channels, out_channels=out_channels * (scale ** 2), kernel_size=3, stride=1,
                      padding=1))
        layers.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        output = self.body(x)
        return output


class REGOModule(nn.Module):
    def __init__(self, args, scale):
        super(REGOModule, self).__init__()
        self.mean_shift = MeanShift([114.4, 111.5, 103.0], sign=1.0)
        self.feature_extraction = nn.Conv2d(in_channels=3, out_channels=args.num_filters, kernel_size=3, stride=1, padding=1)
        self.len_side = args.len_side
        self.num_regos = args.num_regos
        for k in range(self.num_regos):
            for i in range(self.len_side):
                for j in range(self.len_side - i):
                    setattr(self, f'RESB_{k}_{i}_{j}', RESBlock(num_channels=args.num_filters, weight=args.weight_scale))
            if k != (self.num_regos-1):
                setattr(self, f'conv_{k}', nn.Conv2d(in_channels=(self.len_side+1) * args.num_filters,
                                                     out_channels=args.num_filters, kernel_size=3, stride=1, padding=1))
        self.SRrecon = UpsampleBlock(num_channels=(self.len_side+1) * args.num_filters, out_channels=3, scale=scale)
        initialize_weights([self.feature_extraction, self.SRrecon], args.weight_scale)
        self.interpolate = args.interpolate

    def forward(self, x):
        fea = self.feature_extraction(self.mean_shift(x))
        for k in range(self.num_regos):
            # err, fea = self.RESB_bricks[k][0][0](fea)
            err, fea = getattr(self, f'RESB_{k}_0_0')(fea)
            err_in = [err]
            fea_in = [fea]
            for i in range(1, self.len_side):
                err_out = []
                fea_out = []

                # err, fea = self.RESB_bricks[i][0](err_in[0])
                err, fea = getattr(self, f'RESB_{k}_{i}_0')(err_in[0])
                err_out.append(err)
                fea_out.append(fea)

                for j in range(1, i):
                    # err, fea = self.RESB_bricks[i - j][j](fea_in[j - 1] + err_in[j])
                    err, fea = getattr(self, f'RESB_{k}_{i-j}_{j}')(fea_in[j - 1] + err_in[j])
                    err_out.append(err)
                    fea_out.append(fea)

                # fea, err = self.RESB_bricks[0][i](fea_in[i - 1])
                err, fea = getattr(self, f'RESB_{k}_0_{i}')(fea_in[i - 1])
                err_out.append(err)
                fea_out.append(fea)

                fea_in = fea_out
                err_in = err_out

            if k != (self.num_regos - 1):
                fea = getattr(self, f'conv_{k}')(
                    torch.cat((err_out[0], *[err + fea for err, fea in zip(err_out[1:], fea_out[:-1])], fea_out[-1]),
                              dim=1))

        sr = self.SRrecon(
            torch.cat((err_out[0], *[err + fea for err, fea in zip(err_out[1:], fea_out[:-1])], fea_out[-1]), dim=1))
        sr += F.interpolate(x, scale_factor=4, mode=self.interpolate, align_corners=False)
        return sr
