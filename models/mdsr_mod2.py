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
    return MDSR()


class MDSR(BaseModel):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--edsr_conv_features', type=int, default=64, help='The number of convolutional features.')
        parser.add_argument('--edsr_res_blocks', type=int, default=16, help='The number of residual blocks.')
        parser.add_argument('--edsr_res_weight', type=float, default=1.0, help='The scaling factor.')
        parser.add_argument('--steps_per_epoch', type=int, default=30000, help='Num of steps that equal to 1 epoch.')

        parser.add_argument('--edsr_learning_rate', type=float, default=1e-4, help='Initial learning rate.')
        parser.add_argument('--edsr_learning_rate_decay', type=float, default=0.5, help='Learning rate decay factor.')
        parser.add_argument('--edsr_learning_rate_decay_steps', type=int, default=200000,
                            help='The number of training steps to perform learning rate decay.')
        parser.add_argument('--coef', type=int, default=1, help='Whether use additional path or not')


        self.args, remaining_args = parser.parse_known_args(args=args)
        return copy.deepcopy(self.args), remaining_args

    def prepare(self, is_training, scales, global_step=0):
        # config. parameters
        self.global_step = global_step

        self.scale_list = scales
        for scale in self.scale_list:
            if (not scale in [2, 3, 4]):
                raise ValueError('Unsupported scale is provided.')
        if len(self.scale_list) < 3:
          raise ValueError('Three scales should be provided.')

        # PyTorch model
        self.model = MDSRModule(args=self.args, scale_list=self.scale_list)

        optim_params = []
        for k, v in self.model.named_parameters():
            v.requires_grad = False
            if k.find('transformer') >= 0:
                v.requires_grad = True
                optim_params.append(v)

        if (is_training):
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self._get_learning_rate()
            )

            # self.scheduler = optim.lr_scheduler.OneCycleLR(
            #     self.optim, max_lr=0.005, total_steps=None, epochs=200, steps_per_epoch=self.args.steps_per_epoch,
            #     anneal_strategy='cos', div_factor=50, final_div_factor=100, last_epoch=-1)
            self.loss_fn = nn.L1Loss()

        # configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def save(self, base_path):
        save_path = os.path.join(base_path, 'model_%d.pth' % (self.global_step))
        torch.save(self.model.state_dict(), save_path)

    def restore(self, ckpt_path, target=None):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)

    def get_model(self):
        return self.model

    def get_next_train_scale(self):
        scale = self.scale_list[np.random.randint(len(self.scale_list))]
        return scale

    def train_step(self, input_list, scale, truth_list, summary=None):
        # numpy to torch
        input_tensor = torch.as_tensor(input_list, dtype=torch.float32, device=self.device)
        truth_tensor = torch.as_tensor(truth_list, dtype=torch.float32, device=self.device)

        # get SR and calculate loss
        output_tensor = self.model(input_tensor, scale=scale)
        loss = self.loss_fn(output_tensor, truth_tensor)

        # adjust learning rate
        lr = self._get_learning_rate()
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr


        # do back propagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # self.scheduler.step()

        # finalize
        self.global_step += 1

        # write summary
        if (summary is not None):
            summary.add_scalar('loss', loss, self.global_step)
            summary.add_scalar('lr', lr, self.global_step)

            input_tensor_uint8 = input_tensor.clamp(0, 255).byte()
            output_tensor_uint8 = output_tensor.clamp(0, 255).byte()
            truth_tensor_uint8 = truth_tensor.clamp(0, 255).byte()

            for i in range(min(4, len(input_list))):
                summary.add_image('input/%d' % i, input_tensor_uint8[i, :, :, :], self.global_step)
                summary.add_image('output/%d' % i, output_tensor_uint8[i, :, :, :], self.global_step)
                summary.add_image('truth/%d' % i, truth_tensor_uint8[i, :, :, :], self.global_step)

        return loss.item()

    def upscale(self, input_list, scale):
        # numpy to torch
        input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)

        # get SR
        output_tensor = self.model(input_tensor, scale=scale)

        # finalize
        return output_tensor.detach().cpu().numpy()

    def _get_learning_rate(self):
        return self.args.edsr_learning_rate * (self.args.edsr_learning_rate_decay ** (
                    self.global_step // self.args.edsr_learning_rate_decay_steps))


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
        self.weight_data = torch.eye(3).view(3, 3, 1, 1)
        self.bias_data = sign * torch.Tensor(rgb_mean)

        for params in self.parameters():
            params.requires_grad = False


class AAFM(nn.Module):
    def __init__(self, num_channels, kernel_size=3, coef=1):
        super(AAFM, self).__init__()
        self.coef = coef
        self.transformer = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size//2, groups=num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=num_channels)
        )
        self.scaling = nn.Sigmoid()

    def forward(self, x):
        if self.coef == 0:
            return x
        else:
            tmp = x + self.transformer(x)
            N, C, W, H = tmp.size()
            tmp_var = tmp.view(N, C, -1).var(dim=-1, keepdim=True)
            mean_var = tmp_var.view(N, -1).mean(dim=-1, keepdim=True).unsqueeze(-1).expand_as(tmp_var)
            var_var = tmp_var.view(N, -1).var(dim=-1, keepdim=True).unsqueeze(-1).expand_as(tmp_var) + 1e-5
            std_var = var_var.sqrt()

            tmp_var = (tmp_var - mean_var) / std_var
            modulation_map_CSI = tmp_var.unsqueeze(-1).expand_as(x)
            tmp = tmp * self.scaling(modulation_map_CSI)
            return self.coef * tmp + x

class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size=3, weight=1.0, coef=1):
        super(ResidualBlock, self).__init__()
        self.coef = coef
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2)
        )

        self.aafm = AAFM(num_channels=num_channels, coef=self.coef)
        self.weight = weight

    def forward(self, x):
        res = self.body(x).mul(self.weight)
        res = self.aafm(res)
        output = torch.add(x, res)
        return output


class UpsampleBlock(nn.Sequential):
    def __init__(self, num_channels, scale):
        # super(UpsampleBlock, self).__init__()

        layers = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                layers.append(
                    nn.Conv2d(in_channels=num_channels, out_channels=4 * num_channels, kernel_size=3, stride=1,
                              padding=1))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(
                nn.Conv2d(in_channels=num_channels, out_channels=9 * num_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.PixelShuffle(3))

        super(UpsampleBlock, self).__init__(*layers)


class MDSRModule(nn.Module):
    def __init__(self, args, scale_list):
        super(MDSRModule, self).__init__()

        self.sub_mean = MeanShift([114.4, 111.5, 103.0], sign=1.0)
        m_head = [nn.Conv2d(in_channels=3, out_channels=args.edsr_conv_features, kernel_size=3, stride=1, padding=1)]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(num_channels=args.edsr_conv_features, weight=args.edsr_res_weight, kernel_size=5, coef=args.coef),
                ResidualBlock(num_channels=args.edsr_conv_features, weight=args.edsr_res_weight, kernel_size=5, coef=args.coef)
            ) for _ in scale_list
        ])

        m_body = [
            ResidualBlock(num_channels=args.edsr_conv_features, weight=args.edsr_res_weight, coef=args.coef) for _ in
            range(args.edsr_res_blocks)
        ]
        m_body.append(
            nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=args.edsr_conv_features, kernel_size=3,
                      stride=1, padding=1))

        self.upsample = nn.ModuleList([
            UpsampleBlock(num_channels=args.edsr_conv_features, scale=scale) for scale in scale_list
        ])

        m_tail = [nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=3, kernel_size=3, stride=1, padding=1)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.add_mean = MeanShift([114.4, 111.5, 103.0], sign=-1.0)


    def forward(self, x, scale=2):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[scale - 2](x)

        res = self.body(x)
        res += x

        x = self.upsample[scale - 2](res)
        x = self.tail(x)
        x = self.add_mean(x)

        return x