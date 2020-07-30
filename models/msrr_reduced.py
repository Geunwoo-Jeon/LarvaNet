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
import validate

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
        self.volume_per_step = 0

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_blocks', type=int, default=32, help='The number of residual blocks.')
        parser.add_argument('--interpolate', type=str, default='bicubic', help='Interpolation method.')
        parser.add_argument('--res_weight', type=float, default=1.0, help='The scaling factor.')

        parser.add_argument('--val_volume', type=float, default=3e9, help='How much volume need for validation.')

        parser.add_argument('--lr', type=float, default=4e-4, help='Initial learning rate.')
        parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor.')
        parser.add_argument('--threshold', type=float, default=0.001, help='Learning rate decay factor.')
        parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate.')
        parser.add_argument('--patience', type=int, default=3, help='patience for lr scheduler')

        self.args, remaining_args = parser.parse_known_args(args=args)
        return copy.deepcopy(self.args), remaining_args

    def prepare(self, is_training, scales, global_step=0):
        # config. parameters
        self.global_step = global_step
        self.total_volume = 0.0
        self.temp_volume = 0

        self.scale_list = scales
        for scale in self.scale_list:
            if (not scale in [2, 3, 4]):
                raise ValueError('Unsupported scale is provided.')
        if len(self.scale_list) != 1:
            raise ValueError('Only one scale should be provided.')
        self.scale = self.scale_list[0]

        # PyTorch model
        self.model = MSRRModule(args=self.args, scale=self.scale)

        # if (is_training):
        #     self.optim = optim.Adam(
        #         filter(lambda p: p.requires_grad, self.model.parameters()),
        #         lr=self._get_learning_rate()
        #     )
        #     self.loss_fn = nn.L1Loss()

        if is_training:
            self.loss_fn = nn.L1Loss()
            self.optim = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, mode='max', factor=self.args.lr_decay, patience=self.args.patience,
                threshold=self.args.threshold, threshold_mode='abs', min_lr=self.args.min_lr, verbose=True)

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

    def train_step_larva(self, args, val_dataloader, input_tensor, truth_tensor, summary=None):
        self.global_step += 1
        self.temp_volume += self.volume_per_step

        # make outputs(forward)
        out = self.model(input_tensor)
        loss = self.loss_fn(out, truth_tensor)

        # do back propagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.global_step == 1:
            self.validate_for_train(args, val_dataloader)

        if self.temp_volume >= self.args.val_volume:
            self.total_volume += self.temp_volume
            self.temp_volume = 0
            self.validate_for_train(args, val_dataloader)
            self.save(base_path=args.train_path)
            print(f'saved a model checkpoint at volume {self.total_volume/1e9:.0f}G')
            # summary, not important
            if summary is not None:
                lr = self.get_lr()
                summary.add_scalar('loss', loss, self.global_step)
                summary.add_scalar('lr', lr, self.global_step)

                input_tensor_uint8 = input_tensor.clamp(0, 255).byte()
                output_tensor_uint8 = out.clamp(0, 255).byte()
                truth_tensor_uint8 = truth_tensor.clamp(0, 255).byte()
                for i in range(min(4, len(input_tensor_uint8))):
                    summary.add_image('input/%d' % i, input_tensor_uint8[i, :, :, :], self.global_step)
                    summary.add_image('output/%d' % i, output_tensor_uint8[i, :, :, :], self.global_step)
                    summary.add_image('truth/%d' % i, truth_tensor_uint8[i, :, :, :], self.global_step)

        return loss.item()

    def validate_for_train(self, args, dataloader):
        # scheduling lr by validation
        print('begin validation')
        num_images = dataloader.get_num_images()
        psnr_list = []

        for image_index in range(num_images):
            input_image, truth_image, image_name = dataloader.get_image_pair(image_index=image_index,
                                                                             scale=4)
            output_image = self.upscale(input_list=[input_image], scale=4)[0]
            truth_image = validate._image_to_uint8(truth_image)
            output_image = validate._image_to_uint8(output_image)
            truth_image = validate._fit_truth_image_size(output_image=output_image, truth_image=truth_image)

            psnr = validate._image_psnr(output_image=output_image, truth_image=truth_image)
            psnr_list.append(psnr)

        average_psnr = np.mean(psnr_list)
        print(f'step {self.global_step}, volume {self.total_volume/1e9:.0f}G,'
              f' psnr={average_psnr:.8f}, lr = {self.get_lr():.6f}')
        self.scheduler.step(average_psnr)

    def upscale(self, input_list, scale):
        # numpy to torch
        input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)

        # get SR
        output_tensor = self.model(input_tensor)

        # finalize
        return output_tensor.detach().cpu().numpy()

    def get_lr(self):
        return self.optim.param_groups[0]['lr']
    
    def fwd_runtime(self, input_tensor):
        output_tensor = self.model(input_tensor)
        return output_tensor

    def _get_learning_rate(self):
        return self.args.learning_rate * (self.args.learning_rate_decay ** (
                self.global_step // self.args.learning_rate_decay_steps))


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, weight=1.0):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
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

        res_block_layers = []
        for i in range(args.num_blocks):
            res_block_layers.append(ResidualBlock(num_channels=num_filters, weight=args.res_weight))
        self.res_blocks = nn.Sequential(*res_block_layers)
        self.upsample = nn.PixelShuffle(scale)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights(self.first_conv, 0.1)

        self.interpolate = args.interpolate

    def forward(self, x):
        out = self.lrelu(self.first_conv(x))

        out = self.res_blocks(out)

        out = self.upsample(out)
        base = F.interpolate(x, scale_factor=4, mode=self.interpolate, align_corners=False)
        out += base

        return out
