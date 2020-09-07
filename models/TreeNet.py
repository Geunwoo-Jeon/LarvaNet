import argparse
import copy
import math
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import validate

from models.base import BaseModel


def create_model():
    return ModelContainer()


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


class ModelContainer(BaseModel):
    def __init__(self):
        super().__init__()
        self.volume_per_step = 0

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_common_blocks', type=int, default=8, help='The number of residual blocks.')
        parser.add_argument('--num_branches', type=int, default=1, help='The number of residual blocks.')
        parser.add_argument('--num_branch_blocks', type=int, default=8, help='The number of residual blocks.')
        parser.add_argument('--interpolate', type=str, default='bicubic', help='Interpolation method.')
        parser.add_argument('--res_weight', type=float, default=1.0, help='The scaling factor.')

        parser.add_argument('--lr', type=float, default=4e-4, help='Initial learning rate.')
        parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor.')
        parser.add_argument('--lr_step', type=int, default=200000, help='Learning rate decay step.')

        parser.add_argument('--val_volume', type=float, default=30e9, help='How much volume need for validation.')
        parser.add_argument('--threshold', type=float, default=0.001, help='Threshold for reduceLRonPlateau.')
        parser.add_argument('--min_lr', type=float, default=1e-8, help='Minimum learning rate.')
        parser.add_argument('--patience', type=int, default=1, help='patience for lr scheduler')

        self.args, remaining_args = parser.parse_known_args(args=args)
        return copy.deepcopy(self.args), remaining_args

    def prepare(self, is_training, scales, global_step=0):
        # config. parameters
        self.global_step = global_step
        self.total_volume = 0.0
        self.temp_volume = 0
        self.tmp_time = time.time()

        self.scale_list = scales
        for scale in self.scale_list:
            if (not scale in [2, 3, 4]):
                raise ValueError('Unsupported scale is provided.')
        if len(self.scale_list) != 1:
            raise ValueError('Only one scale should be provided.')
        self.scale = self.scale_list[0]

        # PyTorch model, first branch is default
        self.model = TreeNet(args=self.args, scale=self.scale)

        if is_training:
            self.loss_fn = nn.L1Loss()
            self.optim = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optim, self.args.lr_step, self.args.lr_decay)
            # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            #     self.optim, mode='max', factor=self.args.lr_decay, patience=self.args.patience,
            #     threshold=self.args.threshold, threshold_mode='abs', min_lr=self.args.min_lr, verbose=True)

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

    def train_step_larva(self, args, val_dataloader, input_tensor, truth_tensor, summary=None):
        # if self.global_step == 0:
        #     self.validate_for_train(args, val_dataloader)

        self.global_step += 1
        self.temp_volume += self.volume_per_step

        # get SR and calculate loss

        fea = self.model.common_parts(input_tensor)
        loss = 0
        for i in range(self.args.num_branches):
            out = getattr(self.model, f'branch_{i}')(fea)
            out += F.interpolate(input_tensor, scale_factor=4, mode=self.args.interpolate, align_corners=False)
            loss += self.loss_fn(out, truth_tensor)
        loss = loss / self.args.num_branches

        # do back propagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.scheduler.step()

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
        time_per_val = time.time() - self.tmp_time
        self.tmp_time = time.time()
        step_per_val = self.args.val_volume / self.volume_per_step
        print(f'begin validation. {step_per_val:.0f} steps for {time_per_val:.0f} sec.')
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
              f' psnr={average_psnr:.8f}, lr = {self.get_lr():.8f}')

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


class TreeNet(nn.Module):
    def __init__(self, args, scale):
        super(TreeNet, self).__init__()

        num_filters = 3 * (scale * scale)

        # functions
        self.upsample = nn.PixelShuffle(scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.interpolate = args.interpolate

        # common parts
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=3, stride=1,
                                    padding=1)
        common_block_layers = []
        for i in range(args.num_common_blocks):
            common_block_layers.append(ResidualBlock(num_channels=num_filters, weight=args.res_weight))
        self.common_blocks = nn.Sequential(*common_block_layers)

        # independent parts
        for i in range(args.num_branches):
            tmp_branch_blocks = []
            for _ in range(args.num_branch_blocks):
                tmp_branch_blocks.append(ResidualBlock(num_channels=num_filters, weight=args.res_weight))
            setattr(self, f'branch_{i}', nn.Sequential(*tmp_branch_blocks, self.upsample))

        # initialization
        initialize_weights(self.first_conv, 0.1)

        # common parts
        self.common_parts = nn.Sequential(self.first_conv, self.lrelu, self.common_blocks)

    def forward(self, x):
        out = self.common_parts(x)
        out = self.branch_0(out)
        base = F.interpolate(x, scale_factor=4, mode=self.interpolate, align_corners=False)
        out += base

        return out
