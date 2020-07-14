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
    return SquidNet()


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


class SquidNet(BaseModel):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_modules', type=int, default=2, help='The number of residual blocks at LR domain.')
        parser.add_argument('--num_blocks', type=int, default=16, help='The number of residual blocks at HR domain.')
        parser.add_argument('--interpolate', type=str, default='bicubic', help='Interpolation method.')

        parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
        parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor.')
        parser.add_argument('--threshold', type=float, default=0.005, help='Learning rate decay factor.')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='Initial learning rate.')
        parser.add_argument('--patience', type=int, default=6, help='patience for lr scheduler')
        parser.add_argument('--lambda_cons', type=float, default=0.1, help='lambda for consistency loss')

        self.args, remaining_args = parser.parse_known_args(args=args)
        return copy.deepcopy(self.args), remaining_args

    def prepare(self, is_training, scales, global_step=0):
        # config. parameters
        self.global_step = global_step
        self.epoch = 0

        self.scale_list = scales
        for scale in self.scale_list:
            if (not scale in [2, 3, 4]):
                raise ValueError('Unsupported scale is provided.')
        if len(self.scale_list) != 1:
            raise ValueError('Only one scale should be provided.')
        self.scale = self.scale_list[0]

        # PyTorch model
        self.model = SquidNetModule(args=self.args)

        if is_training:
            self.loss_fn = nn.L1Loss()

            self.optim = optim.AdamW(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=self.args.lr
            )

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optim, mode='max', factor=self.args.lr_decay, patience=self.args.patience,
                    threshold=self.args.threshold, threshold_mode='abs', min_lr=self.args.min_lr, verbose=True)

        # configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def train_step_squid(self, args, val_dataloader, scale, input_tensor, truth_tensor, summary=None):
        self.global_step += 1
        # make outputs(forward)
        outputs = []
        outputs.append(self.model.head(input_tensor))
        for i in range(self.args.num_modules - 1):
            outputs.append(getattr(self.model, f'leg{i}')(outputs[i]))

        # calculate loss
        loss = 0
        for output in outputs:
            loss = loss + self.loss_fn(output, truth_tensor)
        # do back propagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
            
        if self.global_step == 1:
            self.validate_for_train(args, val_dataloader)

        if self.global_step % args.step_per_epoch == 0:
            self.epoch += 1
            if self.epoch % 5 == 0 and self.epoch != 0:
                self.validate_for_train(args, val_dataloader)
                # write summary
                lr = self.get_lr()
                if summary is not None:
                    summary.add_scalar('loss', loss, self.global_step)
                    summary.add_scalar('lr', lr, self.global_step)

                    input_tensor_uint8 = input_tensor.clamp(0, 255).byte()
                    output_tensor_uint8 = out.clamp(0, 255).byte()
                    truth_tensor_uint8 = truth_tensor.clamp(0, 255).byte()
                    for i in range(min(4, len(input_tensor_uint8))):
                        summary.add_image('input/%d' % i, input_tensor_uint8[i, :, :, :], self.epoch)
                        summary.add_image('output/%d' % i, output_tensor_uint8[i, :, :, :], self.epoch)
                        summary.add_image('truth/%d' % i, truth_tensor_uint8[i, :, :, :], self.epoch)
            if self.epoch % 10 == 0:
                self.save(base_path=args.train_path)
                print('saved a model checkpoint at epoch %d' % self.epoch)

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
        print(f'step {self.global_step}, epoch {self.global_step / args.step_per_epoch:.0f},'
              f' psnr={average_psnr:.8f}, lr = {self.get_lr():.6f}')

        for scheduler in self.schedulers:
            scheduler.step(average_psnr)

    def upscale(self, input_list, scale):
        # numpy to torch
        input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)

        # get SR
        output_tensor = self.model(input_tensor)

        # finalize
        return output_tensor.detach().cpu().numpy()

    def save(self, base_path):
        save_path = os.path.join(base_path, 'model_%depoch.pth' % (self.epoch))
        torch.save(self.model.state_dict(), save_path)

    def restore(self, ckpt_path, target=None):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    def get_model(self):
        return self.model

    def get_next_train_scale(self):
        scale = self.scale_list[np.random.randint(len(self.scale_list))]
        return scale

    def get_lr(self):
        return self.optims[0].param_groups[0]['lr']

    def fwd_runtime(self, input_tensor):
        output_tensor = self.model(input_tensor)
        return output_tensor


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
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


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class SquidHeadModule(nn.Module):
    def __init__(self, args):
        super(SquidHeadModule, self).__init__()
        num_filters = 48
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=3, stride=1,
                                    padding=1)
        res_block_layers = []
        for i in range(args.num_blocks):
            res_block_layers.append(ResidualBlock(num_channels=num_filters))
        self.res_blocks = nn.Sequential(*res_block_layers)
        self.upsample = nn.PixelShuffle(4)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights(self.first_conv, 0.1)

        # interpolate method
        self.interpolate = args.interpolate

    def forward(self, x):
        fea = self.lrelu(self.first_conv(x))
        fea = self.res_blocks(fea)
        out = self.upsample(fea)
        base = F.interpolate(x, scale_factor=4, mode=self.interpolate, align_corners=False)
        out = out + base
        return out


class SquidLegModule(nn.Module):
    def __init__(self, args):
        super(SquidLegModule, self).__init__()
        num_filters = 48
        self.pixel_unshuffle = SpaceToDepth(block_size=4)
        res_block_layers = []
        for i in range(args.num_blocks):
            res_block_layers.append(ResidualBlock(num_channels=num_filters))
        self.res_blocks = nn.Sequential(*res_block_layers)
        self.upsample = nn.PixelShuffle(4)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        fea = self.pixel_unshuffle(x)
        fea = self.res_blocks(fea)
        out = self.upsample(fea)
        out += x
        return out


class SquidNetModule(nn.Module):
    def __init__(self, args):
        super(SquidNetModule, self).__init__()
        self.head = SquidHeadModule(args=args)
        self.legs = args.num_modules - 1
        for i in range(self.legs):
            setattr(self, f'leg{i}', SquidLegModule(args=args))

    def forward(self, x):
        out = self.head(x)
        for i in range(self.legs):
            out = getattr(self, f'leg{i}')(out)
        return out
