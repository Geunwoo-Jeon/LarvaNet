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
    return LarvaNet()


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


class LarvaNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.volume_per_step = 0

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--num_modules', type=int, default=2, help='The number of residual blocks at LR domain.')
        parser.add_argument('--num_blocks', type=str, default=16, help='The number of residual blocks at HR domain.')
        parser.add_argument('--interpolate', type=str, default='bicubic', help='Interpolation method.')

        parser.add_argument('--val_volume', type=float, default=3e9, help='How much volume need for validation.')

        parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
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
            if not (scale in [2, 3, 4]):
                raise ValueError('Unsupported scale is provided.')
        if len(self.scale_list) != 1:
            raise ValueError('Only one scale should be provided.')
        self.scale = self.scale_list[0]

        # PyTorch model
        self.model = LarvaNetModule(args=self.args)

        if is_training:
            self.loss_fn = nn.L1Loss()
            self.optim = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, mode='max', factor=self.args.lr_decay, patience=self.args.patience,
                threshold=self.args.threshold, threshold_mode='abs', min_lr=self.args.min_lr, verbose=True)

            # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            #     self.optim, self.steps_per_epoch * 150, T_mult=1, eta_min=self.args.lr/10, last_epoch=-1)

            # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     self.optim, 0.0005, total_steps=None, epochs=200, steps_per_epoch=self.steps_per_epoch,
            #     anneal_strategy='linear', div_factor=50.0, final_div_factor=10)

        # configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def train_step_larva(self, args, val_dataloader, input_tensor, truth_tensor, summary=None):
        self.global_step += 1
        self.temp_volume += self.volume_per_step
        # make outputs(forward)
        fea = self.model.head(input_tensor)
        base = self.model.base(input_tensor)
        loss = 0
        features = []
        for i in range(self.args.num_modules):
            if i == 0:
                features.append(getattr(self.model, f'body_{i}')(fea))
            else:
                features.append(getattr(self.model, f'body_{i}')(features[i-1]))
            out = getattr(self.model, f'body_{i}').leg(features[i], base)
            loss += self.loss_fn(out, truth_tensor)
        out = self.model.tail(features, base)
        loss += self.loss_fn(out, truth_tensor)
        loss = loss / (self.args.num_modules + 1)

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
        with torch.no_grad():
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

    def save(self, base_path):
        save_path = os.path.join(base_path, 'model_step%d_vol%.0fG.pth' % (self.global_step, self.total_volume/1e9))
        torch.save(self.model.state_dict(), save_path)

    def restore(self, ckpt_path, target=None):
        pretrained_dict = torch.load(ckpt_path)
        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)

    def get_model(self):
        return self.model

    def get_next_train_scale(self):
        scale = self.scale_list[np.random.randint(len(self.scale_list))]
        return scale

    def get_lr(self):
        return self.optim.param_groups[0]['lr']

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

#
# class CAlayer(nn.Module):
#     def __init__(self, num_channels, reduction=12):
#         super(CAlayer, self).__init__()
#         self.linear_res = nn.Sequential(
#             nn.Linear(in_features=num_channels, out_features=num_channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=num_channels // reduction, out_features=num_channels)
#         )
#         initialize_weights(self.linear_res, 0.1)
#         self.scaling = nn.Sigmoid()
#
#     def forward(self, x):
#         N, C, W, H = x.size()
#         var = x.view(N, C, -1).var(dim=2, keepdim=True)
#         var = var.view(N, C)
#         mean_var = var.mean(dim=1, keepdim=True)
#         var_var = var.var(dim=1, keepdim=True) + 1e-5
#         std_var = var_var.sqrt()
#         normalized_var = (var - mean_var) / std_var
#         val_res = self.linear_res(normalized_var)
#         normalized_var = normalized_var.view(N, C, 1, 1)
#         val_res = val_res.view(N, C, 1, 1)
#         y = self.scaling(normalized_var + val_res).expand(N, C, W, H)
#         return x * y


class LarvaHead(nn.Module):
    def __init__(self):
        super(LarvaHead, self).__init__()
        num_filters = 48
        self.feature_extraction = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=3, stride=1,
                                            padding=1)
        initialize_weights(self.feature_extraction, 0.1)

    def forward(self, x):
        fea = self.feature_extraction(x)
        return fea


class LarvaBody(nn.Module):
    def __init__(self, num_blocks):
        super(LarvaBody, self).__init__()
        num_filters = 48
        res_block_layers = []
        for i in range(num_blocks):
            res_block_layers.append(ResidualBlock(num_channels=num_filters))
        self.res_blocks = nn.Sequential(*res_block_layers)
        self.leg = LarvaLeg()

    def forward(self, x):
        fea = self.res_blocks(x)
        return x + fea


class LarvaLeg(nn.Module):
    def __init__(self):
        super(LarvaLeg, self).__init__()
        num_filters = 48
        self.recon_block = nn.Sequential(
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        )
        initialize_weights(self.recon_block, 0.1)
        self.upsample = nn.PixelShuffle(4)

    def forward(self, fea, base):
        fea = self.recon_block(fea)
        out = self.upsample(fea)
        out += base
        return out


class LarvaTail(nn.Module):
    def __init__(self, num_modules):
        super(LarvaTail, self).__init__()
        num_filters = 48
        self.merge_conv = nn.Conv2d(in_channels=num_filters * num_modules, out_channels=num_filters,
                                    kernel_size=3, stride=1, padding=1)
        self.recon_block = nn.Sequential(
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        )
        initialize_weights([self.recon_block, self.merge_conv], 0.1)
        self.upsample = nn.PixelShuffle(4)

    def forward(self, features, base):
        fea = torch.cat(features, dim=1)
        fea = self.merge_conv(fea)
        fea = self.recon_block(fea)
        out = self.upsample(fea)
        out += base
        return out


class LarvaNetModule(nn.Module):
    def __init__(self, args):
        super(LarvaNetModule, self).__init__()
        self.len = args.num_modules
        self.interpolate = args.interpolate
        self.head = LarvaHead()
        blocks = list(map(lambda x: int(x), args.num_blocks.split(',')))
        if len(blocks) != self.len:
            raise GeneratorExit('Argument num_blocks should have the same number of elements as num_modules.')
        else:
            for i in range(self.len):
                setattr(self, f'body_{i}', LarvaBody(num_blocks=blocks[i]))
        self.tail = LarvaTail(self.len)

    def base(self, x):
        base = F.interpolate(x, scale_factor=4, mode=self.interpolate, align_corners=False)
        return base

    def forward(self, x):
        fea = self.head(x)
        features = []
        for i in range(self.len):
            if i == 0:
                features.append(getattr(self, f'body_{i}')(fea))
            else:
                features.append(getattr(self, f'body_{i}')(features[i-1]))
        base = self.base(x)
        out = self.tail(features, base)
        return out
