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
from collections import OrderedDict


def create_model():
    return EDSR()


class EDSR(BaseModel):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--edsr_conv_features', type=int, default=64, help='The number of convolutional features.')
        parser.add_argument('--edsr_res_blocks', type=int, default=16, help='The number of residual blocks.')
        parser.add_argument('--edsr_res_weight', type=float, default=1.0, help='The scaling factor.')

        parser.add_argument('--edsr_learning_rate', type=float, default=1e-4, help='Initial learning rate.')
        parser.add_argument('--edsr_learning_rate_decay', type=float, default=0.5, help='Learning rate decay factor.')
        parser.add_argument('--edsr_learning_rate_decay_steps', type=int, default=200000,
                            help='The number of training steps to perform learning rate decay.')
        parser.add_argument('--batch_size', type=int, default=16, help='Size of the batches for each training step.')

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
        self.model = EDSRModule(args=self.args, scale=self.scale)
        self.model_tmp = EDSRModule(args=self.args, scale=self.scale)
        if (is_training):
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self._get_learning_rate()
            )

            self.optim_tmp = optim.Adam(
                filter(lambda p: p.requires_grad, self.model_tmp.parameters()),
                lr=self._get_learning_rate()
            )
            self.loss_fn = nn.L1Loss()

        # configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model_tmp = self.model_tmp.to(self.device)

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

    def get_batch(self):
        return self.args.batch_size

    def get_named_parameters(self, model):
        return self.model.named_parameters()

    def copy_weight(self, model1, model2):
        for i, j in zip(self.get_named_parameters(model1), self.get_named_parameters(model2)):
            i[1].data = j[1].data.clone()

    def tv_loss(self, img, tv_weight):
        w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        loss = tv_weight * (h_variance + w_variance)
        return loss

    def train_step(self, input_list, scale, truth_list, summary=None, train_kind=True):
        lmbda = 0.05
        # numpy to torch
        input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)
        truth_tensor = torch.tensor(truth_list, dtype=torch.float32, device=self.device)


        # adjust learning rate
        lr = self._get_learning_rate()
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        if (train_kind):
            # get SR and calculate loss
            output_tensor_1, output_tensor_2, output_tensor_3 = self.model(input_tensor)
            loss_primary = self.loss_fn(output_tensor_1, truth_tensor)
            loss_aux = self.loss_fn(output_tensor_2, output_tensor_3)
            loss_tv = self.tv_loss(output_tensor_3, lmbda)

            output_tensor_3_forloss = torch.mean(output_tensor_3, dim=0)
            loss_entropy = output_tensor_3_forloss * torch.log(output_tensor_3_forloss + 1e-20)
            loss_entropy = torch.sum(loss_entropy)

            loss_1 = loss_primary + loss_aux
            loss_2 = loss_primary + loss_tv + lmbda * loss_entropy

            # do back propagation
            self.optim.zero_grad()
            loss_1.backward()
            self.optim.step()

            self.global_step += 1

            # write summary
            if (summary is not None):
                summary.add_scalar('loss_1', loss_1, self.global_step)
                summary.add_scalar('loss_2', loss_2, self.global_step)
                summary.add_scalar('loss', loss_primary, self.global_step)
                summary.add_scalar('loss_aux', loss_aux, self.global_step)
                summary.add_scalar('lr', lr, self.global_step)

                output_tensor_uint8 = output_tensor_1.clamp(0, 255).byte()
                for i in range(min(4, len(input_list))):
                    summary.add_image('input/%d' % i, input_list[i], self.global_step)
                    summary.add_image('output/%d' % i, output_tensor_uint8[i, :, :, :], self.global_step)
                    summary.add_image('truth/%d' % i, truth_list[i], self.global_step)

            return loss_1.item()

        else:
            with torch.autograd.set_detect_anomaly(True):
                self.copy_weight(self.model_tmp, self.model)

                output_tensor_1, output_tensor_2, output_tensor_3 = self.model_tmp(input_tensor)
                loss_primary = self.loss_fn(output_tensor_1, truth_tensor)
                loss_aux = self.loss_fn(output_tensor_2, output_tensor_3)

                loss_1 = loss_primary + loss_aux


            # # current theta_1
            # fast_weights = OrderedDict((name, param) for (name, param) in self.model.named_parameters())
            # print(fast_weights)
            # # create_graph flag for computing second-derivative
            # grads = torch.autograd.grad(loss_1, self.model.parameters(), create_graph=True)
            # data = [p.data for p in list(self.model.parameters())]
            #
            # # compute theta_1^+ by applying sgd on multi-task loss
            # fast_weights = OrderedDict(
            #     (name, param - lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

            # compute primary loss with the updated thetat_1^+

                self.optim_tmp.zero_grad()
                loss_1.backward()
                self.optim_tmp.step()

                output_tensor_1, output_tensor_2, output_tensor_3 = self.model_tmp(input_tensor)

                loss_primary = self.loss_fn(output_tensor_1, truth_tensor)
                loss_entropy = output_tensor_3 * torch.log(output_tensor_3 + 1e-20)
                loss_entropy = torch.sum(loss_entropy)
                loss_tv = self.tv_loss(output_tensor_3, lmbda)
                loss_2 = loss_primary + loss_tv + lmbda * loss_entropy

                self.optim_tmp.zero_grad()
                loss_2.backward()
                self.optim_tmp.step()

            self.global_step += 1

            # write summary
            if (summary is not None):
                summary.add_scalar('loss_1', loss_1, self.global_step)
                summary.add_scalar('loss_2', loss_2, self.global_step)
                summary.add_scalar('loss', loss_primary, self.global_step)
                summary.add_scalar('loss_aux', loss_aux, self.global_step)
                summary.add_scalar('lr', lr, self.global_step)

                output_tensor_uint8 = output_tensor_1.clamp(0, 255).byte()
                for i in range(min(4, len(input_list))):
                    summary.add_image('input/%d' % i, input_list[i], self.global_step)
                    summary.add_image('output/%d' % i, output_tensor_uint8[i, :, :, :], self.global_step)
                    summary.add_image('truth/%d' % i, truth_list[i], self.global_step)

            return loss_2.item()



    def upscale(self, input_list, scale):
        # numpy to torch
        input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)

        # get SR
        output_tensor = self.model(input_tensor)

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
            params.requires_grad = True


class ResidualBlock(nn.Module):
    def __init__(self, num_channels, weight=1.0):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        )
        self.weight = weight

    def forward(self, x):
        res = self.body(x).mul(self.weight)
        output = torch.add(x, res)
        return output


class UpsampleBlock(nn.Module):
    def __init__(self, num_channels, scale):
        super(UpsampleBlock, self).__init__()

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

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        output = self.body(x)
        return output


class EDSRModule(nn.Module):
    def __init__(self, args, scale):
        super(EDSRModule, self).__init__()

        self.mean_shift = MeanShift([114.4, 111.5, 103.0], sign=1.0)
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=args.edsr_conv_features, kernel_size=3, stride=1,
                                    padding=1)

        res_block_layers = []
        for i in range(args.edsr_res_blocks):
            res_block_layers.append(ResidualBlock(num_channels=args.edsr_conv_features, weight=args.edsr_res_weight))
        self.res_blocks = nn.Sequential(*res_block_layers)
        self.after_res_conv = nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=args.edsr_conv_features,
                                        kernel_size=3, stride=1, padding=1)

        self.upsample = UpsampleBlock(num_channels=args.edsr_conv_features, scale=scale)
        self.final_conv = nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=3, kernel_size=3, stride=1,
                                    padding=1)

        self.aux_conv = nn.Sequential(
            nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=args.edsr_conv_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=args.edsr_conv_features//4, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.real_criterion = nn.Sequential(
            nn.Conv2d(in_channels=args.edsr_conv_features, out_channels=args.edsr_conv_features//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=args.edsr_conv_features//4, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.sigmoid = nn.Sigmoid()
        self.mean_inverse_shift = MeanShift([114.4, 111.5, 103.0], sign=-1.0)

    def crc_ff(self, input, weights):
        net = F.conv2d(input, weights['block{:d}.0.weight'], weights['block{:d}.0.bias'],
                       padding=1)
        net = F.relu(net, inplace=True)
        net = F.conv2d(net, weights['block{:d}.3.weight'], weights['block{:d}.3.bias'],
                       padding=1)

        return net

    def conv_ff(self, input, weights):
        net = F.conv2d(input, weights['block{:d}.0.weight'], weights['block{:d}.0.bias'],
                       padding=1)
        return net


    def forward(self, x, weights=None):

        if weights is None:
            x = self.mean_shift(x)
            x = self.first_conv(x)

            res = self.res_blocks(x)
            res = self.after_res_conv(res)
            tmp = torch.add(x, res)

            x1 = self.upsample(tmp)
            x1 = self.final_conv(x1)
            x1 = self.mean_inverse_shift(x1)

            x2 = self.aux_conv(tmp)

            x3 = self.real_criterion(x)
            x3 = self.sigmoid(x3)

            return x1, x2, x3

        else:
            x = self.mean_shift(x)
            x = self.conv_ff(x, weights)

            res = self.res_blocks(x)
            res = self.conv_ff(res, weights)
            tmp = torch.add(x, res)

            x1 = self.upsample(x)
            x1 = self.conv_ff(x1)
            x1 = self.mean_inverse_shift(x1)

            x2 = self.conv_ff(tmp)

            x3 = self.conv_ff(tmp)
            x3 = self.sigmoid(x3)

            return x1, x2, x3
