import argparse
import copy
import math
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


from models.base import BaseModel

# ipkpn with l1 loss

def create_model():
  return LPKPN()

class LPKPN(BaseModel):
  def __init__(self):
    super().__init__()

  def parse_args(self, args):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_filters', type=int, default=64, help='The number of filters.')
    parser.add_argument('--num_blocks', type=int, default=16, help='The number of modules.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='Learning rate decay factor.')
    parser.add_argument('--learning_rate_decay_steps', type=int, default=200000, help='The number of training steps to perform learning rate decay.')
    parser.add_argument('--edsr_train_patch_size', type=int, default=48,
                        help='Size of the input patch during training.')
    parser.add_argument('--kernel_size', type=int, default=5, help='The size of kernel.')

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
    self.model = LPKPN_Module(args=self.args, scale=self.scale)
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
    return self.args.learning_rate * (self.args.learning_rate_decay ** (self.global_step // self.args.learning_rate_decay_steps))
  
class Loss(nn.Module):
  def __init__(self, device, width):
    super(Loss, self).__init__()

    self.device = device
    self.sigma = 8

    gaussian = np.exp(-1.*np.arange(-(width/2), width/2)**2/(2*self.sigma**2))
    gaussian = np.outer(gaussian, gaussian.reshape((width, 1)))	# extend to 2D
    gaussian = gaussian/np.sum(gaussian)								# normailization
    gaussian = np.reshape(gaussian, (1, 1, width, width)) 	# reshape to 4D
    gaussian = np.tile(gaussian, (3, 1, 1, 1))
    self.gaussian = torch.FloatTensor(gaussian).to(self.device)

  def forward(self, output_tensor, truth_tensor):
    ms_ssim_loss = 1 - ms_ssim(output_tensor, truth_tensor, data_range=255, size_average=True)

    l1_loss = torch.abs(truth_tensor - output_tensor)
    #l1_loss_filtered = torch.mul(l1_loss, self.gaussian)
    l1_loss_filtered = F.conv2d(input=l1_loss, weight=self.gaussian, groups=3)

    l1_loss_filtered = torch.mean(l1_loss_filtered)

    return 0.16*torch.mean(ms_ssim_loss)+0.84*l1_loss_filtered

class MeanShift(nn.Conv2d):
  def __init__(self, rgb_mean, sign):
    super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
    self.weight_data = torch.eye(3).view(3, 3, 1, 1)
    self.bias_data = sign * torch.Tensor(rgb_mean)

    for params in self.parameters():
      params.requires_grad = False


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
  def __init__(self, scale):
    super(UpsampleBlock, self).__init__()

    self.shuffle = nn.PixelShuffle(scale)

  def forward(self, x):
    output = self.shuffle(x)
    return output


class pixelConv(nn.Module):
  # Generate pixel kernel  (3*k*k)xHxW
  def __init__(self, num_channels, ksize=3):
    super(pixelConv,self).__init__()
    self.padding = (ksize-1)//2
    self.ksize = ksize
    self.zero_padding = nn.ZeroPad2d(self.padding)
    self.kernel_conv = nn.Conv2d(in_channels=num_channels, out_channels=3*ksize**2, kernel_size=3, stride=1, padding=1)

  def forward(self, x_feature, x):
    
    kernel_set = self.kernel_conv(x_feature)
    # print(kernel_set.size()) # 16 (3*3*3) 48 48 

    dtype = kernel_set.data.type() 
    ks = self.ksize
    N = self.ksize**2 # patch size 

    # padding the input image with zero values
    if self.padding:
      x = self.zero_padding(x)
        
    p = self._get_index(kernel_set, dtype)
    # print(p.size()) # 16 3*3+3*3 (x axis+ y axis) 48 48 
    p = p.contiguous().permute(0, 2, 3, 1).long()
    # print(p.size()) # 16 48 48 3*3+3*3 

    x_pixel_set = self._get_x_q(x, p, N)
    # print(x_pixel_set.size()) # 16 3 48 48 9  

    b,c,h,w = kernel_set.size()
    kernel_set_reshape = kernel_set.reshape(-1,self.ksize**2,3,h,w).permute(0,2,3,4,1)
    # print(kernel_set_reshape.size()) # 16 3 48 48 N (=kxk)  

    x_ = x_pixel_set
 
    out = x_ * kernel_set_reshape
    out = out.sum(dim=-1, keepdim=True).squeeze(dim=-1)
    out = out
    # print(out.size()) # 16 3 48 48 
 
    return out 

  def _get_index(self, kernel_set, dtype):
    '''
    get absolute index of each pixel in image
    '''
    # print('------get_index function------')
    N, b, h, w = self.ksize**2, kernel_set.size(0), kernel_set.size(2), kernel_set.size(3)
    # get absolute index of center index
    p_0_x, p_0_y = np.meshgrid(range(self.padding, h + self.padding), range(self.padding, w + self.padding), indexing='ij')
    # print(np.shape(p_0_x)) #48 48
    # print(np.shape(p_0_y)) #48 48 

    p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1) # 1 3x3 48 48 
    p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1) # 1 3x3 48 48 
    p_0 = np.concatenate((p_0_x, p_0_y), axis=1) 
    p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)
    # print(p_0.size()) # 1 3x3+3x3 48 48 


    # get relative index around center pixel
    p_n_x, p_n_y = np.meshgrid(range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1),
                               range(-(self.ksize - 1) // 2, (self.ksize - 1) // 2 + 1), indexing='ij')
    # (2N, 1)
    p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
    p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
    p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False) # 1 2N 1 1
    p = p_0 + p_n
    p = p.repeat(b,1,1,1)
    # print('------get_index function------')

    return p

  def _get_x_q(self, x, q, N):
    # print('------get_x_q function------')

    b, h, w, _ = q.size()  # dimension of q: (b,h,w,2N)
    padded_w = x.size(3)
    c = x.size(1)
    # (b, c, h*padded_w)
    x = x.contiguous().view(b, c, -1)
    # (b, h, w, N)
    # index_x*w + index_y
    index = q[..., :N] * padded_w + q[...,N:] 

    # (b, c, h*w*N)
    index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
    x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

    # print('------get_x_q function------')
    return x_offset



class LPKPN_Module(nn.Module):
  def __init__(self, args, scale):
    super(LPKPN_Module, self).__init__()

    self.pixelconv = pixelConv(num_channels=args.num_filters, ksize=args.kernel_size)
    self.mean_shift = MeanShift([114.4, 111.5, 103.0], sign=1.0)
    self.first_conv = nn.Conv2d(in_channels=3, out_channels=args.num_filters, kernel_size=3, stride=1, padding=1)
    
    res_block_layers = []
    for i in range(args.num_blocks):
      res_block_layers.append(ResidualBlock(num_channels=args.num_filters))
    self.res_blocks = nn.Sequential(*res_block_layers)
    self.after_res_conv = nn.Conv2d(in_channels=args.num_filters, out_channels=args.num_filters, kernel_size=3, stride=1, padding=1)

    self.upsample = UpsampleBlock(scale=scale)
    self.final_conv_1 = nn.Conv2d(in_channels=args.num_filters//16, out_channels=args.num_filters, kernel_size=3, stride=1, padding=1)
    self.final_conv_2 = nn.Conv2d(in_channels=args.num_filters, out_channels=args.num_filters, kernel_size=3, stride=1, padding=1)

    self.mean_inverse_shift = MeanShift([114.4, 111.5, 103.0], sign=-1.0)
  
  def forward(self, x):
    # x = self.mean_shift(x)
    first_x = x 
    x = self.first_conv(x)

    res = self.res_blocks(x)
    res = self.after_res_conv(res)
    
    x = torch.add(x, res)
    x = self.upsample(x)
    x = self.final_conv_1(x)
    x = self.final_conv_2(x)

    bilinear_x = F.interpolate(first_x, scale_factor=4, mode='bilinear', align_corners=False)

    x = self.pixelconv(x, bilinear_x)
    # x = self.mean_inverse_shift(x)
    
    return x
    

