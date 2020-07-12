import argparse
import copy
import os

import torch
import numpy as np
import cv2 as cv

from .base import BaseLoader


# DIV2K dataset loader

def create_loader():
    return DIV2KLoader()


class DIV2KLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_input_path', type=str, default='c:/aim2020/data/DIV2K_train_LR_bicubic',
                            help='Base path of the input images.')
        parser.add_argument('--train_truth_path', type=str, default='c:/aim2020/data/DIV2K_train_HR',
                            help='Base path of the ground-truth images.')
        parser.add_argument('--data_cached', action='store_true', help='If true, cache the data on the memory.')

        self.args, remaining_args = parser.parse_known_args(args=args)
        return copy.deepcopy(self.args), remaining_args

    def prepare(self, scales):
        self.scale = scales[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # retrieve image name list
        self.image_name_list = [os.path.splitext(f)[0] for f in os.listdir(self.args.train_truth_path) if f.lower().endswith('.png')]
        self.num_images = len(self.image_name_list)

        # make image list
        self.input_image_list = []
        self.truth_image_list = []
        for i, image_name in enumerate(self.image_name_list):
            image_path = os.path.join(self.args.train_input_path, ('X%d' % self.scale),
                                      ('%sx%d.png' % (image_name, self.scale)))
            input_image = self._load_image(image_path)
            input_image = torch.tensor(input_image, dtype=torch.float32)
            self.input_image_list.append(input_image)

            image_path = os.path.join(self.args.train_truth_path, ('%s.png' % image_name))
            truth_image = self._load_image(image_path)
            truth_image = torch.tensor(truth_image, dtype=torch.float32)
            self.truth_image_list.append(truth_image)
        print('data: %d images are prepared (%s)' % (len(self.image_name_list), 'caching enabled'))

    def get_patch_batch(self, batch_size, scale, input_patch_size):
        input_tensor = torch.empty((batch_size, 3, input_patch_size, input_patch_size))
        truth_tensor = torch.empty((batch_size, 3, input_patch_size*scale, input_patch_size*scale))

        for i in range(batch_size):
            image_index = np.random.randint(self.num_images)
            input_patch, truth_patch = self.get_image_patch_pair(image_index=image_index, scale=scale,
                                                                 input_patch_size=input_patch_size)
            input_tensor[i] = input_patch
            truth_tensor[i] = truth_patch

        return input_tensor, truth_tensor

    def get_image_patch_pair(self, image_index, scale, input_patch_size):
        # retrieve image
        input_image = self.input_image_list[image_index]
        truth_image = self.truth_image_list[image_index]

        # randomly crop
        truth_patch_size = input_patch_size * scale
        _, height, width = input_image.shape
        input_x = np.random.randint(width - input_patch_size)
        input_y = np.random.randint(height - input_patch_size)
        truth_x = input_x * scale
        truth_y = input_y * scale
        input_patch = input_image[:, input_y:(input_y + input_patch_size), input_x:(input_x + input_patch_size)]
        truth_patch = truth_image[:, truth_y:(truth_y + truth_patch_size), truth_x:(truth_x + truth_patch_size)]

        # randomly rotate
        rot90_k = np.random.randint(4) + 1
        input_patch = torch.rot90(input_patch, k=rot90_k, dims=(1, 2))
        truth_patch = torch.rot90(truth_patch, k=rot90_k, dims=(1, 2))

        # randomly flip
        flip = (np.random.uniform() < 0.5)
        if (flip):
            input_patch = torch.flip(input_patch, (2,))
            truth_patch = torch.flip(truth_patch, (2,))

        # finalize
        return input_patch, truth_patch

    def _load_image(self, path):
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.transpose(image, [2, 0, 1])
        return image
