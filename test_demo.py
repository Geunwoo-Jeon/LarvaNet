import os.path
import argparse
import importlib
import logging
import time
from collections import OrderedDict
import torch

from utils import utils_logger
from utils import utils_image as util


def main():
    # logger set up
    utils_logger.logger_info('AIM-track', log_path='AIM-track_msrr.log')
    logger = logging.getLogger('AIM-track')

    # --------------------------------
    # basic settings
    # --------------------------------

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataloader', type=str, default='div2k_loader', help='Name of the data loader.')
    parser.add_argument('--model', type=str, default='edsr', help='Name of the model.')

    parser.add_argument('--scales', type=str, default='4',
                        help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
    parser.add_argument('--cuda_device', type=str, default='-1',
                        help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

    parser.add_argument('--restore_path', type=str, required=True,
                        help='Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
    parser.add_argument('--restore_target', type=str, help='Target of the restoration.')
    parser.add_argument('--restore_global_step', type=int, default=0,
                        help='Global step of the restored model. Some models may require to specify this.')

    parser.add_argument('--save_path', type=str,
                        help='Base output path of the upscaled images. Specify this to save the upscaled images.')

    args, remaining_args = parser.parse_known_args()

    # initialize
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    scale_list = list(map(lambda x: int(x), args.scales.split(',')))

    # model
    print('prepare model - %s' % (args.model))
    MODEL_MODULE = importlib.import_module('models.' + args.model)
    model = MODEL_MODULE.create_model()
    _, remaining_args = model.parse_args(remaining_args)
    model.prepare(is_training=False, scales=scale_list, global_step=args.restore_global_step)

    # check remaining args
    if (len(remaining_args) > 0):
        print('WARNING: found unhandled arguments: %s' % (remaining_args))

    # model > restore
    model.restore(ckpt_path=args.restore_path, target=args.restore_target)
    print('restored the model')

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = 'C:/aim2020/data/DIV2K_valid_LR_bicubic'
    E_folder = 'C:/aim2020/edsrb/result/300k/x4(2)'

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        start.record()
        img_E = model.model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)

        # util.imsave(img_E, os.path.join(E_folder, img_name+ext))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

if __name__ == '__main__':

    main()