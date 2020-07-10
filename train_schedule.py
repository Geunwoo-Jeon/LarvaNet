import argparse
import importlib
import json
import os
import time

import dataloaders
import models
import validate

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from math import log10, floor


def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))


def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataloader', type=str, default='div2k_train_loader', help='Name of the data loader.')
    parser.add_argument('--dataloader_val', type=str, default='div2k_val_loader', help='Name of the data loader.')
    parser.add_argument('--model', type=str, default='edsr', help='Name of the model.')

    parser.add_argument('--batch_size', type=int, default=16, help='Size of the batches for each training step.')
    parser.add_argument('--input_patch_size', type=int, default=48, help='Size of each input image patch.')
    parser.add_argument('--step_per_epoch', type=float, help='Num of steps on 1 epoch.')
    parser.add_argument('--scales', type=str, default='4',
                        help='Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

    parser.add_argument('--train_path', type=str, default='c:/aim2020/edsrb/train/',
                        help='Base path of the trained model to be saved.')
    parser.add_argument('--max_steps', type=int, default=300000, help='The maximum number of training steps.')
    parser.add_argument('--log_freq', type=int, default=10, help='The frequency of logging.')
    parser.add_argument('--summary_freq', type=int, default=1000, help='The frequency of logging on TensorBoard.')
    parser.add_argument('--save_freq', type=int, default=1000, help='The frequency of saving the trained model.')
    parser.add_argument('--sleep_ratio', type=float, default=0.05,
                        help='The ratio of sleeping time for each training step, which prevents overheating of GPUs. Specify 0 to disable sleeping.')

    parser.add_argument('--restore_path', type=str,
                        help='Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
    parser.add_argument('--restore_target', type=str, help='Target of the restoration.')
    parser.add_argument('--global_step', type=int, default=0,
                        help='Initial global step. Specify this to resume the training.')


    args, remaining_args = parser.parse_known_args()

    # initialize
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    scale_list = list(map(lambda x: int(x), args.scales.split(',')))
    os.makedirs(args.train_path, exist_ok=True)

    # data loader
    print('prepare data loader - %s' % (args.dataloader))
    DATALOADER_MODULE = importlib.import_module('dataloaders.' + args.dataloader)
    dataloader = DATALOADER_MODULE.create_loader()
    dataloader.prepare(scales=scale_list)

    DATALOADER_MODULE = importlib.import_module('dataloaders.' + args.dataloader_val)
    dataloader_val = DATALOADER_MODULE.create_loader()
    dataloader_val.prepare(scales=scale_list)

    # model
    print('prepare model - %s' % (args.model))
    MODEL_MODULE = importlib.import_module('models.' + args.model)
    model = MODEL_MODULE.create_model()
    model_args, remaining_args = model.parse_args(remaining_args)
    model.prepare(is_training=True, scales=scale_list, global_step=args.global_step)

    # check remaining args
    if (len(remaining_args) > 0):
        print('WARNING: found unhandled arguments: %s' % (remaining_args))

    # model > restore
    if (args.restore_path is not None):
        model.restore(ckpt_path=args.restore_path, target=args.restore_target)
        print('restored the model')

    # model > summary
    summary_writers = {}
    for scale in scale_list:
        summary_path = os.path.join(args.train_path, 'x%d' % (scale))
        summary_writer = SummaryWriter(log_dir=summary_path)
        summary_writers[scale] = summary_writer

    # save arguments
    arguments_path = os.path.join(args.train_path, 'arguments.json')
    all_args = {**vars(args), **vars(model_args)}
    with open(arguments_path, 'w') as f:
        f.write(json.dumps(all_args, sort_keys=True, indent=2))


    # start fetching data
    if (dataloader.is_threaded):
        dataloader.start_training_queue_runner(batch_size=args.batch_size, input_patch_size=args.input_patch_size)


    if args.step_per_epoch is None:
        batch_data_size = (args.input_patch_size**2)*args.batch_size*3
        dataset_size = 300*(1024**2)
        step_per_epoch = round_to_1(dataset_size/batch_data_size)
    else:
        step_per_epoch = args.step_per_epoch

    # train
    print('begin training')
    print(f'{step_per_epoch} steps equal to 1 epoch')
    try:
        while True:
            model.global_step += 1

            scale = model.get_next_train_scale()
            summary = summary_writers[scale] if (model.global_step % args.summary_freq == 0) else None

            start_time = time.time()
            if (dataloader.is_threaded):
                input_list, truth_list = dataloader.get_queue_data(scale=scale)
            else:
                input_list, truth_list = dataloader.get_patch_batch(batch_size=args.batch_size, scale=scale,
                                                                    input_patch_size=args.input_patch_size)
            dataload_time = time.time() - start_time

            # numpy to torch
            start_time = time.time()
            input_tensor = torch.as_tensor(input_list, dtype=torch.float32, device=model.device)
            truth_tensor = torch.as_tensor(truth_list, dtype=torch.float32, device=model.device)
            np2ts_time = time.time() - start_time

            # get SR and calculate loss
            start_time = time.time()
            output_tensor = model.model(input_tensor)
            loss = model.loss_fn(output_tensor, truth_tensor)
            fwd_loss_time = time.time() - start_time

            # do back propagation
            start_time = time.time()
            model.optim.zero_grad()
            loss.backward()
            model.optim.step()
            back_prop_time = time.time() - start_time

            # scheduling lr by validation
            if model.global_step % step_per_epoch == 0:
                print('begin validation')
                num_images = dataloader_val.get_num_images()
                psnr_list = []

                for image_index in range(num_images):
                    input_image, truth_image, image_name = dataloader_val.get_image_pair(image_index=image_index,
                                                                                        scale=scale)
                    output_image = model.upscale(input_list=[input_image], scale=scale)[0]
                    truth_image = validate._image_to_uint8(truth_image)
                    output_image = validate._image_to_uint8(output_image)
                    truth_image = validate._fit_truth_image_size(output_image=output_image, truth_image=truth_image)

                    psnr = validate._image_psnr(output_image=output_image, truth_image=truth_image)
                    psnr_list.append(psnr)

                average_psnr = np.mean(psnr_list)
                print(f'step {model.global_step}, epoch {model.global_step/step_per_epoch:.0f}, psnr={average_psnr:.8f}, lr = {model.get_lr():.10f}')
                model.lr_scheduler.step(average_psnr)

                # save model after each validation
                model.save(base_path=args.train_path)
                print('saved a model checkpoint at step %d' % model.global_step)

            # write summary
            lr = model.get_lr()
            if summary is not None:
                summary.add_scalar('loss', loss, model.global_step)
                summary.add_scalar('lr', lr, model.global_step)

                output_tensor_uint8 = output_tensor.clamp(0, 255).byte()
                for i in range(min(4, len(input_list))):
                    summary.add_image('input/%d' % i, input_list[i], model.global_step)
                    summary.add_image('output/%d' % i, output_tensor_uint8[i, :, :, :], model.global_step)
                    summary.add_image('truth/%d' % i, truth_list[i], model.global_step)

            duration = time.time() - start_time
            if args.sleep_ratio > 0 and duration > 0:
                time.sleep(min(10.0, duration * args.sleep_ratio))

            if model.global_step < step_per_epoch and model.global_step % args.log_freq == 0:
                print('step %d, lr %.10f, loss %.6f (%.3f sec/batch)' % (model.global_step, lr, loss, duration))
                print(f'dataload_time:{dataload_time:.4f}s, np2ts_time:{np2ts_time:.4f}s, '
                    f'fwd_loss_time: {fwd_loss_time:.4f}s, back_prop_time: {back_prop_time}s')
    except KeyboardInterrupt:
        print('interrupted (KeyboardInterrupt)')
    except Exception as e:
        print(e)

    # finalize
    print('finished')
    for scale in scale_list:
        summary_writers[scale].close()
    if (dataloader.is_threaded):
        dataloader.stop_queue_runners()


if __name__ == '__main__':
    main()
