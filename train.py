import argparse
import importlib
import json
import os
import time

import dataloaders
import models

import torch
from torch.utils.tensorboard import SummaryWriter

def main():
  # parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataloader', type=str, default='div2k_train_loader', help='Name of the data loader.')
  parser.add_argument('--model', type=str, default='edsr', help='Name of the model.')

  parser.add_argument('--batch_size', type=int, default=16, help='Size of the batches for each training step.')
  parser.add_argument('--input_patch_size', type=int, default=48, help='Size of each input image patch.')
  parser.add_argument('--scales', type=str, default='4', help='Scale of the input images.')
  parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  parser.add_argument('--train_path', type=str, default='c:/aim2020/edsrb/train/', help='Base path of the trained model to be saved.')
  parser.add_argument('--max_steps', type=int, default=300000, help='The maximum number of training steps.')
  parser.add_argument('--log_freq', type=int, default=10, help='The frequency of logging.')
  parser.add_argument('--summary_freq', type=int, default=1000, help='The frequency of logging on TensorBoard.')
  parser.add_argument('--save_freq', type=int, default=10000, help='The frequency of saving the trained model.')
  parser.add_argument('--sleep_ratio', type=float, default=0.05, help='The ratio of sleeping time for each training step, which prevents overheating of GPUs. Specify 0 to disable sleeping.')

  parser.add_argument('--restore_path', type=str, help='Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
  parser.add_argument('--restore_target', type=str, help='Target of the restoration.')
  parser.add_argument('--global_step', type=int, default=0, help='Initial global step. Specify this to resume the training.')

  args, remaining_args = parser.parse_known_args()


  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
  scale_list = list(map(lambda x: int(x), args.scales.split(',')))
  os.makedirs(args.train_path, exist_ok=True)

  # data loader
  print('prepare data loader - %s' % (args.dataloader))
  DATALOADER_MODULE = importlib.import_module('dataloaders.' + args.dataloader)
  dataloader = DATALOADER_MODULE.create_loader()
  dataloader_args, remaining_args = dataloader.parse_args(remaining_args)
  dataloader.prepare(scales=scale_list)

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

  # train
  print('begin training')
  local_train_step = 0
  while (model.global_step < args.max_steps):
    global_train_step = model.global_step + 1
    local_train_step += 1

    start_time = time.time()

    scale = model.get_next_train_scale()
    summary = summary_writers[scale] if (local_train_step % args.summary_freq == 0) else None
    input_list, truth_list = dataloader.get_patch_batch(batch_size=args.batch_size, scale=scale, input_patch_size=args.input_patch_size)
    input_tensor = torch.as_tensor(input_list, dtype=torch.float32, device=model.device)
    truth_tensor = torch.as_tensor(truth_list, dtype=torch.float32, device=model.device)
    loss = model.train_step(input_list=input_tensor, scale=scale, truth_list=truth_tensor, summary=summary)

    duration = time.time() - start_time
    if (args.sleep_ratio > 0 and duration > 0):
      time.sleep(min(10.0, duration*args.sleep_ratio))

    if (local_train_step % args.log_freq == 0):
      print('step %d, lr %f, loss %.6f (%.3f sec/batch)' % (global_train_step, model.optim.param_groups[0]['lr'], loss, duration))
    
    if (local_train_step % args.save_freq == 0):
      model.save(base_path=args.train_path)
      print('saved a model checkpoint at step %d' % (global_train_step))

    
  # finalize
  print('finished')
  for scale in scale_list:
    summary_writers[scale].close()



if __name__ == '__main__':
  main()