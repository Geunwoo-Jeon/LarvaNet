# PyTorchSRBase
 
## Warning
현재 잘 작동하는 지 테스트 중이며, 아직 구현이 끝나지 않았습니다.

## Code snippets

### EDSR-baseline 학습 (DIV2K)
```shell
python train.py
  --dataloader=div2k_loader --data_input_path=c:\data\DIV2K_train_LR_bicubic --data_truth_path=c:\data\DIV2K_train_HR
  --model=edsr
  --batch_size=16 --input_patch_size=48 --scales=4 --max_steps=300000 --save_freq=50000
  --tran_path=d:\tmp\aim2020\edsrb
```

### EDSR-baseline 검증 (DIV2K)
```shell
python validate.py
  --dataloader=div2k_loader --data_input_path=c:\data\DIV2K_val_LR_bicubic --data_truth_path=c:\data\DIV2K_val_HR
  --model=edsr
  --restore_path=d:\tmp\aim2020\edsrb\model_300000.ptb
  --save_path=d:\tmp\aim2020\edsrb\results\300k\
```
