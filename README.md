# PyTorchSRBase
 
## Warning
현재 잘 작동하는 지 테스트 중이며, 아직 구현이 끝나지 않았습니다.

## Code snippets

### EDSR 학습 (DIV2K)
```shell
python train.py
  --dataloader=div2k_loader --data_input_path=c:\data\DIV2K_train_LR_bicubic --data_truth_path=c:\data\DIV2K_train_HR
  --model=edsr
  --batch_size=16 --input_patch_size=48 --scales=4 --max_steps=300000 --save_freq=50000
```
