# PyTorchSRBase
 
AIM2020 Challenge를 위한 Super-Resolution 모델을 구현할 수 있는 PyTorch 기반 코드 Repository 입니다.


## Code snippets

### EDSR-baseline 학습 (DIV2K)
```shell
python train.py
  --dataloader=div2k_loader --data_input_path=c:\data\DIV2K_train_LR_bicubic --data_truth_path=c:\data\DIV2K_train_HR
  --model=edsr
  --batch_size=16 --input_patch_size=48 --scales=4 --max_steps=300000 --save_freq=50000
  --tran_path=d:\tmp\aim2020\edsrb
```

### EDSR-baseline 학습 과정 살펴보기
```shell
tensorboard --host=127.0.0.1 --logdir=d:\tmp\aim2020\edsrb
```
브라우저에서 ```127.0.0.1:6006``` 접속

### EDSR-baseline 검증 (DIV2K)
```shell
python validate.py
  --dataloader=div2k_loader --data_input_path=c:\data\DIV2K_val_LR_bicubic --data_truth_path=c:\data\DIV2K_val_HR
  --model=edsr
  --restore_path=d:\tmp\aim2020\edsrb\model_300000.pth
  --save_path=d:\tmp\aim2020\edsrb\results\300k\
```
