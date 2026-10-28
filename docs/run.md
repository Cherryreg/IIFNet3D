# Training, evaluation, and visualization

Run all commands from the IIFNet3D repository root. The config files import the
custom modules from `projects/IIFNet`, so invoking the tools from an unrelated
working directory can cause plugin import errors.

## Training

### Distributed training

```bash
# ScanNet V2 on four GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash tools/dist_train.sh projects/configs/IIFNet_scannet.py 4 \
  --work-dir work_dirs/iifnet_scannet

# SUN RGB-D on two GPUs
CUDA_VISIBLE_DEVICES=0,1 \
  bash tools/dist_train.sh projects/configs/IIFNet_sunrgbd.py 2 \
  --work-dir work_dirs/iifnet_sunrgbd
```

The second positional argument to `dist_train.sh` is the number of visible GPUs,
not a GPU index. It must match the number of entries in
`CUDA_VISIBLE_DEVICES`.

### Single-GPU training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  projects/configs/IIFNet_scannet.py \
  --work-dir work_dirs/iifnet_scannet_1gpu
```

### Resume training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash tools/dist_train.sh projects/configs/IIFNet_scannet.py 4 \
  --work-dir work_dirs/iifnet_scannet \
  --resume-from work_dirs/iifnet_scannet/latest.pth
```

## Checkpoints

Create a local checkpoint directory and place downloaded IIFNet3D weights there:

```bash
mkdir -p checkpoints
```

Download the released weights using the links in
[Main Results](../README.md#main-results).

The SUN RGB-D config initializes its image branch from the public ImVoteNet
checkpoint hosted by OpenMMLab. MMCV downloads and caches it automatically on
the first training run. An offline copy can instead be selected with:

```bash
python tools/train.py projects/configs/IIFNet_sunrgbd.py \
  --work-dir work_dirs/iifnet_sunrgbd \
  --cfg-options load_from=/absolute/path/to/imvotenet_pretrain.pth
```

## Evaluation

### Single GPU

```bash
# ScanNet V2
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  projects/configs/IIFNet_scannet.py \
  checkpoints/iifnet3d_scannet_epoch_20.pth \
  --eval mAP

# SUN RGB-D
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  projects/configs/IIFNet_sunrgbd.py \
  checkpoints/iifnet3d_sunrgbd.pth \
  --eval mAP
```

### Multiple GPUs

The positional arguments to `dist_test.sh` are `CONFIG`, `CHECKPOINT`, and
`GPUS`, in that order.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash tools/dist_test.sh \
  projects/configs/IIFNet_scannet.py \
  checkpoints/iifnet3d_scannet_epoch_20.pth \
  4 --eval mAP
```

## Visualization

Use `--show-dir` to save the predictions. For less cluttered figures, set
`model.test_cfg.score_thr=0.25` at the command line.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  projects/configs/IIFNet_scannet.py \
  checkpoints/iifnet3d_scannet_epoch_20.pth \
  --eval mAP \
  --show \
  --show-dir work_dirs/iifnet_scannet/visualization \
  --cfg-options model.test_cfg.score_thr=0.25
```
