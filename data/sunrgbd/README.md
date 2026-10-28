# Prepare SUN RGB-D Data

Most preparation steps follow the
[MMDetection3D SUN RGB-D guide](https://github.com/open-mmlab/mmdetection3d/blob/main/data/sunrgbd/README.md).

## Generate superpoints with VCCS

1. Install [pclpy](https://github.com/davidcaron/pclpy):

```bash
# Python 3.8.16
conda install -c conda-forge/label/gcc7 qhull
conda install -c conda-forge -c davidcaron pclpy
```

2. From the IIFNet3D repository root, generate the superpoint labels:

```bash
python data/sunrgbd/slic_function_multi_processing.py \
    --points-dir data/sunrgbd/points \
    --output-dir data/sunrgbd/superpoints
```

The script uses all available CPU cores by default. Use `--workers N` to set
the number of processes. Existing output files are skipped; pass `--overwrite`
to regenerate them.

To check one generated file and optionally visualize it:

```bash
python data/sunrgbd/check_function.py \
    data/sunrgbd/points/006887.bin \
    data/sunrgbd/superpoints/006887.bin \
    --visualize
```

Alternatively, you can use
[SPG/SSP](https://github.com/loicland/superpoint_graph) to generate
superpoints. We use VCCS in our experiments.


The directory structure after preprocessing should be:

```text
sunrgbd
├── README.md
├── matlab
│   ├── extract_rgbd_data_v1.m
│   ├── extract_rgbd_data_v2.m
│   └── extract_split.m
├── OFFICIAL_SUNRGBD
│   ├── SUNRGBD
│   ├── SUNRGBDMeta2DBB_v2.mat
│   ├── SUNRGBDMeta3DBB_v2.mat
│   └── SUNRGBDtoolbox
├── sunrgbd_trainval
│   ├── calib
│   ├── depth
│   ├── image
│   ├── label
│   ├── label_v1
│   ├── seg_label
│   ├── train_data_idx.txt
│   └── val_data_idx.txt
├── points
├── superpoints
├── sunrgbd_infos_train.pkl
└── sunrgbd_infos_val.pkl
```
