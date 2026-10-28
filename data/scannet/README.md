# Prepare ScanNet Data for Indoor Detection

Most steps follow the
[MMDetection3D ScanNet guide](https://github.com/open-mmlab/mmdetection3d/blob/main/data/scannet/README.md).
IIFNet3D additionally generates mesh superpoints during data conversion.

## Generate point-cloud annotations and superpoints

Install
[segmentator](https://github.com/Karbo123/segmentator), which is used for
superpoint generation. Arrange the raw ScanNet files as follows:

```text
data/scannet/
├── scans/
├── scans_test/                 # Optional when test data is not used
└── meta_data/
    ├── scannetv2-labels.combined.tsv
    ├── scannet_train.txt
    └── scannetv2_test.txt
```

From the IIFNet3D repository root, run:

```bash
python data/scannet/batch_load_scannet_data_superpoint.py \
  --scannet-root data/scannet
```

The generated intermediate files are written to
`data/scannet/scannet_instance_data` by default. Use `--output-folder` to
choose another location. Run the following command to see all path overrides:

```bash
python data/scannet/batch_load_scannet_data_superpoint.py --help
```

Do not pass `--max-num-point`: IIFNet3D requires all points to remain aligned
with their superpoint labels.

## ScanNet V2 2D data
The ScanNet config expects the following layout. `scannet_2d` contains the
color images, depth maps, and camera poses exported from each ScanNet `.sens`
file. `scannet_fused` contains the per-point image features generated in the
next section.

[OpenScene preprocessing script](https://github.com/pengsongyou/openscene/blob/main/scripts/preprocess/preprocess_2d_scannet.py)
implements the same protocol. It is adapted from
[3DMV](https://github.com/angeladai/3DMV/blob/master/prepare_data/prepare_2d_data.py)
and ScanNet's official
[SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python).
Run it from the IIFNet3D root as follows:

```bash
git clone --depth 1 \
  https://github.com/pengsongyou/openscene.git third_party/openscene

python third_party/openscene/scripts/preprocess/preprocess_2d_scannet.py \
  --scannet_path /absolute/path/to/scannet/scans \
  --output_path data/scannet/scannet_2d \
  --frame_skip 20 \
  --output_image_width 320 \
  --output_image_height 240
```

Each raw scene directory must contain its matching `.sens` file before running
the command. The exporter processes all scene directories and preserves the
original frame IDs, producing names such as `0`, `20`, and `40`. Its output is
the `{color,depth,pose}` structure consumed by `scannet_rgb_extract.py`.

## Generate the cached ScanNet image features

Download the OpenMMLab ImVoteNet checkpoint used to initialize the ResNet-50 +
FPN image branch:

```bash
mkdir -p checkpoints
wget -P checkpoints \
  https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222-cad62aeb.pth
```

Generate one `.pt` file per scene:

```bash
CUDA_VISIBLE_DEVICES=0 python data/scannet/scannet_rgb_extract.py \
  --data_dir data/scannet \
  --output_dir data/scannet/scannet_fused \
  --model_path checkpoints/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222-cad62aeb.pth
```

To divide the work across processes, pass an inclusive scene-index range, for
example `--process_id_range 0,199`. Use a separate GPU per process and keep the
same output directory; existing scene files are skipped.
The final directory is:

```text
data/scannet/
├── points/
│   └── sceneXXXX_XX.bin
├── superpoints/
│   └── sceneXXXX_XX.bin
├── instance_mask/
│   └── sceneXXXX_XX.bin
├── semantic_mask/
│   └── sceneXXXX_XX.bin
├── scannet_2d/
│   └── sceneXXXX_XX/
│       ├── color/
│       ├── depth/
│       └── pose/
├── scannet_fused/
│   └── sceneXXXX_XX.pt
├── scannet_infos_train.pkl
└── scannet_infos_val.pkl
```
