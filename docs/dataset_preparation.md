# Dataset Preparation

- We mainly follow the MMDetection3D data preparation protocols for
  [ScanNet](../data/scannet/README.md) and
  [SUN RGB-D](../data/sunrgbd/README.md), including superpoint generation.
  IIFNet3D-specific changes for generating the `pkl` files are implemented in
  [scannet_data_utils](../tools/data_converter/scannet_data_utils.py) and
  [sunrgbd_data_utils](../tools/data_converter/sunrgbd_data_utils.py).

- Superpoint generation requires many dependencies, so we recommend using the
  preprocessed superpoints and `pkl` files provided for IIFNet3D. Download them
  from [Baidu Drive](https://pan.baidu.com/s/1fCf4nMq__xh-XGUnQqql4A?pwd=vdte),
  place them in the corresponding dataset directories, and replace the original
  `pkl` files.

- Do not sample points during data preparation. Sampling changes the point order
  and causes a mismatch with the provided superpoint labels.
