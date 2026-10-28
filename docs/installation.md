# Installation

We recommend using Conda to keep the dependency versions consistent.

## Environment

| Package | Version |
| --- | --- |
| Python | 3.8.13 |
| CUDA | 11.1 |
| cuDNN | 8.0.5 |
| PyTorch | 1.10.1+cu111 |
| TorchVision | 0.11.2+cu111 |
| MMCV-Full | 1.6.0 |
| MMDetection | 2.24.1 |
| MMSegmentation | 0.24.1 |
| MMDetection3D | 1.0.0rc3 |
| MinkowskiEngine | 0.5.4 |
| torch-scatter | 2.0.9 |
| NumPy | 1.21.5 |

An exported snapshot of the development environment is provided in
[`our_env.yaml`](../our_env.yaml) for reference. On a compatible Linux and
CUDA system, you can try to create the environment with:

```bash
conda env create -f our_env.yaml
conda activate iifnet
```

Alternatively, follow the steps below to install the dependencies manually.

## Installation steps

**Step 1. Create and activate the environment.**

```bash
conda create -n iifnet python=3.8.13 -y
conda activate iifnet
python -m pip install numpy==1.21.5
```

**Step 2. Install PyTorch.**

```bash
python -m pip install \
  torch==1.10.1+cu111 \
  torchvision==0.11.2+cu111 \
  --extra-index-url https://download.pytorch.org/whl/cu111
```

**Step 3. Install the OpenMMLab dependencies and torch-scatter.**

```bash
python -m pip install \
  mmcv-full==1.6.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

python -m pip install mmdet==2.24.1 mmsegmentation==0.24.1

python -m pip install \
  torch-scatter==2.0.9 \
  -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
```

**Step 4. Install MinkowskiEngine.**

```bash
conda install -c conda-forge openblas-devel -y
git clone --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas=openblas --force_cuda
cd ..
```

**Step 5. Install IIFNet3D.**

```bash
git clone https://github.com/Cherryreg/IIFNet3D.git
cd IIFNet3D
python -m pip install -v -e .
```
