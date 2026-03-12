The code is tested with Python3.7, PyTorch == 1.11,0, CUDA == 11.3, mmdet3d == 0.15.0, mmcv_full == 1.6.2, mmdet == 2.28.2 and MinkowskiEngine == 0.5.4. We recommend you to use anaconda to make sure that all dependencies are in place. 

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name mmdet3d python=3.7 -y
conda activate mmdet3d
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```shell
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

**Step 3.** Install mmcv-full, mmdet and mmsegmentation.

```shell
pip install openmim
pip install chardet
mim install mmcv-full==1.6.2
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
```

**Step 4.** Following [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/get_started.md) to install mmdetection3d.

```shell
git clone https://github.com/LambdaGuard/TS3D.git
cd TS3D
export FORCE_CUDA="1"
pip install --no-cache-dir -e .
```

**Step 5.** Install MinkowskiEngine.
```shell
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=/opt/conda/include" --install-option="--blas=openblas"
```

**Step 6.** Install differentiable IoU.
```shell
git clone https://github.com/lilanxiao/Rotated_IoU /rotated_iou
cd /rotated_iou
git checkout 3bdca6b20d981dffd773507e97f1b53641e98d0a
cp -r /rotated_iou/cuda_op /TS3D/mmdet3d/ops/rotated_iou
cd /mmdetection3d/mmdet3d/ops/rotated_iou/cuda_op
python setup.py install
```