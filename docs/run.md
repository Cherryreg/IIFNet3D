**Training**

To start training, run [dist_train](../tools/dist_train.sh) with `TS3D` configs.  

For the localization setting on ScanNet, run
```shell
bash tools/dist_train.sh configs/ts3d/ts3d_localization.py 8 --work-dir  work_dirs/ts3d_localization
```

For the classification setting on ScanNet, run
```shell
bash tools/dist_train.sh configs/ts3d/ts3d_classification_1st.py 8 --work-dir  work_dirs/ts3d_classification/1st
bash tools/dist_train.sh configs/ts3d/ts3d_classification_2st.py 8 --work-dir  work_dirs/ts3d_classification/2st
```

For the unsupervised setting on ScanNet-25k, run
```shell
bash tools/dist_train.sh configs/ts3d/ts3d_single_view_1st.py 8 --work-dir  work_dirs/ts3d_single_view/1st
bash tools/dist_train.sh configs/ts3d/ts3d_single_view_2st.py 8 --work-dir  work_dirs/ts3d_single_view/2st
bash tools/dist_train.sh configs/ts3d/ts3d_single_view_3st.py 8 --work-dir  work_dirs/ts3d_single_view/3st
```

**Evaluation**

To evaluate the model, run [dist_test](../tools/dist_test.sh) with `TS3D` configs.

For the localization setting on ScanNet, run
```shell
bash tools/dist_test.sh configs/ts3d/ts3d_localization.py 8 work_dirs/ts3d_localization/latest.pth --eval mAP
```

For the classification setting on ScanNet, run
```shell
bash tools/dist_test.sh configs/ts3d/ts3d_classification_2st.py 8 work_dirs/ts3d_classification/2st/latest.pth --eval mAP
```

For the unsupervised setting on ScanNet-25k, run
```shell
bash tools/dist_train.sh configs/ts3d/ts3d_single_view_3st.py 8 work_dirs/ts3d_single_view/3st/latest.pth --eval mAP
```

For the dataset transfer setting on SUNRGBD, run
```shell
bash tools/dist_train.sh configs/ts3d/ts3d_single_view_3st_sunrgbd.py 8 work_dirs/ts3d_single_view/3st/latest.pth --eval mAP
```
