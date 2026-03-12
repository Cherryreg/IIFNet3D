import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import torch.nn as nn
from mmdet3d.models import build_backbone, build_neck
# import tensorflow as tf2
# import tensorflow.compat.v1 as tf
from os.path import join, exists
from fusion_util import extract_img_feature, PointCloudToImageMapper, save_fused_feature_v2, adjust_intrinsic, make_intrinsic


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='train', help='split: "train"| "val"')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')
    parser.add_argument('--model_path', type=str, default='', help='model path')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    scene_id = data_path.split('/')[-1].split('.bin')[0]

    num_rand_file_per_scene = 1
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    model = args.model
    # model_path = args.model_path
    # text_emb = args.text_emb
    keep_features_in_memory = args.keep_features_in_memory

    # load 3D data (point cloud)，从bin文件中加载数据---
    point_cloud = np.fromfile(data_path, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 6)[:,:3]
    locs_in = torch.tensor(point_cloud, dtype=torch.float32)
    
    #locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    n_interval = num_rand_file_per_scene
    n_finished = 0
    for n in range(n_interval):

        if exists(join(out_dir, scene_id +'.pt')):
            n_finished += 1
            print(scene_id +'.pt' + ' already done!')
            continue
    if n_finished == n_interval:
        return 1

    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # extract image features and keep them in the memory
    # default: False (extract image on the fly)
    if keep_features_in_memory and model is not None:
        img_features = []
        for img_dir in tqdm(img_dirs):
            img_features.append(extract_img_feature(img_dir, model, img_size=[240, 320]))

    n_points_cur = n_points
    # counter = torch.zeros((n_points_cur, 1), device=device)
    # sum_features = torch.zeros((n_points_cur, feat_dim), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    ###--added
    feat_list_per_point = [[] for _ in range(n_points_cur)]  # 每个点一个列表收集其在可见帧的特征
    
    for img_id, img_dir in enumerate(tqdm(img_dirs)):
        # load pose
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        # load depth and convert to meter
        depth = imageio.v2.imread(img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3] #提取有效性标志，表示哪些点在当前图像中是可见的
        vis_id[:, img_id] = mask #记录哪些点在当前图像中是可见的
        if keep_features_in_memory:
            feat_2d = img_features[img_id].to(device)
        else:
            #feat_2d = extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=[240, 320]).to(device)
            feat_2d = extract_img_feature(img_dir, model, img_size=[240, 320]).to('cpu')
            

        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)
        
        for idx in torch.where(mask != 0)[0]:
            feat_list_per_point[idx.item()].append(feat_2d_3d[idx])

    # 对于每个点而言，随机选一帧的特征
    feat_bank = torch.zeros(n_points, feat_2d_3d.shape[1], device=device)
    for i, feat_list in enumerate(feat_list_per_point):
        if feat_list:
            rand_idx = np.random.choice(len(feat_list)) 
            rand_feat = feat_list[rand_idx]
            feat_bank[i] = rand_feat
         
        # y_coords = mapping[:, 1]  # 第二列是 y 坐标
        # x_coords = mapping[:, 2]  # 第三列是 x 坐标

        # 将坐标合并成一个二维数组并保存到文件
        # coordinates = np.column_stack((x_coords, y_coords))  # 合并为 (x, y) 形式的数组
        # np.savetxt('coordinates00.txt', coordinates, fmt='%d', delimiter=',', header='x,y', comments='')
        
    #     counter[mask!=0]+= 1
    #     sum_features[mask!=0] += feat_2d_3d[mask!=0]

    # counter[counter==0] = 1e-5
    # feat_bank = sum_features/counter   #[n, feat_dim]
    #将坐标与特征拼接
    locs_in = locs_in.to(device=device) 
    feat_bank = torch.cat((locs_in,feat_bank), dim=1)
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    save_fused_feature_v2(feat_bank, point_ids, n_points, out_dir, scene_id, args)

class ImVoteNetBackboneNeck(nn.Module):
    def __init__(self, backbone, neck):
        super(ImVoteNetBackboneNeck, self).__init__()
        self.img_backbone = build_backbone(backbone)
        self.img_neck = build_neck(neck)

    def forward(self, img):
        with torch.no_grad():
            x = self.img_backbone(img)
            img_features = self.img_neck(x)[0]
        return img_features
    

#no use
class ImVoxelNetBackboneNeck(nn.Module):
    def __init__(self, backbone, neck):
        super(ImVoxelNetBackboneNeck, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)

    def forward(self, img):
        with torch.no_grad():
            x = self.backbone(img)
            img_features = self.neck(x)[0]
        return img_features  

def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #!### Dataset specific parameters #####
    img_dim = (80, 60) #final img_dim after extract
    depth_scale = 1000.0
    fx = 577.870605
    fy = 577.870605
    mx=319.5
    my=239.5
    #######################################
    visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 0 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 256 # imvoxel feature dimension
    
    model_config = dict(
    # type='TR3DFF3DDetector',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))
    
    
    backbone_cfg = model_config['img_backbone']
    neck_cfg = model_config['img_neck']
    model = ImVoteNetBackboneNeck(backbone =backbone_cfg, neck=neck_cfg)
    checkpoint_path = args.model_path
    # device = 'cpu'
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'],strict=False)
    model = model.cuda()
    model.eval()
    
    args.model = model

    # split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'points')
    data_root_2d = join(data_dir,'scannet_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range
    
    args.n_split_points = float('inf') #设置为无穷大禁用这个特征
    args.num_rand_file_per_scene = 1


    # calculate image pixel-3D points correspondances
    intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
    intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim=[640, 480], image_dim=img_dim)


    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsic,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root,'*.bin')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        process_one_scene(data_paths[i], out_dir, args)



if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)

