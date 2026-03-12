import os
import torch
import glob
import math
import numpy as np
# from tensorflow import io
# import tensorflow.compat.v1 as tf
import cv2
import torch.nn as nn
import mmcv
from mmdet.datasets.pipelines import LoadImageFromFile
import time

# def read_bytes(path):
#     '''Read bytes for OpenSeg model running.'''

#     with io.gfile.GFile(path, 'rb') as f:
#         file_bytes = f.read()
#     return file_bytes


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


# def extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=None, regional_pool=True):
#     '''Extract per-pixel OpenSeg features.'''

#     # load RGB image
#     np_image_string = read_bytes(img_dir)
#     # run OpenSeg
#     results = openseg_model.signatures['serving_default'](
#             inp_image_bytes=tf.convert_to_tensor(np_image_string),
#             inp_text_emb=text_emb)
#     img_info = results['image_info']
#     crop_sz = [
#         int(img_info[0, 0] * img_info[2, 0]),
#         int(img_info[0, 1] * img_info[2, 1])
#     ]
#     if regional_pool:
#         image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
#     else:
#         image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
#     if img_size is not None:
#         feat_2d = tf.cast(tf.image.resize_nearest_neighbor(
#             image_embedding_feat, img_size, align_corners=True)[0], dtype=tf.float16).numpy()
#     else:
#         feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()

#     feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)

#     return feat_2d

#!important
def read_image(img_dir, img_size=None):
    # loader = LoadImageFromFile(to_float32=False, color_type='color', channel_order='bgr')
    file_client = mmcv.FileClient(backend='disk')

    filename = img_dir
    img_bytes = file_client.get(filename)
    image = mmcv.imfrombytes(img_bytes, flag='color', channel_order='bgr')
    # results = loader(results)
    
    image_test = cv2.imread(img_dir)
    if img_size is not None:
        image = cv2.resize(image, (img_size[1], img_size[0]))  
    img_norm_cfg = dict(mean=[103.53, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    #std=[57.375, 57.12, 58.395],
    # image= mmcv.imnormalize(image, mean=[103.53, 116.28, 123.675],
    #                      std=[57.375, 57.12, 58.395], to_rgb=False)
    
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)
    to_rgb = img_norm_cfg['to_rgb']
    image = mmcv.imnormalize(image,mean,std,to_rgb)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转换为RGB
    return image

def preprocess_image(image):
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 调整维度顺序为(C, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
    return image_tensor


def extract_img_feature(img_dir, model, img_size=None):
    '''Extract pixel features.'''
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载并预处理图像
    image = read_image(img_dir, img_size)
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    
    
    # # 禁用梯度计算，节省内存和加速推理
    # with torch.no_grad():
    #计算模型运行时间
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    feat_2d = model(image_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"Image feature extraction time: {elapsed:.3f} s")
    #FPS
    print(f"Image feature extraction FPS: {1/elapsed:.1f} img / s")
    
    feat_2d = feat_2d.squeeze(0)
    
    return feat_2d

def save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

        mask_entire = torch.zeros(n_points, dtype=torch.bool)
        mask_entire[rand_ind] = True
        
        feat_dim = feat_bank.shape[1] # feature dimension
        full_feat = torch.zeros((n_points, feat_dim), dtype=feat_bank.dtype, device=feat_bank.device)
        full_feat[point_ids] = feat_bank[point_ids]
        
        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire_double = mask_entire & mask

        # torch.save({"feat": feat_bank[mask_entire].cpu(),
        #             "mask_full": mask_entire
        # },  os.path.join(out_dir, scene_id +'.pt'))
        # print(os.path.join(out_dir, scene_id +'.pt') + ' is saved!')
        
        torch.save({
            "feat": full_feat[mask_entire].cpu(),  # 包含有用特征和0特征
            "mask_full": mask_entire_double                # 表示有用点
        }, os.path.join(out_dir, scene_id + '.pt'))

        print(os.path.join(out_dir, scene_id + '.pt') + ' is saved!')
        
def save_fused_feature_v2(feat_bank, point_ids, n_points, out_dir, scene_id, args):
    '''Save features.'''

    for n in range(args.num_rand_file_per_scene):
        if n_points < args.n_split_points:
            n_points_cur = n_points # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = args.n_split_points

        rand_ind = np.random.choice(range(n_points), n_points_cur, replace=False)

        mask_entire = torch.zeros(n_points, dtype=torch.bool)
        mask_entire[rand_ind] = True
        
        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire_double = mask_entire & mask

        
        torch.save({
            "feat": feat_bank[mask_entire].cpu(),  # 包含有用特征和0特征
            "mask_full": mask_entire_double                # 表示有用点
        }, os.path.join(out_dir, scene_id + '.pt'))

        print(os.path.join(out_dir, scene_id + '.pt') + ' is saved!')


class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
            visibility_threshold=0.25, cut_bound=0, intrinsics=None):
        
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None: # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int) # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[0]-self.cut_bound) \
                    * (pi[1] < self.image_dim[1]-self.cut_bound)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                                    self.vis_thres * depth_cur

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


def obtain_intr_extr_matterport(scene):
    '''Obtain the intrinsic and extrinsic parameters of Matterport3D.'''

    img_dir = os.path.join(scene, 'color')
    pose_dir = os.path.join(scene, 'pose')
    intr_dir = os.path.join(scene, 'intrinsic')
    img_names = sorted(glob.glob(img_dir+'/*.jpg'))

    intrinsics = []
    extrinsics = []
    for img_name in img_names:
        name = img_name.split('/')[-1][:-4]

        extrinsics.append(np.loadtxt(os.path.join(pose_dir, name+'.txt')))
        intrinsics.append(np.loadtxt(os.path.join(intr_dir, name+'.txt')))

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    img_names = np.asarray(img_names)

    return img_names, intrinsics, extrinsics

def get_matterport_camera_data(data_path, locs_in, args):
    '''Get all camera view related infomation of Matterport3D.'''

    # find bounding box of the current region
    bbox_l = locs_in.min(axis=0)
    bbox_h = locs_in.max(axis=0)

    building_name = data_path.split('/')[-1].split('_')[0]
    scene_id = data_path.split('/')[-1].split('.')[0]

    scene = os.path.join(args.data_root_2d, building_name)
    img_names, intrinsics, extrinsics = obtain_intr_extr_matterport(scene)

    cam_loc = extrinsics[:, :3, -1]
    ind_in_scene = (cam_loc[:, 0] > bbox_l[0]) & (cam_loc[:, 0] < bbox_h[0]) & \
                    (cam_loc[:, 1] > bbox_l[1]) & (cam_loc[:, 1] < bbox_h[1]) & \
                    (cam_loc[:, 2] > bbox_l[2]) & (cam_loc[:, 2] < bbox_h[2])

    img_names_in = img_names[ind_in_scene]
    intrinsics_in = intrinsics[ind_in_scene]
    extrinsics_in = extrinsics[ind_in_scene]
    num_img = len(img_names_in)

    # some regions have no views inside, we consider it differently for test and train/val
    if args.split == 'test' and num_img == 0:
        print('no views inside {}, take the nearest 100 images to fuse'.format(scene_id))
        #! take the nearest 100 views for feature fusion of regions without inside views
        centroid = (bbox_l+bbox_h)/2
        dist_centroid = np.linalg.norm(cam_loc-centroid, axis=-1)
        ind_in_scene = np.argsort(dist_centroid)[:100]
        img_names_in = img_names[ind_in_scene]
        intrinsics_in = intrinsics[ind_in_scene]
        extrinsics_in = extrinsics[ind_in_scene]
        num_img = 100

    img_names_in = img_names_in.tolist()

    return intrinsics_in, extrinsics_in, img_names_in, scene_id, num_img


