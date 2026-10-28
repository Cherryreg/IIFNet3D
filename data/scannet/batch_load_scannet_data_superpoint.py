# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Export ScanNet scenes with superpoints and ground-truth annotations."""
import argparse
import datetime
import os
from os import path as osp

import numpy as np

DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))


def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    scannet_dir,
                    test_mode=False):
    # Import lazily so that ``--help`` works before the optional superpoint
    # dependencies (segmentator and Open3D) are installed.
    from load_scannet_data_superpoint import export

    mesh_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '.aggregation.json')
    seg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '_vh_clean_2.0.010000.segs.json')
    # includes axisAlignment info for the train set scans.
    meta_file = osp.join(scannet_dir, scan_name, f'{scan_name}.txt')
    mesh_vertices, mesh_superpoint, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic, axis_align_matrix = export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None,
            test_mode)

    if not test_mode:
        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        mesh_superpoint = mesh_superpoint[mask]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')

        bbox_mask = np.in1d(unaligned_bboxes[:, -1], OBJ_CLASS_IDS)
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        bbox_mask = np.in1d(aligned_bboxes[:, -1], OBJ_CLASS_IDS)
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f'Num of care instances: {unaligned_bboxes.shape[0]}')

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            mesh_superpoint = mesh_superpoint[choices]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]

    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    np.save(f'{output_filename_prefix}_superpoint.npy', mesh_superpoint)
    if not test_mode:
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
        np.save(f'{output_filename_prefix}_unaligned_bbox.npy',
                unaligned_bboxes)
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)
        np.save(f'{output_filename_prefix}_axis_align_matrix.npy',
                axis_align_matrix)


def batch_export(max_num_point,
                 output_folder,
                 scan_names_file,
                 label_map_file,
                 scannet_dir,
                 test_mode=False):
    if test_mode and not os.path.exists(scannet_dir):
        # test data preparation is optional
        return
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.makedirs(output_folder, exist_ok=True)

    with open(scan_names_file) as f:
        scan_names = [line.rstrip() for line in f]
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        # try:
        export_one_scan(scan_name, output_filename_prefix, max_num_point,
                        label_map_file, scannet_dir, test_mode)
        # except Exception:
        #     print(f'Failed export scan: {scan_name}')
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser(
        description='Generate ScanNet instance data and mesh superpoints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--scannet-root',
        default=SCRIPT_DIR,
        help=('ScanNet data root containing scans/, scans_test/, and '
              'meta_data/.'))
    parser.add_argument(
        '--max-num-point',
        '--max_num_point',
        dest='max_num_point',
        default=None,
        type=int,
        help=('Maximum number of points kept per scene. Leave unset to keep '
              'all points and preserve superpoint alignment.'))
    parser.add_argument(
        '--output-folder',
        '--output_folder',
        dest='output_folder',
        default=None,
        help=('Output directory. Defaults to '
              '<scannet-root>/scannet_instance_data.'))
    parser.add_argument(
        '--train-scannet-dir',
        '--train_scannet_dir',
        dest='train_scannet_dir',
        default=None,
        help='Training scans directory. Defaults to <scannet-root>/scans.')
    parser.add_argument(
        '--test-scannet-dir',
        '--test_scannet_dir',
        dest='test_scannet_dir',
        default=None,
        help=('Test scans directory. Defaults to '
              '<scannet-root>/scans_test.'))
    parser.add_argument(
        '--label-map-file',
        '--label_map_file',
        dest='label_map_file',
        default=None,
        help=('Label-map file. Defaults to <scannet-root>/meta_data/'
              'scannetv2-labels.combined.tsv.'))
    parser.add_argument(
        '--train-scan-names-file',
        '--train_scan_names_file',
        dest='train_scan_names_file',
        default=None,
        help=('Training split file. Defaults to <scannet-root>/meta_data/'
              'scannet_train.txt.'))
    parser.add_argument(
        '--test-scan-names-file',
        '--test_scan_names_file',
        dest='test_scan_names_file',
        default=None,
        help=('Test split file. Defaults to <scannet-root>/meta_data/'
              'scannetv2_test.txt.'))
    args = parser.parse_args()

    scannet_root = osp.abspath(osp.expanduser(args.scannet_root))
    output_folder = args.output_folder or osp.join(
        scannet_root, 'scannet_instance_data')
    train_scannet_dir = args.train_scannet_dir or osp.join(
        scannet_root, 'scans')
    test_scannet_dir = args.test_scannet_dir or osp.join(
        scannet_root, 'scans_test')
    label_map_file = args.label_map_file or osp.join(
        scannet_root, 'meta_data', 'scannetv2-labels.combined.tsv')
    train_scan_names_file = args.train_scan_names_file or osp.join(
        scannet_root, 'meta_data', 'scannet_train.txt')
    test_scan_names_file = args.test_scan_names_file or osp.join(
        scannet_root, 'meta_data', 'scannetv2_test.txt')

    batch_export(
        args.max_num_point,
        output_folder,
        train_scan_names_file,
        label_map_file,
        train_scannet_dir,
        test_mode=False)
    batch_export(
        args.max_num_point,
        output_folder,
        test_scan_names_file,
        label_map_file,
        test_scannet_dir,
        test_mode=True)


if __name__ == '__main__':
    main()
