"""Generate SUN RGB-D superpoints with VCCS.

The VCCS parameters follow:
G. Gao, M. Lauri, J. Zhang and S. Frintrop, "Saliency-guided adaptive
seeding for supervoxel segmentation," IROS 2017.
"""

import argparse
import multiprocessing
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate VCCS superpoint labels for SUN RGB-D point clouds."
    )
    parser.add_argument(
        "--points-dir",
        type=Path,
        default=Path("data/sunrgbd/points"),
        help="Directory containing float32 point-cloud .bin files "
        "(default: data/sunrgbd/points).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sunrgbd/superpoints"),
        help="Directory in which to save int64 superpoint labels "
        "(default: data/sunrgbd/superpoints).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes (default: all available CPU cores).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output files that already exist.",
    )
    return parser.parse_args()


def process_file(task):
    """Generate and save superpoint labels for one point-cloud file."""
    point_path, output_dir, overwrite = task
    point_path = Path(point_path)
    save_path = Path(output_dir) / point_path.name

    if save_path.exists() and not overwrite:
        print(f"File already exists, skipping: {save_path}")
        return

    # Import in each worker so that argument parsing and --help do not require
    # pclpy to be installed.
    from pclpy import pcl

    start_time = time.time()
    points = np.fromfile(point_path, dtype=np.float32)
    if points.size % 6 != 0:
        raise ValueError(
            f"{point_path} contains {points.size} float32 values; "
            "the value count must be divisible by 6 (x, y, z, r, g, b)."
        )
    points = points.reshape(-1, 6)
    xyz = points[:, :3]
    rgb = points[:, 3:]

    point_cloud = pcl.PointCloud.PointXYZRGBA(xyz, rgb)
    normals = point_cloud.compute_normals(radius=0.25)

    vox = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(
        voxel_resolution=0.02, seed_resolution=0.25
    )
    vox.setInputCloud(point_cloud)
    vox.setNormalCloud(normals)
    vox.setSpatialImportance(0.15)
    vox.setNormalImportance(0.15)
    vox.setColorImportance(0.7)

    output = pcl.vectors.map_uint32t_PointXYZRGBA()
    vox.extract(output)
    labeled_cloud = vox.getLabeledCloud()

    labels = np.empty((labeled_cloud.size(),), dtype=np.int64)
    for index in range(labeled_cloud.size()):
        labels[index] = labeled_cloud.at(index).label

    labels.tofile(save_path)
    elapsed = time.time() - start_time
    num_superpoints = len(list(output.items()))
    print(
        f"Saved {save_path} (points: {labels.size}, "
        f"superpoints: {num_superpoints}, time: {elapsed:.2f}s)"
    )


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if not args.points_dir.is_dir():
        raise FileNotFoundError(f"Points directory not found: {args.points_dir}")

    point_files = sorted(
        path for path in args.points_dir.glob("*.bin") if path.is_file()
    )
    if not point_files:
        raise FileNotFoundError(f"No .bin files found in {args.points_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(path), str(args.output_dir), args.overwrite) for path in point_files
    ]

    print(f"Found {len(point_files)} point-cloud files.")
    print(f"Using {args.workers} worker process(es).")
    with multiprocessing.Pool(processes=args.workers) as pool:
        for _ in pool.imap_unordered(process_file, tasks):
            pass


if __name__ == "__main__":
    main()
