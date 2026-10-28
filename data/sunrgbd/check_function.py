"""Inspect and optionally visualize one SUN RGB-D superpoint result."""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check that SUN RGB-D points and superpoint labels align."
    )
    parser.add_argument(
        "points_file",
        type=Path,
        help="Float32 .bin file containing x, y, z, r, g, b values.",
    )
    parser.add_argument(
        "superpoints_file",
        type=Path,
        help="Int64 .bin file containing one superpoint label per point.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for a colored xyzrgb text file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display the colored point cloud with Open3D.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to assign superpoint colors (default: 0).",
    )
    return parser.parse_args()


def load_result(points_file, superpoints_file):
    point_values = np.fromfile(points_file, dtype=np.float32)
    if point_values.size % 6 != 0:
        raise ValueError(
            f"{points_file} contains {point_values.size} float32 values; "
            "the value count must be divisible by 6."
        )
    points = point_values.reshape(-1, 6)
    labels = np.fromfile(superpoints_file, dtype=np.int64)

    if points.shape[0] != labels.size:
        raise ValueError(
            "Point/label count mismatch: "
            f"{points.shape[0]} points versus {labels.size} labels."
        )
    if labels.size == 0:
        raise ValueError("The input files contain no points.")
    if np.any(labels < 0):
        raise ValueError("Superpoint labels must be non-negative.")
    return points, labels


def colorize(points, labels, seed):
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    rng = np.random.default_rng(seed)
    color_map = rng.uniform(0.1, 1.0, size=(unique_labels.size, 3))
    colored_points = np.concatenate((points[:, :3], color_map[inverse]), axis=1)
    return colored_points, unique_labels.size


def main():
    args = parse_args()
    points, labels = load_result(args.points_file, args.superpoints_file)
    colored_points, num_superpoints = colorize(points, labels, args.seed)

    print(f"Point shape: {points.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Number of superpoints: {num_superpoints}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.output, colored_points)
        print(f"Saved colored point cloud to: {args.output}")

    if args.visualize:
        try:
            import open3d as o3d
        except ImportError as error:
            raise ImportError(
                "Open3D is required for --visualize. Install it with "
                "`pip install open3d`."
            ) from error

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(colored_points[:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(colored_points[:, 3:])
        o3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    main()
