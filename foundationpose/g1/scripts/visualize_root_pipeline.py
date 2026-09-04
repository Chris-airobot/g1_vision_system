#!/usr/bin/env python3
"""Offline 2D/3D visualizer for box + P0-P3 + interpolated root path."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = ROOT / "g1/results/root_pipeline_demo"
BOX_SIZE = np.array([0.40, 0.30, 0.30], dtype=float)


def load_data(result_dir):
    T_box = np.loadtxt(result_dir / "T_robot_box.txt").reshape(4, 4)
    meta = json.loads((result_dir / "waypoints.json").read_text())
    rows = []
    with (result_dir / "root_trajectory_50hz.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise RuntimeError("trajectory CSV is empty")
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    return T_box, meta, xyz


def box_corners(T, size):
    hx, hy, hz = size / 2.0
    local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz],  [hx, -hy, hz],  [hx, hy, hz],  [-hx, hy, hz],
    ])
    return (T[:3, :3] @ local.T).T + T[:3, 3]


def waypoint_arrays(meta):
    names = []
    xyz = []
    yaw = []
    for w in meta["waypoints"]:
        names.append(w["name"])
        xyz.append(w["xyz"])
        yaw.append(w["rpy"][2])
    return names, np.asarray(xyz, dtype=float), np.asarray(yaw, dtype=float)


def draw_box_2d(ax, T_box, size):
    c = box_corners(T_box, size)
    poly = c[[0, 1, 2, 3, 0], :2]
    ax.plot(poly[:, 0], poly[:, 1], linewidth=2, label="box")
    ax.scatter([T_box[0, 3]], [T_box[1, 3]], marker="s")


def draw_box_3d(ax, T_box, size):
    c = box_corners(T_box, size)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot([c[a,0], c[b,0]], [c[a,1], c[b,1]], [c[a,2], c[b,2]], linewidth=1.5)
    ax.scatter([T_box[0,3]], [T_box[1,3]], [T_box[2,3]], marker="s")


def save_2d(result_dir, T_box, meta, path_xyz, size):
    names, wp, yaw = waypoint_arrays(meta)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(path_xyz[:,0], path_xyz[:,1], linewidth=1.6, label="interpolated root path")
    draw_box_2d(ax, T_box, size)
    ax.scatter(wp[:,0], wp[:,1], s=45)
    for name, p, a in zip(names, wp, yaw):
        ax.text(p[0], p[1], "  " + name, fontsize=9)
        ax.arrow(p[0], p[1], 0.12*np.cos(a), 0.12*np.sin(a), head_width=0.035, length_includes_head=True)
    ax.set_xlabel("robot x [m]")
    ax.set_ylabel("robot y [m]")
    ax.set_title(f"Root pipeline top view | prior={meta['selected_prior']['name']}")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = result_dir / "root_pipeline_2d.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_3d(result_dir, T_box, meta, path_xyz, size):
    names, wp, _ = waypoint_arrays(meta)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(path_xyz[:,0], path_xyz[:,1], path_xyz[:,2], linewidth=1.6, label="interpolated root path")
    draw_box_3d(ax, T_box, size)
    ax.scatter(wp[:,0], wp[:,1], wp[:,2], s=45)
    for name, p in zip(names, wp):
        ax.text(p[0], p[1], p[2], "  " + name, fontsize=8)
    ax.set_xlabel("robot x [m]")
    ax.set_ylabel("robot y [m]")
    ax.set_zlabel("robot z [m]")
    ax.set_title(f"Root pipeline 3D | prior={meta['selected_prior']['name']}")
    ax.legend()
    fig.tight_layout()
    out = result_dir / "root_pipeline_3d.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    ap.add_argument("--box-size", type=float, nargs=3, default=BOX_SIZE.tolist(), metavar=("X","Y","Z"))
    args = ap.parse_args()

    T_box, meta, path_xyz = load_data(args.result_dir)
    size = np.asarray(args.box_size, dtype=float)
    out2 = save_2d(args.result_dir, T_box, meta, path_xyz, size)
    out3 = save_3d(args.result_dir, T_box, meta, path_xyz, size)
    print("Saved:", out2)
    print("Saved:", out3)


if __name__ == "__main__":
    main()
