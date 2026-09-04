#!/usr/bin/env python3
"""Run several software-only fake box/prior cases and generate 2D/3D plots."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "g1/config/root_pipeline_demo.json"
PIPELINE = ROOT / "g1/scripts/offline_root_pipeline.py"
VIS = ROOT / "g1/scripts/visualize_root_pipeline.py"
OUT_ROOT = ROOT / "g1/results/root_pipeline_fake_tests"


def pose_matrix(xyz, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = xyz
    return T


def read_csv(path):
    rows = list(csv.DictReader(path.open()))
    t = np.array([float(r["time_s"]) for r in rows])
    p = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    yaw = np.unwrap(np.array([float(r["yaw"]) for r in rows]))
    return t, p, yaw


def validate(case, case_dir, cfg):
    got_box = np.loadtxt(case_dir / "T_robot_box.txt").reshape(4, 4)
    expected_box = pose_matrix(case["box_xyz"], case["box_yaw"])
    if not np.allclose(got_box, expected_box, atol=2e-6):
        raise AssertionError(f"{case['name']}: robot-frame box transform mismatch")

    meta = json.loads((case_dir / "waypoints.json").read_text())
    if case["prior"] is not None and meta["selected_prior"]["name"] != case["prior"]:
        raise AssertionError(f"{case['name']}: wrong prior selected")

    t, p, yaw = read_csv(case_dir / "root_trajectory_50hz.csv")
    if len(t) < 4 or not np.all(np.diff(t) > 0):
        raise AssertionError(f"{case['name']}: bad trajectory timestamps")
    if not np.isfinite(p).all():
        raise AssertionError(f"{case['name']}: non-finite path")

    dt = np.diff(t)
    linear_speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    angular_speed = np.abs(np.diff(yaw)) / dt
    vmax = float(cfg["trajectory"]["max_linear_mps"])
    wmax = float(cfg["trajectory"]["max_angular_radps"])
    if linear_speed.max(initial=0.0) > vmax * 1.03:
        raise AssertionError(f"{case['name']}: linear speed limit exceeded")
    if angular_speed.max(initial=0.0) > wmax * 1.03:
        raise AssertionError(f"{case['name']}: angular speed limit exceeded")

    wp = meta["waypoints"]
    end = np.asarray(wp[-1]["xyz"], dtype=float)
    if np.linalg.norm(p[-1] - end) > 1e-6:
        raise AssertionError(f"{case['name']}: trajectory does not end at P3")

    for image in ["root_pipeline_2d.png", "root_pipeline_3d.png"]:
        if not (case_dir / image).exists():
            raise AssertionError(f"{case['name']}: missing {image}")

    return len(t), float(t[-1]), float(linear_speed.max(initial=0.0)), meta["selected_prior"]["name"]


def main():
    cfg = json.loads(CONFIG.read_text())
    T_robot_camera = np.asarray(cfg["T_robot_camera"], dtype=float)
    T_camera_robot = np.linalg.inv(T_robot_camera)

    cases = [
        {"name": "front_center", "box_xyz": [1.20, 0.00, 0.65], "box_yaw": 0.00, "prior": "front_045"},
        {"name": "box_left", "box_xyz": [1.00, 0.55, 0.70], "box_yaw": 0.25, "prior": "front_left_050"},
        {"name": "box_right", "box_xyz": [1.15, -0.50, 0.60], "box_yaw": -0.35, "prior": "front_right_050"},
        {"name": "auto_prior", "box_xyz": [0.85, 0.25, 0.68], "box_yaw": 0.60, "prior": None},
    ]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("FAKE ROOT PIPELINE TESTS — no robot commands")

    for case in cases:
        case_dir = OUT_ROOT / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)

        T_robot_box = pose_matrix(case["box_xyz"], case["box_yaw"])
        T_camera_box = T_camera_robot @ T_robot_box
        pose_file = case_dir / "fake_T_camera_box.txt"
        np.savetxt(pose_file, T_camera_box, fmt="%.9f")

        cmd = [sys.executable, str(PIPELINE), "--config", str(CONFIG),
               "--box-pose-camera", str(pose_file), "--out", str(case_dir)]
        if case["prior"] is not None:
            cmd += ["--prior", case["prior"]]
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, str(VIS), "--result-dir", str(case_dir)], check=True)

        n, duration, max_v, selected = validate(case, case_dir, cfg)
        print(f"PASS {case['name']}: prior={selected}, rows={n}, duration={duration:.2f}s, max_v={max_v:.3f}m/s")

    print(f"ALL {len(cases)} CASES PASSED")
    print("plots/results:", OUT_ROOT)


if __name__ == "__main__":
    main()
