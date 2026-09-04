#!/usr/bin/env python3
"""Software-only prototype for pipeline steps 1-4.

1) T_camera_box -> T_robot_box using fixed T_robot_camera.
2) Select one simple box-relative interaction prior.
3) Build P0/P1/P2/P3 root waypoints.
4) Interpolate a 50 Hz root trajectory between them.

No robot commands are sent. All default extrinsics/priors are placeholders.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "g1/config/root_pipeline_demo.json"
DEFAULT_OUT = ROOT / "g1/results/root_pipeline_demo"
DEFAULT_LIVE_POSE = ROOT / "g1/results/live_foundationpose/latest_pose.txt"


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def R_to_rpy(R):
    p = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    cp = math.cos(p)
    if abs(cp) > 1e-7:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = 0.0
        y = math.atan2(-R[0, 1], R[1, 1])
    return np.array([r, p, y], dtype=float)


def pose_matrix(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def R_to_quat(R):
    # xyzw quaternion
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=float)


def quat_angle(q0, q1):
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def quat_slerp(q0, q1, u):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    s = math.sin(theta)
    return (math.sin((1-u)*theta) / s) * q0 + (math.sin(u*theta) / s) * q1


def make_root_target(box_xyz, current_xyz, prior):
    box_xy = np.asarray(box_xyz[:2], dtype=float)
    cur_xy = np.asarray(current_xyz[:2], dtype=float)
    away = cur_xy - box_xy
    n = np.linalg.norm(away)
    if n < 1e-6:
        away = np.array([-1.0, 0.0])
    else:
        away /= n
    left = np.array([-away[1], away[0]])
    root_xy = box_xy + prior["distance_m"] * away + prior["lateral_m"] * left
    yaw = math.atan2(box_xy[1] - root_xy[1], box_xy[0] - root_xy[0])
    return root_xy, yaw


def select_prior(priors, box_xyz, current_xyz, forced_name=None):
    if forced_name:
        for p in priors:
            if p["name"] == forced_name:
                return p
        raise ValueError(f"Unknown prior: {forced_name}")

    # V1 selection: choose the candidate requiring the least planar root travel.
    scored = []
    for p in priors:
        xy, _ = make_root_target(box_xyz, current_xyz, p)
        cost = float(np.linalg.norm(xy - np.asarray(current_xyz[:2])))
        scored.append((cost, p))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def interpolate_segment(a, b, hz, vmax, wmax):
    pa, pb = a["T"][:3, 3], b["T"][:3, 3]
    qa, qb = R_to_quat(a["T"][:3, :3]), R_to_quat(b["T"][:3, :3])
    dist = float(np.linalg.norm(pb - pa))
    ang = quat_angle(qa, qb)

    # smoothstep has peak derivative 1.5, so scale duration to respect limits.
    t_lin = 1.5 * dist / vmax if vmax > 0 else 0.0
    t_ang = 1.5 * ang / wmax if wmax > 0 else 0.0
    duration = max(t_lin, t_ang, 1.0 / hz)
    n = max(1, int(math.ceil(duration * hz)))
    dt = duration / n

    samples = []
    for i in range(1, n + 1):
        u = i / n
        s = u*u*(3.0 - 2.0*u)
        pos = pa + s * (pb - pa)
        q = quat_slerp(qa, qb, s)
        T = np.eye(4)
        T[:3, :3] = quat_to_R(q)
        T[:3, 3] = pos
        # Contact changes only once the destination interaction state is reached.
        contact = a["contact"] if i < n else b["contact"]
        samples.append((dt, T, contact))
    return duration, samples


def row_from_T(t, segment, T, contact):
    q = R_to_quat(T[:3, :3])
    rpy = R_to_rpy(T[:3, :3])
    return {
        "time_s": t,
        "segment": segment,
        "x": T[0, 3], "y": T[1, 3], "z": T[2, 3],
        "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
        "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
        "contact": int(contact),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--box-pose-camera", type=Path, default=None,
                    help="4x4 FoundationPose T_camera_box. If omitted, use live latest_pose if present, else demo pose.")
    ap.add_argument("--prior", type=str, default=None,
                    help="Force a named interaction prior instead of auto-selecting.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    T_robot_camera = np.asarray(cfg["T_robot_camera"], dtype=float)

    pose_path = args.box_pose_camera
    if pose_path is None and DEFAULT_LIVE_POSE.exists():
        pose_path = DEFAULT_LIVE_POSE

    if pose_path is not None:
        T_camera_box = np.loadtxt(pose_path).reshape(4, 4)
        pose_source = str(pose_path)
    else:
        T_camera_box = np.asarray(cfg["demo_T_camera_box"], dtype=float)
        pose_source = "config demo_T_camera_box"

    # Step 1: camera frame -> robot frame.
    T_robot_box = T_robot_camera @ T_camera_box

    cur = cfg["current_root"]
    T0 = pose_matrix(cur["xyz"], cur["rpy"])
    box_xyz = T_robot_box[:3, 3]

    # Step 2: choose a simple interaction prior.
    prior = select_prior(cfg["interaction_priors"], box_xyz, T0[:3, 3], args.prior)
    target_xy, target_yaw = make_root_target(box_xyz, T0[:3, 3], prior)

    # Step 3: P0=current, P1=stand near box, P2=grasp/lowered, P3=stand/lift.
    stand_h = float(cfg["stand_height_m"])
    grasp_h = float(prior["grasp_root_height_m"])
    P0 = {"name": "P0_current", "T": T0, "contact": 0}
    P1 = {"name": "P1_approach", "T": pose_matrix([target_xy[0], target_xy[1], stand_h], [0, 0, target_yaw]), "contact": 0}
    P2 = {"name": "P2_grasp", "T": pose_matrix([target_xy[0], target_xy[1], grasp_h], [0, 0, target_yaw]), "contact": 1}
    P3 = {"name": "P3_lift", "T": pose_matrix([target_xy[0], target_xy[1], stand_h], [0, 0, target_yaw]), "contact": 1}
    waypoints = [P0, P1, P2, P3]

    # Step 4: 50 Hz interpolation.
    tc = cfg["trajectory"]
    hz = float(tc["hz"])
    vmax = float(tc["max_linear_mps"])
    wmax = float(tc["max_angular_radps"])

    rows = [row_from_T(0.0, "P0", P0["T"], P0["contact"])]
    t = 0.0
    segment_info = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        duration, samples = interpolate_segment(a, b, hz, vmax, wmax)
        segment = f'{a["name"]}->{b["name"]}'
        segment_info.append((segment, duration, len(samples)))
        for dt, T, contact in samples:
            t += dt
            rows.append(row_from_T(t, segment, T, contact))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out / "T_robot_box.txt", T_robot_box, fmt="%.8f")

    wp_json = []
    for p in waypoints:
        rpy = R_to_rpy(p["T"][:3, :3])
        wp_json.append({
            "name": p["name"],
            "xyz": p["T"][:3, 3].tolist(),
            "rpy": rpy.tolist(),
            "contact": p["contact"],
        })
    (args.out / "waypoints.json").write_text(json.dumps({
        "pose_source": pose_source,
        "selected_prior": prior,
        "waypoints": wp_json,
    }, indent=2))

    csv_path = args.out / "root_trajectory_50hz.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("OFFLINE ROOT PIPELINE — no robot commands")
    print("box pose source:", pose_source)
    print("T_robot_box xyz:", np.round(T_robot_box[:3, 3], 4))
    print("selected prior:", prior["name"])
    for p in wp_json:
        print(f'{p["name"]}: xyz={np.round(p["xyz"], 4)} yaw={p["rpy"][2]:.3f} contact={p["contact"]}')
    for name, duration, n in segment_info:
        print(f"{name}: {duration:.2f} s, {n} samples")
    print(f"total: {t:.2f} s, {len(rows)} trajectory rows @ nominal {hz:.0f} Hz")
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
