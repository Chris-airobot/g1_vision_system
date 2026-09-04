#!/usr/bin/env python3
"""Real-RGB stack overlay using a fixed global-up direction.

This visualization intentionally does NOT use any FoundationPose object-local
axis to decide where "on top" is. The desired final box center is translated
one box height along a fixed global-up direction expressed in camera coordinates.

For the current stationary G1 camera recording, the default approximation is:
    global/world +Z ~= camera -Y
because OpenCV camera coordinates are x-right, y-down, z-forward.

Once a trusted T_world_camera/T_robot_camera is available, pass the exact
camera-frame global-up vector with --global-up-camera.

Visualization only: saved RGB + saved FoundationPose poses. No FoundationPose
inference, Isaac Sim, or robot commands are run.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import render_stack_target_on_rgb as base

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "g1/data/offline_bundle_20260813_140853"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--box-dims", type=float, nargs=3, default=(0.40, 0.30, 0.30))
    p.add_argument("--box-height", type=float, default=0.30)
    p.add_argument("--support-record-index", type=int, default=-1)
    p.add_argument(
        "--global-up-camera", type=float, nargs=3, default=(0.0, -1.0, 0.0),
        metavar=("UX", "UY", "UZ"),
        help=(
            "World/global +Z expressed in camera coordinates. "
            "Default 0 -1 0 assumes a level OpenCV camera: image-up is world-up."
        ),
    )
    return p.parse_args()


def normalized(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("global-up vector must be non-zero")
    return v / n


def make_target(T_support, global_up_camera, box_height):
    """Preserve support orientation; translate center along global/world up."""
    T_B = T_support.copy()
    T_B[:3, 3] = T_support[:3, 3] + float(box_height) * global_up_camera
    return T_B


def choose_support(records, K, dims, box_height, global_up_camera, w, h):
    pts = base.box_corners(dims)
    best = None
    stride = max(1, len(records) // 300)

    first_center = base.project_center(records[0]["pose"], K)

    for idx in range(0, len(records), stride):
        T_support = records[idx]["pose"]
        T_B = make_target(T_support, global_up_camera, box_height)

        uv_support = base.project(pts, T_support, K)
        uv_B = base.project(pts, T_B, K)
        frac_support = base.visible_corner_fraction(uv_support, w, h)
        frac_B = base.visible_corner_fraction(uv_B, w, h)
        c_support = base.project_center(T_support, K)
        c_B = base.project_center(T_B, K)

        if min(frac_support, frac_B) < 0.50:
            continue
        if not (base.in_image(c_support, w, h) and base.in_image(c_B, w, h)):
            continue

        # Prefer a clearly separated support location while keeping both boxes visible.
        separation = 0.0 if first_center is None else float(np.linalg.norm(c_support - first_center))
        visibility = frac_support + frac_B
        score = separation + 120.0 * visibility
        if best is None or score > best[0]:
            best = (score, idx)

    return best[1] if best is not None else len(records) // 2


def render_frame(img, T_A, T_support, T_B, K, pts, idx, total):
    vis = img.copy()

    # Virtual support and final desired box.
    for label, T, key, alpha, thick in [
        ("VIRTUAL SUPPORT", T_support, "support", 0.64, 2),
        ("DESIRED FINAL B - ON TOP", T_B, "target", 0.82, 3),
    ]:
        base.draw_wireframe(vis, base.project(pts, T, K), base.COLORS[key], thick, alpha)
        base.draw_label(vis, base.project_center(T, K), label, base.COLORS[key])

    # Real saved FoundationPose track.
    base.draw_wireframe(vis, base.project(pts, T_A, K), base.COLORS["tracked"], 3, 1.0)
    base.draw_label(vis, base.project_center(T_A, K), "TRACKED A", base.COLORS["tracked"])

    cA = base.project_center(T_A, K)
    cB = base.project_center(T_B, K)
    cS = base.project_center(T_support, K)

    if cA is not None and cB is not None:
        cv2.arrowedLine(
            vis,
            tuple(np.round(cA).astype(int)),
            tuple(np.round(cB).astype(int)),
            base.COLORS["path"], 2, cv2.LINE_AA, tipLength=0.04,
        )

    # Explicit vertical stack arrow support -> B.
    if cS is not None and cB is not None:
        cv2.arrowedLine(
            vis,
            tuple(np.round(cS).astype(int)),
            tuple(np.round(cB).astype(int)),
            base.COLORS["target"], 3, cv2.LINE_AA, tipLength=0.12,
        )

    base.put_text(vis, "GLOBAL-UP STACKING: desired B is above support", 26)
    base.put_text(vis, f"frame {idx}/{total}", 50, scale=0.44)
    base.put_text(vis, "green=A tracked | cyan=support | magenta=desired final B", 74, scale=0.41)
    return vis


def main():
    args = parse_args()
    bundle = args.bundle
    if args.every < 1:
        raise ValueError("--every must be >= 1")
    if args.box_height <= 0:
        raise ValueError("--box-height must be positive")

    global_up_camera = normalized(args.global_up_camera)

    K = np.loadtxt(bundle / "cam_K.txt").reshape(3, 3)
    timestamps = list(csv.DictReader((bundle / "timestamps.csv").open()))
    fp_csv = bundle / "foundationpose_offline" / "foundationpose_poses.csv"
    if not fp_csv.exists():
        raise FileNotFoundError(f"Missing {fp_csv}")
    fp_rows = {str(r["frame"]): r for r in csv.DictReader(fp_csv.open())}

    records = []
    for ts in timestamps:
        row = fp_rows.get(str(ts["frame"]))
        rgb_path = bundle / ts["rgb_file"]
        if row is None or not rgb_path.exists():
            continue
        records.append({
            "frame_id": str(ts["frame"]),
            "rgb_path": rgb_path,
            "pose": base.load_pose(row),
        })

    if len(records) < 2:
        raise RuntimeError("not enough saved RGB + pose records")

    first = cv2.imread(str(records[0]["rgb_path"]))
    if first is None:
        raise RuntimeError("could not read first RGB frame")
    h, w = first.shape[:2]

    dims = np.asarray(args.box_dims, dtype=np.float64)
    pts = base.box_corners(dims)

    if args.support_record_index >= 0:
        support_idx = args.support_record_index
    else:
        support_idx = choose_support(
            records, K, dims, args.box_height, global_up_camera, w, h
        )

    if not (0 <= support_idx < len(records)):
        raise IndexError(f"support index out of range: {support_idx}")

    T_support = records[support_idx]["pose"].copy()
    T_B = make_target(T_support, global_up_camera, args.box_height)

    out_dir = bundle / "stack_target_rgb_overlay_v2"
    out_dir.mkdir(exist_ok=True)
    out_video = out_dir / "stack_target_rgb_overlay_v2.mp4"

    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError("could not open video writer")

    indices = list(range(0, len(records), args.every))
    snap = {
        indices[0]: "start",
        indices[len(indices) // 2]: "mid",
        indices[-1]: "end",
    }

    written = 0
    for ridx in indices:
        img = cv2.imread(str(records[ridx]["rgb_path"]))
        if img is None:
            continue
        vis = render_frame(
            img, records[ridx]["pose"], T_support, T_B,
            K, pts, ridx, len(records) - 1,
        )
        writer.write(vis)
        written += 1
        if ridx in snap:
            cv2.imwrite(str(out_dir / f"stack_target_preview_{snap[ridx]}.png"), vis)

    writer.release()

    summary = {
        "mode": "global_up_no_preplace",
        "support_record_index": int(support_idx),
        "support_frame_id": records[support_idx]["frame_id"],
        "global_up_camera_vector": [float(x) for x in global_up_camera],
        "support_center_camera_m": [float(x) for x in T_support[:3, 3]],
        "desired_final_B_center_camera_m": [float(x) for x in T_B[:3, 3]],
        "stack_center_distance_m": float(args.box_height),
        "written_frames": int(written),
        "note": (
            "Desired final B is translated along a fixed global/world-up direction, "
            "not along any FoundationPose object-local axis. Default camera-space up "
            "is [0,-1,0] for the current level-camera visualization."
        ),
    }
    (out_dir / "stack_target_rgb_overlay_v2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("GLOBAL-UP REAL-RGB STACK OVERLAY")
    print("support record:", support_idx, "frame:", records[support_idx]["frame_id"])
    print("global-up in camera frame:", np.round(global_up_camera, 6))
    print("support center:", np.round(T_support[:3, 3], 4))
    print("desired final B:", np.round(T_B[:3, 3], 4))
    print("center offset norm:", round(float(np.linalg.norm(T_B[:3, 3] - T_support[:3, 3])), 4))
    print("written:", written)
    print("saved:", out_video)


if __name__ == "__main__":
    main()
