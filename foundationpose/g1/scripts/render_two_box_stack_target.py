#!/usr/bin/env python3
"""Render and export a real two-box stacking target from two FoundationPose streams.

Inputs are two independently tracked instances in the same RGB-D bundle:
  - carried box A FoundationPose CSV
  - real support/top-box FoundationPose CSV

For each synchronized frame:
  1. read T_camera_A and T_camera_support,
  2. use RGB-D-derived global up (estimated once from a reference depth frame),
  3. compute desired final pose B:
         p_B = p_support + box_height * global_up_camera
         R_B = R_support
  4. export B for downstream control,
  5. overlay real A, real support, and desired B on RGB.

This script does not run FoundationPose itself. It consumes saved per-instance
FoundationPose trajectories. It does not run Isaac Sim or send robot commands.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import render_stack_target_on_rgb as base
import render_stack_target_on_rgb_v3 as v3


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "g1/data/offline_bundle_20260813_140853"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument(
        "--carried-csv", type=Path, default=None,
        help=(
            "FoundationPose CSV for carried box A. Default: "
            "<bundle>/foundationpose_instances/carried/foundationpose_poses.csv"
        ),
    )
    p.add_argument(
        "--support-csv", type=Path, default=None,
        help=(
            "FoundationPose CSV for real support/top box. Default: "
            "<bundle>/foundationpose_instances/support/foundationpose_poses.csv"
        ),
    )
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--box-dims", type=float, nargs=3, default=(0.40, 0.30, 0.30))
    p.add_argument("--box-height", type=float, default=0.30)
    p.add_argument(
        "--up-reference-frame", type=int, default=-1,
        help="Frame id used to fit global up from depth. Default: first synchronized frame.",
    )
    p.add_argument("--plane-threshold", type=float, default=0.015)
    p.add_argument("--plane-iters", type=int, default=500)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_pose_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    rows = {}
    for row in csv.DictReader(path.open()):
        frame = str(row["frame"])
        rows[frame] = base.load_pose(row)
    if not rows:
        raise RuntimeError(f"No poses found in {path}")
    return rows


def rotation_error_deg(R_a, R_b):
    R = R_a.T @ R_b
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def alignment_metrics(T_support, T_B, up_camera):
    up = np.asarray(up_camera, dtype=np.float64)
    up /= np.linalg.norm(up)
    delta = T_B[:3, 3] - T_support[:3, 3]
    vertical = float(np.dot(delta, up))
    lateral_vec = delta - vertical * up
    return {
        "center_distance_m": float(np.linalg.norm(delta)),
        "vertical_offset_m": vertical,
        "lateral_error_m": float(np.linalg.norm(lateral_vec)),
        "rotation_difference_deg": rotation_error_deg(
            T_support[:3, :3], T_B[:3, :3]
        ),
    }


def write_pose_fields(row, prefix, T):
    for r in range(4):
        for c in range(4):
            row[f"{prefix}_t{r}{c}"] = f"{float(T[r, c]):.9f}"


def draw_frame(img, T_A, T_support, T_B, K, pts, frame_id, metrics):
    vis = img.copy()

    base.draw_wireframe(
        vis, base.project(pts, T_support, K), base.COLORS["support"], 3, 0.80
    )
    base.draw_label(
        vis, base.project_center(T_support, K), "REAL SUPPORT / TOP BOX", base.COLORS["support"]
    )

    base.draw_wireframe(
        vis, base.project(pts, T_B, K), base.COLORS["target"], 3, 0.82
    )
    base.draw_label(
        vis, base.project_center(T_B, K), "DESIRED FINAL B", base.COLORS["target"]
    )

    base.draw_wireframe(
        vis, base.project(pts, T_A, K), base.COLORS["tracked"], 3, 1.0
    )
    base.draw_label(
        vis, base.project_center(T_A, K), "REAL CARRIED A", base.COLORS["tracked"]
    )

    cS = base.project_center(T_support, K)
    cB = base.project_center(T_B, K)
    cA = base.project_center(T_A, K)

    if cS is not None and cB is not None:
        s = tuple(np.round(cS).astype(int))
        b = tuple(np.round(cB).astype(int))
        cv2.line(vis, s, b, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.arrowedLine(vis, s, b, base.COLORS["target"], 3, cv2.LINE_AA, tipLength=0.10)
        cv2.circle(vis, s, 7, base.COLORS["support"], -1, cv2.LINE_AA)
        cv2.circle(vis, b, 7, base.COLORS["target"], -1, cv2.LINE_AA)

    if cA is not None and cB is not None:
        cv2.arrowedLine(
            vis,
            tuple(np.round(cA).astype(int)),
            tuple(np.round(cB).astype(int)),
            base.COLORS["path"], 2, cv2.LINE_AA, tipLength=0.04,
        )

    base.put_text(vis, "TWO REAL BOX INSTANCES -> desired stacking pose B", 26)
    base.put_text(vis, f"frame {frame_id}", 50, scale=0.46)
    base.put_text(
        vis, "green=carried A | cyan=real support | magenta=desired B", 74, scale=0.42
    )
    base.put_text(
        vis,
        f"3D lateral={metrics['lateral_error_m']:.4f} m | "
        f"center={metrics['center_distance_m']:.4f} m | "
        f"dR={metrics['rotation_difference_deg']:.3f} deg",
        98,
        scale=0.40,
    )
    return vis


def main():
    args = parse_args()
    bundle = args.bundle
    carried_csv = args.carried_csv or (
        bundle / "foundationpose_instances/carried/foundationpose_poses.csv"
    )
    support_csv = args.support_csv or (
        bundle / "foundationpose_instances/support/foundationpose_poses.csv"
    )

    K = np.loadtxt(bundle / "cam_K.txt").reshape(3, 3)
    timestamps = list(csv.DictReader((bundle / "timestamps.csv").open()))
    ts_by_frame = {str(r["frame"]): r for r in timestamps}

    carried = load_pose_csv(carried_csv)
    support = load_pose_csv(support_csv)

    shared_frames = [
        str(r["frame"])
        for r in timestamps
        if str(r["frame"]) in carried and str(r["frame"]) in support
    ]
    if not shared_frames:
        raise RuntimeError(
            "No synchronized frame ids are shared by carried/support FoundationPose CSVs."
        )

    if args.up_reference_frame >= 0:
        ref_frame = str(args.up_reference_frame)
        if ref_frame not in shared_frames:
            raise RuntimeError(
                f"--up-reference-frame {ref_frame} is not present in both pose streams"
            )
    else:
        ref_frame = shared_frames[0]

    ref_ts = ts_by_frame[ref_frame]
    ref_depth_path = bundle / ref_ts["depth_file"]
    depth = cv2.imread(str(ref_depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise RuntimeError(f"Could not read synchronized depth frame {ref_depth_path}")

    cloud = v3.depth_to_points(depth, K, stride=4)
    up_camera, plane_d, plane_inliers, support_center_height = v3.fit_support_plane(
        cloud,
        support[ref_frame][:3, 3],
        expected_center_height=float(args.box_height) * 0.5,
        threshold=float(args.plane_threshold),
        iters=int(args.plane_iters),
    )

    first_rgb = cv2.imread(str(bundle / ts_by_frame[shared_frames[0]]["rgb_file"]))
    if first_rgb is None:
        raise RuntimeError("Could not read first synchronized RGB frame")
    h, w = first_rgb.shape[:2]
    pts = base.box_corners(np.asarray(args.box_dims, dtype=np.float64))

    out_dir = args.output_dir or (bundle / "two_box_stack_target")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "two_box_stack_target.mp4"
    out_csv = out_dir / "two_box_target_poses.csv"
    latest_txt = out_dir / "T_camera_target_B_latest.txt"
    summary_path = out_dir / "two_box_stack_target_summary.json"

    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_video}")

    fieldnames = ["frame", "rgb_file"]
    for prefix in ["A", "support", "B"]:
        fieldnames.extend([f"{prefix}_t{r}{c}" for r in range(4) for c in range(4)])
    fieldnames.extend([
        "center_distance_m", "vertical_offset_m", "lateral_error_m",
        "rotation_difference_deg",
    ])

    metric_rows = []
    written = 0
    rendered_shared_indices = list(range(0, len(shared_frames), max(1, args.every)))
    snapshot_positions = {
        rendered_shared_indices[0]: "start",
        rendered_shared_indices[len(rendered_shared_indices) // 2]: "mid",
        rendered_shared_indices[-1]: "end",
    }

    with out_csv.open("w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for shared_i, frame_id in enumerate(shared_frames):
            T_A = carried[frame_id]
            T_support = support[frame_id]
            T_B = v3.make_target(T_support, up_camera, args.box_height)
            metrics = alignment_metrics(T_support, T_B, up_camera)
            metric_rows.append(metrics)

            row = {
                "frame": frame_id,
                "rgb_file": ts_by_frame[frame_id]["rgb_file"],
                **{k: f"{v:.9f}" for k, v in metrics.items()},
            }
            write_pose_fields(row, "A", T_A)
            write_pose_fields(row, "support", T_support)
            write_pose_fields(row, "B", T_B)
            csv_writer.writerow(row)
            np.savetxt(latest_txt, T_B)

            if shared_i not in snapshot_positions and shared_i % max(1, args.every) != 0:
                continue

            img = cv2.imread(str(bundle / ts_by_frame[frame_id]["rgb_file"]))
            if img is None:
                continue
            vis = draw_frame(img, T_A, T_support, T_B, K, pts, frame_id, metrics)
            writer.write(vis)
            written += 1
            if shared_i in snapshot_positions:
                cv2.imwrite(
                    str(out_dir / f"two_box_preview_{snapshot_positions[shared_i]}.png"),
                    vis,
                )

    writer.release()

    lateral = np.asarray([m["lateral_error_m"] for m in metric_rows], dtype=np.float64)
    center = np.asarray([m["center_distance_m"] for m in metric_rows], dtype=np.float64)
    rot = np.asarray([m["rotation_difference_deg"] for m in metric_rows], dtype=np.float64)

    summary = {
        "mode": "two_real_foundationpose_instances",
        "carried_pose_csv": str(carried_csv),
        "support_pose_csv": str(support_csv),
        "shared_frame_count": len(shared_frames),
        "global_up_reference_frame": ref_frame,
        "global_up_camera": [float(x) for x in up_camera],
        "plane_d": float(plane_d),
        "plane_inliers": int(plane_inliers),
        "reference_support_center_height_above_plane_m": float(support_center_height),
        "box_height_m": float(args.box_height),
        "alignment": {
            "max_lateral_error_m": float(lateral.max()),
            "mean_lateral_error_m": float(lateral.mean()),
            "mean_center_distance_m": float(center.mean()),
            "max_abs_center_distance_error_m": float(np.max(np.abs(center - args.box_height))),
            "max_rotation_difference_deg": float(rot.max()),
        },
        "outputs": {
            "video": str(out_video),
            "target_pose_csv": str(out_csv),
            "latest_target_pose": str(latest_txt),
        },
        "note": (
            "Support is now a real independently tracked FoundationPose instance, not a virtual "
            "historical pose. Desired B is recomputed from the current support pose on every "
            "synchronized frame."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("TWO-BOX STACK TARGET PIPELINE")
    print("carried CSV:", carried_csv)
    print("support CSV:", support_csv)
    print("shared frames:", len(shared_frames))
    print("global-up reference frame:", ref_frame)
    print("global up camera:", np.round(up_camera, 6))
    print("max 3D lateral error [m]:", f"{lateral.max():.9f}")
    print("mean center distance [m]:", f"{center.mean():.9f}")
    print("max rotation difference [deg]:", f"{rot.max():.6f}")
    print("target poses:", out_csv)
    print("latest B:", latest_txt)
    print("video:", out_video)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
