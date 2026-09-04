#!/usr/bin/env python3
"""Render desired stacking pose B on real RGB using RGB-D-derived global up.

V3 estimates the horizontal support plane from synchronized depth, treats the
plane normal as camera-frame global +Z, and builds the desired final pose as:

    p_B = p_support + box_height * up_camera
    R_B = R_support

The visualization also reports explicit 3D alignment diagnostics so perspective
projection cannot be confused with a true lateral stack offset.

Visualization only. Uses saved RGB/depth/FoundationPose outputs. It does not run
FoundationPose, Isaac Sim, or robot commands.
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
    p.add_argument("--plane-threshold", type=float, default=0.015,
                   help="RANSAC plane inlier threshold [m].")
    p.add_argument("--plane-iters", type=int, default=500)
    return p.parse_args()


def depth_to_points(depth_mm, K, stride=4, min_m=0.25, max_m=4.0):
    depth = depth_mm.astype(np.float64) * 0.001
    h, w = depth.shape[:2]
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    xx, yy = np.meshgrid(xs, ys)
    z = depth[yy, xx]
    valid = np.isfinite(z) & (z >= min_m) & (z <= max_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    u = xx[valid].astype(np.float64)
    v = yy[valid].astype(np.float64)
    z = z[valid]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.column_stack([x, y, z])


def plane_from_three(a, b, c):
    n = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(n))
    if norm < 1e-8:
        return None
    n /= norm
    d = -float(np.dot(n, a))
    return n, d


def refine_plane(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n /= np.linalg.norm(n)
    d = -float(np.dot(n, centroid))
    return n, d


def fit_support_plane(points, support_center, expected_center_height, threshold, iters):
    if len(points) < 100:
        raise RuntimeError(f"not enough valid depth points for plane fit: {len(points)}")

    rng = np.random.default_rng(7)
    npts = len(points)
    best = None

    if npts > 35000:
        score_idx = rng.choice(npts, size=35000, replace=False)
        score_pts = points[score_idx]
    else:
        score_pts = points

    for _ in range(int(iters)):
        ids = rng.choice(npts, size=3, replace=False)
        model = plane_from_three(points[ids[0]], points[ids[1]], points[ids[2]])
        if model is None:
            continue
        n, d = model
        dist = np.abs(score_pts @ n + d)
        inliers = dist < threshold
        count = int(inliers.sum())
        if count < 150:
            continue

        center_dist = abs(float(np.dot(n, support_center) + d))
        height_error = abs(center_dist - expected_center_height)
        plausibility = np.exp(-0.5 * (height_error / 0.10) ** 2)
        score = count * (0.20 + 0.80 * plausibility)
        if best is None or score > best[0]:
            best = (score, n, d)

    if best is None:
        raise RuntimeError("RANSAC could not find a plausible support plane")

    _, n0, d0 = best
    full_dist = np.abs(points @ n0 + d0)
    inlier_pts = points[full_dist < threshold]
    if len(inlier_pts) >= 50:
        n, d = refine_plane(inlier_pts)
    else:
        n, d = n0, d0

    signed = float(np.dot(n, support_center) + d)
    if signed < 0:
        n = -n
        d = -d
        signed = -signed

    inlier_count = int((np.abs(points @ n + d) < threshold).sum())
    return n, d, inlier_count, signed


def choose_support_record(records, K, dims, box_height, w, h):
    pts = base.box_corners(dims)
    best = None
    stride = max(1, len(records) // 300)
    for idx in range(0, len(records), stride):
        T = records[idx]["pose"]
        uv = base.project(pts, T, K)
        c = base.project_center(T, K)
        frac = base.visible_corner_fraction(uv, w, h)
        if frac < 0.75 or not base.in_image(c, w, h, margin=30):
            continue
        score = 2.0 * frac + (c[1] / max(h, 1)) - 0.15 * abs(T[2, 3] - 1.5)
        if best is None or score > best[0]:
            best = (score, idx)
    return best[1] if best is not None else len(records) // 2


def make_target(T_support, up_camera, box_height):
    T_B = T_support.copy()
    up = np.asarray(up_camera, dtype=np.float64)
    up /= np.linalg.norm(up)
    T_B[:3, 3] = T_support[:3, 3] + float(box_height) * up
    return T_B


def rotation_error_deg(R_a, R_b):
    R = R_a.T @ R_b
    cos_angle = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def alignment_metrics(T_support, T_B, up_camera):
    up = np.asarray(up_camera, dtype=np.float64)
    up /= np.linalg.norm(up)
    delta = T_B[:3, 3] - T_support[:3, 3]
    vertical = float(np.dot(delta, up))
    lateral_vec = delta - vertical * up
    lateral = float(np.linalg.norm(lateral_vec))
    center_distance = float(np.linalg.norm(delta))
    rot_deg = rotation_error_deg(T_support[:3, :3], T_B[:3, :3])
    return {
        "center_to_center_distance_m": center_distance,
        "vertical_offset_along_global_up_m": vertical,
        "lateral_alignment_error_m": lateral,
        "lateral_alignment_vector_camera_m": [float(x) for x in lateral_vec],
        "rotation_difference_deg": rot_deg,
    }


def render_frame(img, T_A, T_support, T_B, K, pts, idx, total, up_camera, metrics):
    vis = img.copy()

    base.draw_wireframe(vis, base.project(pts, T_support, K), base.COLORS["support"], 2, 0.65)
    base.draw_label(vis, base.project_center(T_support, K), "VIRTUAL SUPPORT", base.COLORS["support"])

    base.draw_wireframe(vis, base.project(pts, T_B, K), base.COLORS["target"], 3, 0.82)
    base.draw_label(vis, base.project_center(T_B, K), "DESIRED FINAL B - ON TOP", base.COLORS["target"])

    base.draw_wireframe(vis, base.project(pts, T_A, K), base.COLORS["tracked"], 3, 1.0)
    base.draw_label(vis, base.project_center(T_A, K), "TRACKED A", base.COLORS["tracked"])

    cS = base.project_center(T_support, K)
    cB = base.project_center(T_B, K)
    cA = base.project_center(T_A, K)

    # Thick center-to-center line: this is the projected image of the 3D global-up axis.
    if cS is not None and cB is not None:
        s = tuple(np.round(cS).astype(int))
        b = tuple(np.round(cB).astype(int))
        cv2.line(vis, s, b, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.arrowedLine(vis, s, b, base.COLORS["target"], 3, cv2.LINE_AA, tipLength=0.10)
        cv2.circle(vis, s, 7, base.COLORS["support"], -1, cv2.LINE_AA)
        cv2.circle(vis, b, 7, base.COLORS["target"], -1, cv2.LINE_AA)

    if cA is not None and cB is not None:
        cv2.arrowedLine(vis, tuple(np.round(cA).astype(int)), tuple(np.round(cB).astype(int)),
                        base.COLORS["path"], 2, cv2.LINE_AA, tipLength=0.04)

    base.put_text(vis, "RGB-D plane normal = GLOBAL UP; B is 0.30 m above support", 26)
    base.put_text(vis, f"frame {idx}/{total}", 50, scale=0.46)
    base.put_text(vis, "green=A | cyan=support | magenta=desired final B", 74, scale=0.43)
    base.put_text(vis, "up_cam=" + np.array2string(up_camera, precision=3, suppress_small=True), 98, scale=0.40)
    base.put_text(
        vis,
        f"3D lateral={metrics['lateral_alignment_error_m']:.4f} m | "
        f"center={metrics['center_to_center_distance_m']:.4f} m | "
        f"dR={metrics['rotation_difference_deg']:.3f} deg",
        122,
        scale=0.40,
    )
    base.put_text(
        vis,
        f"vertical along global-up={metrics['vertical_offset_along_global_up_m']:.4f} m",
        146,
        scale=0.40,
    )
    return vis


def main():
    args = parse_args()
    bundle = args.bundle
    K = np.loadtxt(bundle / "cam_K.txt").reshape(3, 3)
    timestamps = list(csv.DictReader((bundle / "timestamps.csv").open()))
    fp_csv = bundle / "foundationpose_offline" / "foundationpose_poses.csv"
    fp_rows = {str(r["frame"]): r for r in csv.DictReader(fp_csv.open())}

    records = []
    for ts in timestamps:
        row = fp_rows.get(str(ts["frame"]))
        rgb_path = bundle / ts["rgb_file"]
        depth_path = bundle / ts["depth_file"]
        if row is None or not rgb_path.exists() or not depth_path.exists():
            continue
        records.append({
            "frame_id": str(ts["frame"]),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "pose": base.load_pose(row),
        })
    if len(records) < 2:
        raise RuntimeError("not enough synchronized RGB/depth/FoundationPose records")

    first = cv2.imread(str(records[0]["rgb_path"]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError("could not read first RGB frame")
    h, w = first.shape[:2]
    dims = np.asarray(args.box_dims, dtype=np.float64)
    pts = base.box_corners(dims)

    if args.support_record_index >= 0:
        support_idx = args.support_record_index
    else:
        support_idx = choose_support_record(records, K, dims, args.box_height, w, h)
    if not (0 <= support_idx < len(records)):
        raise IndexError(f"support index out of range: {support_idx}")

    support_rec = records[support_idx]
    T_support = support_rec["pose"].copy()
    depth = cv2.imread(str(support_rec["depth_path"]), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"could not read depth: {support_rec['depth_path']}")
    if depth.ndim != 2:
        raise RuntimeError(f"expected single-channel depth, got shape {depth.shape}")

    cloud = depth_to_points(depth, K, stride=4)
    up_camera, plane_d, plane_inliers, center_height = fit_support_plane(
        cloud,
        T_support[:3, 3],
        expected_center_height=float(args.box_height) * 0.5,
        threshold=float(args.plane_threshold),
        iters=int(args.plane_iters),
    )
    T_B = make_target(T_support, up_camera, args.box_height)
    metrics = alignment_metrics(T_support, T_B, up_camera)

    out_dir = bundle / "stack_target_rgb_overlay_v3"
    out_dir.mkdir(exist_ok=True)
    out_video = out_dir / "stack_target_rgb_overlay_v3.mp4"

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("could not open video writer")

    indices = list(range(0, len(records), max(1, args.every)))
    snap = {
        indices[0]: "start",
        indices[len(indices)//2]: "mid",
        indices[-1]: "end",
    }
    written = 0
    for ridx in indices:
        img = cv2.imread(str(records[ridx]["rgb_path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        vis = render_frame(img, records[ridx]["pose"], T_support, T_B,
                           K, pts, ridx, len(records)-1, up_camera, metrics)
        writer.write(vis)
        written += 1
        if ridx in snap:
            cv2.imwrite(str(out_dir / f"stack_target_preview_{snap[ridx]}.png"), vis)
    writer.release()

    projected_support = base.project_center(T_support, K)
    projected_B = base.project_center(T_B, K)
    projected_pixel_delta = None
    if projected_support is not None and projected_B is not None:
        projected_pixel_delta = [float(x) for x in (projected_B - projected_support)]

    summary = {
        "mode": "rgbd_plane_global_up_v3_with_alignment_diagnostics",
        "support_record_index": int(support_idx),
        "support_frame_id": support_rec["frame_id"],
        "support_depth_file": str(support_rec["depth_path"].relative_to(bundle)),
        "global_up_camera": [float(x) for x in up_camera],
        "plane_d": float(plane_d),
        "plane_inliers": int(plane_inliers),
        "support_center_height_above_plane_m": float(center_height),
        "support_center_camera_m": [float(x) for x in T_support[:3, 3]],
        "desired_final_B_center_camera_m": [float(x) for x in T_B[:3, 3]],
        "projected_support_to_B_pixel_delta_uv": projected_pixel_delta,
        **metrics,
        "requested_stack_offset_m": float(args.box_height),
        "preplace_visualization": False,
        "written_frames": int(written),
        "alignment_interpretation": (
            "lateral_alignment_error_m is measured orthogonal to the fitted global-up vector in 3D. "
            "A nonzero 2D pixel shift can still appear because perspective projection does not preserve vertical image alignment."
        ),
    }
    summary_path = out_dir / "stack_target_rgb_overlay_v3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("REAL-RGB STACK OVERLAY V3 — RGB-D GLOBAL UP + ALIGNMENT DIAGNOSTICS")
    print("support record:", support_idx, "frame:", support_rec["frame_id"])
    print("global up in camera frame:", np.round(up_camera, 6))
    print("plane inliers:", plane_inliers)
    print("support center height above fitted plane [m]:", round(center_height, 4))
    print("support center:", np.round(T_support[:3, 3], 4))
    print("desired final B:", np.round(T_B[:3, 3], 4))
    print("3D center-to-center distance [m]:", f"{metrics['center_to_center_distance_m']:.6f}")
    print("3D vertical offset along global-up [m]:", f"{metrics['vertical_offset_along_global_up_m']:.6f}")
    print("3D lateral alignment error [m]:", f"{metrics['lateral_alignment_error_m']:.9f}")
    print("rotation difference [deg]:", f"{metrics['rotation_difference_deg']:.9f}")
    print("projected support->B pixel delta [u,v]:", projected_pixel_delta)
    print("written:", written)
    print("saved:", out_video)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
