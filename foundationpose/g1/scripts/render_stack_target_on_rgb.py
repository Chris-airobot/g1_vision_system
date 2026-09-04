#!/usr/bin/env python3
"""Render stacking-target geometry directly on the real recorded RGB sequence.

This is a visualization-only tool. It uses the saved FoundationPose trajectory
from an offline RGB-D bundle; it does NOT run FoundationPose, Isaac Sim, or any
robot command.

The recorded bundle used for current development was captured with the G1/camera
stationary. Therefore a pose taken from one frame can be reused as a fixed
camera-frame *virtual support box* for visualization purposes.

Overlays:
  - tracked box A: current saved FoundationPose pose for each frame
  - virtual support: one selected historical FoundationPose pose, held fixed
  - target B: support pose shifted one box height along support local +Z
  - pre-place: B shifted by the requested clearance along local +Z

Outputs:
  - stack_target_rgb_overlay.mp4
  - stack_target_preview_start.png
  - stack_target_preview_mid.png
  - stack_target_preview_end.png
  - stack_target_rgb_overlay_summary.json
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "g1/data/offline_bundle_20260813_140853"

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# BGR colors for OpenCV.
COLORS = {
    "tracked": (0, 220, 0),
    "support": (255, 220, 0),
    "target": (255, 0, 255),
    "preplace": (0, 220, 255),
    "path": (255, 255, 255),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay tracked box A and virtual stacking target geometry on real RGB frames."
    )
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--every", type=int, default=3, help="Write every Nth source frame.")
    p.add_argument("--fps", type=float, default=15.0, help="Output playback FPS.")
    p.add_argument(
        "--box-dims", type=float, nargs=3, default=(0.40, 0.30, 0.30),
        metavar=("X", "Y", "Z"), help="Box dimensions in metres."
    )
    p.add_argument(
        "--box-height", type=float, default=0.30,
        help="Center-to-center stack offset along support local +Z [m]."
    )
    p.add_argument(
        "--preplace-clearance", type=float, default=0.12,
        help="Pre-place clearance above B along target local +Z [m]."
    )
    p.add_argument(
        "--support-record-index", type=int, default=-1,
        help=(
            "Index in the valid saved FoundationPose record list used as the fixed virtual support. "
            "Default -1 chooses one automatically for good on-image visibility."
        ),
    )
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def load_pose(row):
    T = np.eye(4, dtype=np.float64)
    for r in range(4):
        for c in range(4):
            T[r, c] = float(row[f"t{r}{c}"])
    return T


def local_translation(xyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def box_corners(dims):
    hx, hy, hz = np.asarray(dims, dtype=np.float64) * 0.5
    return np.asarray([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ], dtype=np.float64)


def project(points_obj, T_cam_obj, K):
    pc = (T_cam_obj[:3, :3] @ points_obj.T).T + T_cam_obj[:3, 3]
    if np.any(pc[:, 2] <= 1e-6):
        return None
    uv = np.empty((len(pc), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    return uv


def project_center(T, K):
    p = T[:3, 3]
    if p[2] <= 1e-6:
        return None
    return np.asarray([
        K[0, 0] * p[0] / p[2] + K[0, 2],
        K[1, 1] * p[1] / p[2] + K[1, 2],
    ], dtype=np.float64)


def in_image(uv, width, height, margin=8):
    if uv is None:
        return False
    return margin <= uv[0] < width - margin and margin <= uv[1] < height - margin


def visible_corner_fraction(uv, width, height):
    if uv is None:
        return 0.0
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return float(np.mean(inside))


def draw_wireframe(img, uv, color, thickness=2, alpha=1.0):
    if uv is None:
        return
    p = np.round(uv).astype(np.int32)
    layer = img.copy() if alpha < 1.0 else img
    for a, b in EDGES:
        cv2.line(layer, tuple(p[a]), tuple(p[b]), color, thickness, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(layer, alpha, img, 1.0 - alpha, 0.0, dst=img)


def draw_label(img, uv, text, color):
    if uv is None:
        return
    x, y = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, color, 2, cv2.LINE_AA)


def put_text(img, text, y, color=(255, 255, 255), scale=0.50):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def source_fps_from_timestamps(rows):
    times = []
    for row in rows:
        try:
            times.append(float(row["relative_s"]))
        except (KeyError, TypeError, ValueError):
            pass
    if len(times) < 2:
        return 0.0
    dts = np.diff(np.asarray(times, dtype=np.float64))
    dts = dts[dts > 1e-6]
    if len(dts) == 0:
        return 0.0
    return float(1.0 / np.median(dts))


def choose_support_pose(records, K, dims, height_m, clearance_m, width, height):
    """Pick a historical pose that keeps support/B/pre-place visible and separated from frame-0 A."""
    pts = box_corners(dims)
    first_pose = records[0]["pose"]
    first_center = project_center(first_pose, K)

    best = None
    # Sample at most ~300 candidate records to keep this instant even on long bags.
    stride = max(1, len(records) // 300)
    for idx in range(0, len(records), stride):
        T_support = records[idx]["pose"]
        T_target = T_support @ local_translation([0.0, 0.0, height_m])
        T_preplace = T_target @ local_translation([0.0, 0.0, clearance_m])

        support_uv = project(pts, T_support, K)
        target_uv = project(pts, T_target, K)
        pre_uv = project(pts, T_preplace, K)
        support_c = project_center(T_support, K)
        target_c = project_center(T_target, K)
        pre_c = project_center(T_preplace, K)

        fractions = [
            visible_corner_fraction(support_uv, width, height),
            visible_corner_fraction(target_uv, width, height),
            visible_corner_fraction(pre_uv, width, height),
        ]
        if min(fractions) < 0.50:
            continue
        if not (in_image(support_c, width, height) and
                in_image(target_c, width, height) and
                in_image(pre_c, width, height)):
            continue

        separation = 0.0 if first_center is None else float(np.linalg.norm(support_c - first_center))
        visibility = float(sum(fractions))
        score = separation + 120.0 * visibility
        if best is None or score > best[0]:
            best = (score, idx)

    if best is not None:
        return best[1]

    # Guaranteed fallback to a valid historical pose even if the derived target is partly off-screen.
    return len(records) // 2


def render_frame(bgr, T_A, T_support, T_target, T_preplace, K, pts, frame_i, total_i):
    vis = bgr.copy()

    # Virtual geometry first, slightly transparent; real tracked A last and solid.
    for label, T, key, alpha, thickness in [
        ("VIRTUAL support", T_support, "support", 0.62, 2),
        ("TARGET B", T_target, "target", 0.72, 2),
        ("PRE-PLACE", T_preplace, "preplace", 0.62, 2),
    ]:
        uv = project(pts, T, K)
        draw_wireframe(vis, uv, COLORS[key], thickness=thickness, alpha=alpha)
        draw_label(vis, project_center(T, K), label, COLORS[key])

    uv_A = project(pts, T_A, K)
    draw_wireframe(vis, uv_A, COLORS["tracked"], thickness=3, alpha=1.0)
    draw_label(vis, project_center(T_A, K), "TRACKED A", COLORS["tracked"])

    cA = project_center(T_A, K)
    cB = project_center(T_target, K)
    if cA is not None and cB is not None:
        a = tuple(np.round(cA).astype(int))
        b = tuple(np.round(cB).astype(int))
        cv2.arrowedLine(vis, a, b, COLORS["path"], 2, cv2.LINE_AA, tipLength=0.04)

    put_text(vis, "REAL RGB + saved FoundationPose + VIRTUAL stack target", 26)
    put_text(vis, f"frame {frame_i}/{total_i}", 50, scale=0.46)
    put_text(vis, "green=A tracked | cyan=support | magenta=B | yellow=pre-place", 74, scale=0.43)
    return vis


def main():
    args = parse_args()
    bundle = args.bundle
    if args.every < 1:
        raise ValueError("--every must be >= 1")
    if args.box_height <= 0 or args.preplace_clearance < 0:
        raise ValueError("invalid box height / pre-place clearance")

    K = np.loadtxt(bundle / "cam_K.txt").reshape(3, 3)
    timestamps = list(csv.DictReader((bundle / "timestamps.csv").open()))
    fp_csv = bundle / "foundationpose_offline" / "foundationpose_poses.csv"
    if not fp_csv.exists():
        raise FileNotFoundError(f"Missing saved FoundationPose trajectory: {fp_csv}")
    fp_rows = {str(r["frame"]): r for r in csv.DictReader(fp_csv.open())}

    metadata = {}
    metadata_path = bundle / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    stationary = metadata.get("robot_stationary_during_capture", None)
    if stationary is not True:
        print("WARNING: bundle metadata does not confirm a stationary camera.")
        print("A historical camera-frame pose should not be treated as a fixed support if the camera moved.")

    records = []
    for i, ts in enumerate(timestamps):
        frame_id = str(ts["frame"])
        row = fp_rows.get(frame_id)
        if row is None:
            continue
        rgb_path = bundle / ts["rgb_file"]
        if not rgb_path.exists():
            continue
        records.append({
            "timestamp_index": i,
            "frame_id": frame_id,
            "rgb_path": rgb_path,
            "relative_s": ts.get("relative_s", ""),
            "pose": load_pose(row),
        })

    if len(records) < 2:
        raise RuntimeError("Need at least two valid RGB + saved-pose records.")

    first_img = cv2.imread(str(records[0]["rgb_path"]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError(f"Could not read {records[0]['rgb_path']}")
    h, w = first_img.shape[:2]

    dims = np.asarray(args.box_dims, dtype=np.float64)
    pts = box_corners(dims)

    if args.support_record_index >= 0:
        support_idx = args.support_record_index
        if support_idx >= len(records):
            raise IndexError(
                f"--support-record-index {support_idx} is out of range; valid 0..{len(records)-1}"
            )
    else:
        support_idx = choose_support_pose(
            records, K, dims, args.box_height, args.preplace_clearance, w, h
        )

    support_rec = records[support_idx]
    T_support = support_rec["pose"].copy()
    T_target = T_support @ local_translation([0.0, 0.0, args.box_height])
    T_preplace = T_target @ local_translation([0.0, 0.0, args.preplace_clearance])

    out_dir = bundle / "stack_target_rgb_overlay"
    out_dir.mkdir(exist_ok=True)
    output = args.output if args.output is not None else out_dir / "stack_target_rgb_overlay.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    source_fps = source_fps_from_timestamps(timestamps)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    write_record_indices = list(range(0, len(records), args.every))
    snapshot_targets = {
        write_record_indices[0]: "start",
        write_record_indices[len(write_record_indices) // 2]: "mid",
        write_record_indices[-1]: "end",
    }

    written = 0
    for ridx in write_record_indices:
        rec = records[ridx]
        bgr = cv2.imread(str(rec["rgb_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        vis = render_frame(
            bgr, rec["pose"], T_support, T_target, T_preplace,
            K, pts, rec["timestamp_index"], len(timestamps) - 1,
        )
        writer.write(vis)
        written += 1
        if ridx in snapshot_targets:
            cv2.imwrite(
                str(out_dir / f"stack_target_preview_{snapshot_targets[ridx]}.png"), vis
            )

    writer.release()
    if written == 0:
        raise RuntimeError("No video frames were written.")

    summary = {
        "mode": "real_rgb_saved_fp_virtual_stack_overlay_v1",
        "visualization_only": True,
        "no_foundationpose_inference": True,
        "no_robot_commands": True,
        "bundle": str(bundle),
        "camera_stationary_confirmed_by_metadata": stationary is True,
        "valid_records": len(records),
        "written_frames": written,
        "source_fps": source_fps,
        "output_fps": float(args.fps),
        "every": int(args.every),
        "box_dims_m": [float(v) for v in dims],
        "box_height_m": float(args.box_height),
        "preplace_clearance_m": float(args.preplace_clearance),
        "support_record_index": int(support_idx),
        "support_source_frame": support_rec["frame_id"],
        "support_source_timestamp_index": int(support_rec["timestamp_index"]),
        "support_source_relative_s": support_rec["relative_s"],
        "T_camera_support": T_support.tolist(),
        "T_camera_target_B": T_target.tolist(),
        "T_camera_preplace": T_preplace.tolist(),
        "video": str(output),
        "previews": [
            str(out_dir / "stack_target_preview_start.png"),
            str(out_dir / "stack_target_preview_mid.png"),
            str(out_dir / "stack_target_preview_end.png"),
        ],
        "note": (
            "The support pose is virtual and comes from one historical pose of the same tracked box. "
            "This is valid only as a visualization because the recorded camera was stationary. "
            "A future multi-box recording will replace it with the actual top/support-box pose."
        ),
    }
    summary_path = out_dir / "stack_target_rgb_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("REAL-RGB STACK TARGET OVERLAY — visualization only")
    print("bundle:", bundle)
    print("camera stationary metadata:", stationary)
    print("valid saved-pose records:", len(records))
    print("virtual support chosen from record index:", support_idx)
    print("virtual support source frame:", support_rec["frame_id"])
    print("source FPS:", f"{source_fps:.2f}")
    print("written video frames:", written)
    print("saved video:", output)
    print("saved preview:", out_dir / "stack_target_preview_start.png")
    print("saved preview:", out_dir / "stack_target_preview_mid.png")
    print("saved preview:", out_dir / "stack_target_preview_end.png")
    print("saved summary:", summary_path)


if __name__ == "__main__":
    main()
