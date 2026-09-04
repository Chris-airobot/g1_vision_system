#!/usr/bin/env python3
"""Software-only stacking target generator.

Consumes the robot-frame pose of the currently selected/carried box and a
mock robot-frame pose for the current top box in a stack.

Outputs:
  T_robot_carried_box.txt
  T_robot_top_box.txt
  T_robot_place.txt
  T_robot_preplace.txt
  stack_target_summary.json

No robot commands are sent.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARRIED = ROOT / "g1/results/root_pipeline_demo/T_robot_box.txt"
DEFAULT_OUT = ROOT / "g1/results/stack_target_demo"


def yaw_matrix(yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def pose_from_xyz_yaw(xyz, yaw_rad):
    T = np.eye(4, dtype=float)
    T[:3, :3] = yaw_matrix(yaw_rad)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def local_translation(xyz):
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def validate_transform(name, T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has invalid homogeneous last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not approximately orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=2e-3):
        raise ValueError(f"{name} rotation determinant is not approximately +1")


def xyz(T):
    return [float(v) for v in T[:3, 3]]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a mock stack placement target from the existing robot-frame box pose."
    )
    p.add_argument(
        "--carried-pose",
        type=Path,
        default=DEFAULT_CARRIED,
        help="4x4 T_robot_box from offline_root_pipeline.py",
    )
    p.add_argument(
        "--top-xyz",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[1.20, 0.30, 0.15],
        help="Mock top-box center position in robot frame [m]. Default: 1.20 0.30 0.15",
    )
    p.add_argument(
        "--top-yaw-deg",
        type=float,
        default=20.0,
        help="Mock top-box yaw in robot frame [deg]. Default: 20",
    )
    p.add_argument(
        "--box-height-m",
        type=float,
        default=0.30,
        help="Identical box height [m]. Default: 0.30",
    )
    p.add_argument(
        "--preplace-clearance-m",
        type=float,
        default=0.12,
        help="Pre-place clearance above final target along target local +Z [m]. Default: 0.12",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.carried_pose.exists():
        raise FileNotFoundError(
            f"Missing carried-object pose: {args.carried_pose}\n"
            "Run g1/scripts/offline_root_pipeline.py first."
        )
    if args.box_height_m <= 0:
        raise ValueError("--box-height-m must be > 0")
    if args.preplace_clearance_m < 0:
        raise ValueError("--preplace-clearance-m must be >= 0")

    T_robot_carried = np.loadtxt(args.carried_pose).reshape(4, 4)
    validate_transform("T_robot_carried", T_robot_carried)

    # Step 1 mock stack pose. Later this will be replaced by the pose of the
    # actual top/support box returned by the multi-object perception pipeline.
    T_robot_top = pose_from_xyz_yaw(
        args.top_xyz, math.radians(args.top_yaw_deg)
    )
    validate_transform("T_robot_top", T_robot_top)

    # Identical upright boxes: center-to-center stacking offset is one full
    # box height along the top box's local +Z axis. Multiplication on the
    # right intentionally applies the displacement in the top-box frame.
    T_top_to_place = local_translation([0.0, 0.0, args.box_height_m])
    T_robot_place = T_robot_top @ T_top_to_place

    # Pre-place keeps the same desired final orientation and moves upward
    # along the target's local +Z axis.
    T_place_to_preplace = local_translation(
        [0.0, 0.0, args.preplace_clearance_m]
    )
    T_robot_preplace = T_robot_place @ T_place_to_preplace

    validate_transform("T_robot_place", T_robot_place)
    validate_transform("T_robot_preplace", T_robot_preplace)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out / "T_robot_carried_box.txt", T_robot_carried, fmt="%.8f")
    np.savetxt(args.out / "T_robot_top_box.txt", T_robot_top, fmt="%.8f")
    np.savetxt(args.out / "T_robot_place.txt", T_robot_place, fmt="%.8f")
    np.savetxt(args.out / "T_robot_preplace.txt", T_robot_preplace, fmt="%.8f")

    carry_to_place_m = float(
        np.linalg.norm(T_robot_place[:3, 3] - T_robot_carried[:3, 3])
    )
    place_to_top_m = float(
        np.linalg.norm(T_robot_place[:3, 3] - T_robot_top[:3, 3])
    )
    preplace_to_place_m = float(
        np.linalg.norm(T_robot_preplace[:3, 3] - T_robot_place[:3, 3])
    )

    summary = {
        "mode": "software_only_mock_stack_target_v1",
        "no_robot_commands": True,
        "carried_pose_source": str(args.carried_pose),
        "box_height_m": float(args.box_height_m),
        "preplace_clearance_m": float(args.preplace_clearance_m),
        "mock_top_box": {
            "xyz_robot_m": [float(v) for v in args.top_xyz],
            "yaw_deg": float(args.top_yaw_deg),
        },
        "poses": {
            "carried_object_A_xyz_robot_m": xyz(T_robot_carried),
            "top_box_xyz_robot_m": xyz(T_robot_top),
            "placement_B_xyz_robot_m": xyz(T_robot_place),
            "preplace_xyz_robot_m": xyz(T_robot_preplace),
        },
        "sanity": {
            "carried_A_to_place_B_distance_m": carry_to_place_m,
            "top_center_to_place_center_distance_m": place_to_top_m,
            "preplace_to_place_distance_m": preplace_to_place_m,
            "expected_top_to_place_m": float(args.box_height_m),
            "expected_preplace_clearance_m": float(args.preplace_clearance_m),
        },
        "next_step": (
            "Replace the mock T_robot_top_box with the FoundationPose pose of "
            "the detected top/support box; keep the same target-generation math."
        ),
    }
    (args.out / "stack_target_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("STACK TARGET DEMO — software only, no robot commands")
    print("carried object A xyz:", np.round(T_robot_carried[:3, 3], 4))
    print("mock top box xyz:    ", np.round(T_robot_top[:3, 3], 4))
    print("placement B xyz:     ", np.round(T_robot_place[:3, 3], 4))
    print("pre-place xyz:        ", np.round(T_robot_preplace[:3, 3], 4))
    print(f"A -> B distance:       {carry_to_place_m:.4f} m")
    print(
        f"top -> B distance:     {place_to_top_m:.4f} m "
        f"(expected {args.box_height_m:.4f})"
    )
    print(
        f"pre-place clearance:   {preplace_to_place_m:.4f} m "
        f"(expected {args.preplace_clearance_m:.4f})"
    )
    print("saved:", args.out)


if __name__ == "__main__":
    main()
