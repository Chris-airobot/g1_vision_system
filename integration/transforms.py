"""Hardware-free transform and cuboid comparison helpers.

Convention throughout the integration is ``A_T_B``: a homogeneous transform
which maps coordinates expressed in frame B into frame A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


BOX_DIMS_M = np.array([0.40, 0.30, 0.30], dtype=float)


def validate_transform(T: np.ndarray, name: str = "transform") -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
        raise ValueError(f"{name} rotation determinant is not +1")
    return T


def compose_box_poses(
    K_T_E: Optional[np.ndarray],
    E_T_box: Optional[np.ndarray],
    K_T_C: Optional[np.ndarray],
    C_T_box: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compose each camera's box result independently into K."""
    K_T_box_ext = None
    K_T_box_g1 = None
    if K_T_E is not None and E_T_box is not None:
        K_T_box_ext = validate_transform(K_T_E, "K_T_E") @ validate_transform(
            E_T_box, "E_T_box"
        )
    if K_T_C is not None and C_T_box is not None:
        K_T_box_g1 = validate_transform(K_T_C, "K_T_C") @ validate_transform(
            C_T_box, "C_T_box"
        )
    return K_T_box_ext, K_T_box_g1


def invert_transform(T: np.ndarray, name: str = "transform") -> np.ndarray:
    T = validate_transform(T, name)
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return result


def vive_alignment_candidate(
    K_T_B_vision: np.ndarray, T_T_B: np.ndarray, V_T_T: np.ndarray
) -> np.ndarray:
    """One sample of the unchanged legacy automatic-alignment equation."""
    return (
        validate_transform(K_T_B_vision, "K_T_B_vision")
        @ invert_transform(T_T_B, "T_T_B")
        @ invert_transform(V_T_T, "V_T_T")
    )


def tracker_root_and_camera(
    K_T_V: np.ndarray, V_T_T: np.ndarray, T_T_B: np.ndarray,
    B_T_C: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Return K_T_B, K_T_C, K_T_T using the unchanged tracker chain."""
    K_T_T = validate_transform(K_T_V, "K_T_V") @ validate_transform(
        V_T_T, "V_T_T"
    )
    K_T_B = K_T_T @ validate_transform(T_T_B, "T_T_B")
    K_T_C = (
        None if B_T_C is None
        else K_T_B @ validate_transform(B_T_C, "B_T_C")
    )
    return K_T_B, K_T_C, K_T_T


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a, dtype=float).T @ np.asarray(R_b, dtype=float)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rx(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot180_about_yz_axis(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    axis = np.array([0.0, math.cos(a), math.sin(a)], dtype=float)
    return 2.0 * np.outer(axis, axis) - np.eye(3)


# Reused from foundationpose/g1/scripts/compare_offline_fp_apriltag.py.
# These are the 8 proper rotations (D4) of a 0.40 x 0.30 x 0.30 m cuboid.
BOX_SYMMETRIES = tuple(
    [_rx(a) for a in (0, 90, 180, 270)]
    + [_rot180_about_yz_axis(a) for a in (0, 45, 90, 135)]
)


@dataclass(frozen=True)
class BoxDisagreement:
    translation_mm: float
    rotation_raw_deg: float
    rotation_symmetry_deg: float


def box_disagreement(K_T_box_a: np.ndarray, K_T_box_b: np.ndarray) -> BoxDisagreement:
    a = validate_transform(K_T_box_a, "K_T_box_a")
    b = validate_transform(K_T_box_b, "K_T_box_b")
    translation_mm = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000.0)
    raw = rotation_error_deg(a[:3, :3], b[:3, :3])
    symmetry = min(
        rotation_error_deg(a[:3, :3], b[:3, :3] @ S) for S in BOX_SYMMETRIES
    )
    return BoxDisagreement(translation_mm, raw, symmetry)


def save_latest_transforms(
    output_dir: Path,
    *,
    E_T_box: Optional[np.ndarray],
    C_T_box: Optional[np.ndarray],
    K_T_box_ext: Optional[np.ndarray],
    K_T_box_g1: Optional[np.ndarray],
) -> None:
    """Persist each currently available transform without fabricating missing data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "E_T_box": E_T_box,
        "C_T_box": C_T_box,
        "K_T_box_ext": K_T_box_ext,
        "K_T_box_g1": K_T_box_g1,
    }
    for name, value in values.items():
        if value is not None:
            np.savetxt(output_dir / f"{name}.txt", validate_transform(value, name))
