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


BOX_DIMS_M = np.array([0.30, 0.30, 0.30], dtype=float)


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


def invert_transform(T: np.ndarray, name: str = "transform") -> np.ndarray:
    T = validate_transform(T, name)
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return result


def foundationpose_pose_last_from_original(
    camera_T_original: np.ndarray, center_from_original: np.ndarray
) -> np.ndarray:
    """Convert a public FoundationPose output/reseed pose to internal pose_last."""
    return validate_transform(camera_T_original, "camera_T_original") @ invert_transform(
        center_from_original, "center_from_original"
    )


def vive_alignment_candidate(
    world_T_B: np.ndarray, T_T_B: np.ndarray, V_T_T: np.ndarray
) -> np.ndarray:
    """One sample of the existing saved VIVE-world calibration equation."""
    return (
        validate_transform(world_T_B, "world_T_B")
        @ invert_transform(T_T_B, "T_T_B")
        @ invert_transform(V_T_T, "V_T_T")
    )


def tracker_root_and_camera(
    world_T_V: np.ndarray, V_T_T: np.ndarray, T_T_B: np.ndarray,
    B_T_C: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Return world_T_B, world_T_C, world_T_T using the tracker chain."""
    world_T_T = validate_transform(world_T_V, "world_T_V") @ validate_transform(
        V_T_T, "V_T_T"
    )
    world_T_B = world_T_T @ validate_transform(T_T_B, "T_T_B")
    world_T_C = (
        None if B_T_C is None
        else world_T_B @ validate_transform(B_T_C, "B_T_C")
    )
    return world_T_B, world_T_C, world_T_T


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a, dtype=float).T @ np.asarray(R_b, dtype=float)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _cube_symmetries() -> tuple[np.ndarray, ...]:
    """The 24 proper rotational symmetries of a cube."""
    import itertools

    rotations = []

    def add(R):
        R = np.asarray(R, dtype=float)
        if np.linalg.det(R) < 0.5:
            return
        if not any(np.allclose(R, old) for old in rotations):
            rotations.append(R)

    # Keep identity and +90 deg about Z first for deterministic tests.
    add(np.eye(3))
    add(np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ]))

    I = np.eye(3)
    for perm in itertools.permutations(range(3)):
        P = I[:, perm]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            R = P @ np.diag(signs)
            add(R)

    if len(rotations) != 24:
        raise RuntimeError(
            f"Expected 24 cube rotations, got {len(rotations)}"
        )

    return tuple(rotations)


BOX_SYMMETRIES = _cube_symmetries()


def cube_corners() -> np.ndarray:
    half = BOX_DIMS_M * 0.5
    return np.asarray([
        [-half[0], -half[1], -half[2]], [half[0], -half[1], -half[2]],
        [half[0], half[1], -half[2]], [-half[0], half[1], -half[2]],
        [-half[0], -half[1], half[2]], [half[0], -half[1], half[2]],
        [half[0], half[1], half[2]], [-half[0], half[1], half[2]],
    ], dtype=float)


def compose_world_box_poses(
    E_T_V: Optional[np.ndarray],
    V_T_T: Optional[np.ndarray],
    T_T_B: Optional[np.ndarray],
    B_T_C: Optional[np.ndarray],
    E_T_box: Optional[np.ndarray],
    C_T_box: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compose independent box estimates in fixed external-camera world E."""
    # This copy is intentionally independent of every G1/VIVE/FK input.
    E_T_box_ext = (
        None if E_T_box is None
        else validate_transform(E_T_box, "E_T_box").copy()
    )
    E_T_C = E_T_box_g1 = None
    if all(value is not None for value in (E_T_V, V_T_T, T_T_B, B_T_C)):
        try:
            E_T_C = (
                validate_transform(E_T_V, "E_T_V")
                @ validate_transform(V_T_T, "V_T_T")
                @ validate_transform(T_T_B, "T_T_B")
                @ validate_transform(B_T_C, "B_T_C")
            )
            if C_T_box is not None:
                E_T_box_g1 = E_T_C @ validate_transform(C_T_box, "C_T_box")
        except ValueError:
            # G1 path failure must never invalidate the independent external pose.
            E_T_C = E_T_box_g1 = None
    return E_T_box_ext, E_T_box_g1, E_T_C


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in np.asarray(points)})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return float(abs(
        np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
        - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
    ) * 0.5)


def _polygon_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if len(polygon) < 3:
        return mask
    x0 = max(0, int(np.floor(polygon[:, 0].min())))
    x1 = min(width - 1, int(np.ceil(polygon[:, 0].max())))
    y0 = max(0, int(np.floor(polygon[:, 1].min())))
    y1 = min(height - 1, int(np.ceil(polygon[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return mask
    xs, ys = np.meshgrid(
        np.arange(x0, x1 + 1, dtype=float) + 0.5,
        np.arange(y0, y1 + 1, dtype=float) + 0.5,
    )
    inside_positive = np.ones(xs.shape, dtype=bool)
    inside_negative = np.ones(xs.shape, dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        cross = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
        inside_positive &= cross >= -1e-8
        inside_negative &= cross <= 1e-8
    mask[y0:y1 + 1, x0:x1 + 1] = inside_positive | inside_negative
    return mask


@dataclass(frozen=True)
class PoseValidity:
    valid: bool
    quality: float
    age_s: float
    center_z_m: float
    image_overlap: float
    projected_area_px: float
    depth_coverage: float
    depth_consistency: float
    reason: str


def evaluate_camera_pose(
    camera_T_box: Optional[np.ndarray],
    pose_time: float,
    now: float,
    depth_m: Optional[np.ndarray],
    K: Optional[np.ndarray],
    image_shape: Optional[tuple[int, int]],
    *,
    max_age_s: float = 0.75,
) -> PoseValidity:
    """Evaluate freshness, projection, and RGB-D support for a cube pose."""
    invalid = PoseValidity(False, 0.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, "missing")
    if camera_T_box is None or depth_m is None or K is None or image_shape is None:
        return invalid
    try:
        pose = validate_transform(camera_T_box, "camera_T_box")
    except ValueError as exc:
        return PoseValidity(False, 0.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, str(exc))
    age = max(0.0, float(now) - float(pose_time))
    height, width = (int(image_shape[0]), int(image_shape[1]))
    depth = np.asarray(depth_m, dtype=float)
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)) or height <= 0 or width <= 0:
        return PoseValidity(False, 0.0, age, float(pose[2, 3]), 0.0, 0.0, 0.0, 0.0, "invalid intrinsics")
    if depth.shape != (height, width):
        return PoseValidity(False, 0.0, age, float(pose[2, 3]), 0.0, 0.0, 0.0, 0.0, "depth shape")

    points = (pose[:3, :3] @ cube_corners().T).T + pose[:3, 3]
    z = points[:, 2]
    center_z = float(pose[2, 3])
    z_plausible = 0.15 <= center_z <= 5.0 and float(z.min()) > 0.03
    if not z_plausible:
        return PoseValidity(False, 0.0, age, center_z, 0.0, 0.0, 0.0, 0.0, "implausible Z")

    uv = np.column_stack((
        K[0, 0] * points[:, 0] / z + K[0, 2],
        K[1, 1] * points[:, 1] / z + K[1, 2],
    ))
    hull = _convex_hull(uv)
    projected_area = _polygon_area(hull)
    region = _polygon_mask(hull, height, width)
    visible_pixels = int(region.sum())
    overlap = float(min(1.0, visible_pixels / max(projected_area, 1.0)))

    region_depth = depth[region]
    usable = np.isfinite(region_depth) & (region_depth > 0.05) & (region_depth < 10.0)
    coverage = float(usable.mean()) if len(region_depth) else 0.0
    margin = 0.06
    consistent = usable & (region_depth >= float(z.min()) - margin) & (
        region_depth <= float(z.max()) + margin
    )
    consistency = float(consistent.sum() / max(int(usable.sum()), 1))

    fresh_score = float(np.clip(1.0 - age / max_age_s, 0.0, 1.0))
    area_score = float(np.clip(visible_pixels / 4000.0, 0.0, 1.0))
    coverage_score = float(np.clip(coverage / 0.5, 0.0, 1.0))
    quality = float(np.clip(fresh_score * (
        0.25 * overlap + 0.20 * area_score
        + 0.20 * coverage_score + 0.35 * consistency
    ),
        0.0, 1.0,
    ))
    checks = (
        (age <= max_age_s, "stale"),
        (overlap >= 0.30, "outside image"),
        (visible_pixels >= 200, "projection too small"),
        (coverage >= 0.10, "insufficient depth"),
        (consistency >= 0.50, "depth mismatch"),
    )
    reason = next((reason for passed, reason in checks if not passed), "ok")
    return PoseValidity(
        reason == "ok", quality, age, center_z, overlap, projected_area,
        coverage, consistency, reason,
    )


def closest_cube_equivalent(reference_R: np.ndarray, candidate_R: np.ndarray) -> np.ndarray:
    return min(
        (np.asarray(candidate_R) @ symmetry for symmetry in BOX_SYMMETRIES),
        key=lambda rotation: rotation_error_deg(reference_R, rotation),
    )


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    values, vectors = np.linalg.eigh(np.asarray([
        [R[0, 0] - R[1, 1] - R[2, 2], R[0, 1] + R[1, 0], R[0, 2] + R[2, 0], R[2, 1] - R[1, 2]],
        [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], R[1, 2] + R[2, 1], R[0, 2] - R[2, 0]],
        [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], R[1, 0] - R[0, 1]],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R.trace()],
    ]) / 3.0)
    q_xyzw = vectors[:, int(np.argmax(values))]
    q = np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return q / np.linalg.norm(q)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def weighted_rotation_average(R_a: np.ndarray, R_b: np.ndarray, weight_a: float, weight_b: float) -> np.ndarray:
    q_a = _matrix_to_quaternion(R_a)
    q_b = _matrix_to_quaternion(R_b)
    if np.dot(q_a, q_b) < 0:
        q_b = -q_b
    accumulator = max(weight_a, 1e-9) * np.outer(q_a, q_a) + max(weight_b, 1e-9) * np.outer(q_b, q_b)
    values, vectors = np.linalg.eigh(accumulator)
    return _quaternion_to_matrix(vectors[:, int(np.argmax(values))])


@dataclass(frozen=True)
class FusionResult:
    pose: Optional[np.ndarray]
    valid: bool
    source: str
    held: bool = False


def fuse_world_poses(
    external_pose: Optional[np.ndarray], external_valid: bool, external_quality: float,
    g1_pose: Optional[np.ndarray], g1_valid: bool, g1_quality: float,
    *, last_pose: Optional[np.ndarray] = None, last_pose_time: float = 0.0,
    now: float = 0.0, hold_s: float = 0.25,
) -> FusionResult:
    if external_valid and external_pose is not None and g1_valid and g1_pose is not None:
        ext = validate_transform(external_pose, "external_pose")
        g1 = validate_transform(g1_pose, "g1_pose")
        total = max(float(external_quality) + float(g1_quality), 1e-9)
        fused = np.eye(4)
        fused[:3, 3] = (
            float(external_quality) * ext[:3, 3] + float(g1_quality) * g1[:3, 3]
        ) / total
        aligned_g1_R = closest_cube_equivalent(ext[:3, :3], g1[:3, :3])
        fused[:3, :3] = weighted_rotation_average(
            ext[:3, :3], aligned_g1_R, external_quality, g1_quality
        )
        return FusionResult(fused, True, "BOTH")
    if external_valid and external_pose is not None:
        return FusionResult(validate_transform(external_pose, "external_pose").copy(), True, "EXTERNAL")
    if g1_valid and g1_pose is not None:
        return FusionResult(validate_transform(g1_pose, "g1_pose").copy(), True, "G1")
    if last_pose is not None and float(now) - float(last_pose_time) <= hold_s:
        return FusionResult(validate_transform(last_pose, "last_pose").copy(), True, "NONE", True)
    return FusionResult(None, False, "NONE")


@dataclass(frozen=True)
class BoxDisagreement:
    translation_mm: float
    rotation_raw_deg: float
    rotation_symmetry_deg: float


def box_disagreement(world_T_box_a: np.ndarray, world_T_box_b: np.ndarray) -> BoxDisagreement:
    a = validate_transform(world_T_box_a, "world_T_box_a")
    b = validate_transform(world_T_box_b, "world_T_box_b")
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
    E_T_box_ext: Optional[np.ndarray],
    E_T_box_g1: Optional[np.ndarray],
    E_T_box_fused: Optional[np.ndarray],
) -> None:
    """Persist each currently available transform without fabricating missing data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "E_T_box": E_T_box,
        "C_T_box": C_T_box,
        "E_T_box_ext": E_T_box_ext,
        "E_T_box_g1": E_T_box_g1,
        "E_T_box_fused": E_T_box_fused,
    }
    for name, value in values.items():
        path = output_dir / f"{name}.txt"
        if value is not None:
            np.savetxt(path, validate_transform(value, name))
        else:
            path.unlink(missing_ok=True)
