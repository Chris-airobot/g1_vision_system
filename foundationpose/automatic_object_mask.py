"""Automatic first-frame object masks from aligned depth and camera intrinsics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class AutomaticMaskError(RuntimeError):
    """Raised when a reliable foreground object mask cannot be produced."""


@dataclass(frozen=True)
class AutomaticMaskConfig:
    """Geometry thresholds, expressed in metres unless noted otherwise."""

    workspace_roi: tuple[int, int, int, int] | None = None
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0
    table_ransac_threshold_m: float = 0.008
    min_object_height_m: float = 0.012
    max_object_height_m: float = 0.30
    cluster_tolerance_m: float = 0.010
    min_cluster_pixels: int = 200
    ransac_iterations: int = 200
    morphology_kernel_px: int = 5


def _workspace_bounds(
    shape: tuple[int, int], roi: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int]:
    height, width = shape
    if roi is None:
        return 0, 0, width, height
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise AutomaticMaskError(
            f"Workspace ROI {roi} is outside the {width}x{height} image."
        )
    return roi


def _fit_table_plane(
    points: np.ndarray, threshold_m: float, iterations: int
) -> tuple[np.ndarray, float]:
    if len(points) < 3:
        raise AutomaticMaskError("Not enough valid depth points to fit the table.")

    rng = np.random.default_rng(0)
    sample = points
    if len(sample) > 20000:
        sample = sample[rng.choice(len(sample), 20000, replace=False)]

    best_normal = None
    best_offset = 0.0
    best_count = 0
    for _ in range(iterations):
        chosen = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(chosen[1] - chosen[0], chosen[2] - chosen[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            continue
        normal /= length
        offset = -float(normal @ chosen[0])
        count = int(np.count_nonzero(np.abs(sample @ normal + offset) < threshold_m))
        if count > best_count:
            best_normal, best_offset, best_count = normal, offset, count

    if best_normal is None:
        raise AutomaticMaskError("Could not fit a table plane in the workspace ROI.")

    inliers = np.abs(points @ best_normal + best_offset) < threshold_m
    if np.count_nonzero(inliers) < 3:
        raise AutomaticMaskError("Too few table-plane inliers were found.")

    table_points = points[inliers]
    centroid = table_points.mean(axis=0)
    covariance = (table_points - centroid).T @ (table_points - centroid)
    _, _, vh = np.linalg.svd(covariance, full_matrices=False)
    normal = vh[-1]
    offset = -float(normal @ centroid)

    # Orient the normal toward the camera, so points above the table are positive.
    if offset < 0:
        normal = -normal
        offset = -offset
    return normal, offset


def _largest_3d_cluster(
    points: np.ndarray, tolerance_m: float, min_pixels: int
) -> np.ndarray:
    voxels = np.floor(points / tolerance_m).astype(np.int32)
    unique_voxels, inverse, counts = np.unique(
        voxels, axis=0, return_inverse=True, return_counts=True
    )
    lookup = {tuple(voxel): i for i, voxel in enumerate(unique_voxels)}
    visited = np.zeros(len(unique_voxels), dtype=bool)
    best_voxels: list[int] = []
    best_pixels = 0
    neighbours = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    for start in range(len(unique_voxels)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        component = []
        pixel_count = 0
        while queue:
            index = queue.popleft()
            component.append(index)
            pixel_count += int(counts[index])
            x, y, z = unique_voxels[index]
            for dx, dy, dz in neighbours:
                neighbour = lookup.get((x + dx, y + dy, z + dz))
                if neighbour is not None and not visited[neighbour]:
                    visited[neighbour] = True
                    queue.append(neighbour)
        if pixel_count > best_pixels:
            best_voxels = component
            best_pixels = pixel_count

    if best_pixels < min_pixels:
        raise AutomaticMaskError(
            f"Largest object cluster has {best_pixels} pixels; need {min_pixels}."
        )
    return np.isin(inverse, np.asarray(best_voxels, dtype=np.int32))


def generate_object_mask(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    config: AutomaticMaskConfig | None = None,
) -> np.ndarray:
    """Return a uint8 (0/255) mask for the main object above the table."""
    config = config or AutomaticMaskConfig()
    if depth_m.ndim != 2:
        raise AutomaticMaskError("Depth must be a single-channel image.")
    if camera_matrix.shape != (3, 3):
        raise AutomaticMaskError("Camera matrix must have shape (3, 3).")
    if config.min_depth_m >= config.max_depth_m:
        raise AutomaticMaskError("Minimum depth must be below maximum depth.")
    if config.min_object_height_m >= config.max_object_height_m:
        raise AutomaticMaskError("Minimum object height must be below maximum height.")
    if config.cluster_tolerance_m <= 0 or config.table_ransac_threshold_m <= 0:
        raise AutomaticMaskError("Geometry thresholds must be positive.")
    if config.min_cluster_pixels <= 0:
        raise AutomaticMaskError("Minimum cluster size must be positive.")

    x0, y0, x1, y1 = _workspace_bounds(depth_m.shape, config.workspace_roi)
    workspace_depth = depth_m[y0:y1, x0:x1]
    valid = (
        np.isfinite(workspace_depth)
        & (workspace_depth >= config.min_depth_m)
        & (workspace_depth <= config.max_depth_m)
    )
    rows, columns = np.nonzero(valid)
    if len(rows) < config.min_cluster_pixels:
        raise AutomaticMaskError("Workspace ROI contains too few valid depth pixels.")

    image_u = columns + x0
    image_v = rows + y0
    z = workspace_depth[rows, columns].astype(np.float64)
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    x = (image_u - cx) * z / fx
    y = (image_v - cy) * z / fy
    points = np.column_stack((x, y, z))

    normal, offset = _fit_table_plane(
        points, config.table_ransac_threshold_m, config.ransac_iterations
    )
    height = points @ normal + offset
    object_candidates = (
        (height >= config.min_object_height_m)
        & (height <= config.max_object_height_m)
    )
    candidate_points = points[object_candidates]
    if len(candidate_points) < config.min_cluster_pixels:
        raise AutomaticMaskError("No object-sized foreground was found above the table.")

    selected = _largest_3d_cluster(
        candidate_points, config.cluster_tolerance_m, config.min_cluster_pixels
    )
    selected_points = candidate_points[selected]

    # Project the selected 3D cluster back through K into aligned image space.
    projected_u = np.rint(fx * selected_points[:, 0] / selected_points[:, 2] + cx)
    projected_v = np.rint(fy * selected_points[:, 1] / selected_points[:, 2] + cy)
    projected_u = projected_u.astype(np.int32)
    projected_v = projected_v.astype(np.int32)
    inside = (
        (projected_u >= x0)
        & (projected_u < x1)
        & (projected_v >= y0)
        & (projected_v < y1)
    )
    mask = np.zeros(depth_m.shape, dtype=np.uint8)
    mask[projected_v[inside], projected_u[inside]] = 255

    kernel_size = max(1, int(config.morphology_kernel_px))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def save_mask_debug(
    path: Path, color_bgr: np.ndarray, mask: np.ndarray
) -> None:
    """Save RGB, binary mask, and masked RGB as one side-by-side image."""
    if color_bgr.shape[:2] != mask.shape:
        raise AutomaticMaskError("RGB and mask dimensions do not match.")
    rgb_panel = color_bgr.copy()
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked = cv2.bitwise_and(color_bgr, color_bgr, mask=mask)
    for panel, label in (
        (rgb_panel, "RGB"),
        (mask_bgr, "OBJECT MASK"),
        (masked, "MASKED OBJECT"),
    ):
        cv2.putText(
            panel,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    debug = np.concatenate((rgb_panel, mask_bgr, masked), axis=1)
    if not cv2.imwrite(str(path), debug):
        raise AutomaticMaskError(f"Failed to write mask debug image: {path}")
