#!/usr/bin/env python3
"""Replay one recorded RGB-D camera through FoundationPose and validation.

This is an offline diagnostic. It deliberately has no imports or connections
for RealSense, ZMQ, VIVE, or G1 LowState.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.transforms import (
    LOST,
    PARTIAL,
    TRACKING,
    PoseValidity,
    cube_corners,
    evaluate_camera_pose,
)


RECORDER_COLUMNS = [
    "frame",
    "phase_index",
    "phase",
    "sample_monotonic",
    "wall_time",
    "g1_seq",
    "g1_timestamp",
    "external_seq",
    "external_timestamp",
    "pair_dt_ms",
    "g1_rgb",
    "g1_depth",
    "external_rgb",
    "external_depth",
    "mode_machine",
    "lowstate_age_ms",
    "q_json",
    "B_T_C_json",
    "vive_status",
    "vive_age_ms",
    "V_T_T_json",
]
POSE_FIELDS = [field.name for field in fields(PoseValidity)]
POSE_MATRIX_FIELDS = [
    f"raw_camera_T_box_r{row}c{column}"
    for row in range(4)
    for column in range(4)
]
RESULT_FIELDS = [
    "recording_frame_id",
    "camera",
    "recorded_camera_timestamp",
    "phase_index",
    "phase",
    "rgb_path",
    "depth_path",
    "foundationpose_time_ms",
    "raw_camera_T_box_json",
    *POSE_MATRIX_FIELDS,
    *POSE_FIELDS,
    "error",
]

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
STATE_COLORS = {
    TRACKING: (0, 255, 0),
    PARTIAL: (0, 255, 255),
    LOST: (0, 0, 255),
    "ERROR": (255, 0, 255),
}


def _cv2():
    return importlib.import_module("cv2")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--camera", choices=("external", "g1"), required=True)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--start", type=int, default=0,
        help="Zero-based frames.csv row at which to start (inclusive).",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Zero-based frames.csv row at which to stop (exclusive); default is all.",
    )
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args(argv)


def read_recording_rows(recording: Path) -> list[dict[str, str]]:
    frames_path = Path(recording) / "frames.csv"
    with frames_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RECORDER_COLUMNS:
            raise RuntimeError(
                "frames.csv schema does not match record_dual_rgbd_session.py; "
                f"expected {RECORDER_COLUMNS}, got {reader.fieldnames}"
            )
        return list(reader)


def select_rows(
    rows: list[dict[str, str]], start: int, end: Optional[int], stride: int
) -> list[dict[str, str]]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if end is not None and end < start:
        raise ValueError("--end must be >= --start")
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    return rows[start:end:stride]


def depth_png_to_metres(raw: np.ndarray) -> np.ndarray:
    if raw is None or np.asarray(raw).ndim != 2:
        raise RuntimeError("depth PNG must be a single-channel image")
    depth = np.asarray(raw, dtype=np.float32) * 0.001
    depth[(depth < 0.001) | (depth > 10.0)] = 0.0
    return depth


def camera_row_values(row: dict[str, str], camera: str) -> tuple[str, str, str]:
    return (
        row[f"{camera}_rgb"],
        row[f"{camera}_depth"],
        row.get(f"{camera}_timestamp", ""),
    )


def load_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read RGB JPEG: {path}")
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path: Path) -> np.ndarray:
    cv2 = _cv2()
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"cannot read depth PNG: {path}")
    if raw.dtype != np.uint16:
        raise RuntimeError(f"depth PNG is not uint16 millimetres: {path} ({raw.dtype})")
    return depth_png_to_metres(raw)


def load_initialization(init_dir: Path):
    cv2 = _cv2()
    rgb_path = init_dir / "rgb" / "000000.png"
    depth_path = init_dir / "depth" / "000000.png"
    mask_path = init_dir / "masks" / "000000.png"
    K_path = init_dir / "cam_K.txt"
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth_raw is None or mask_raw is None or not K_path.exists():
        raise RuntimeError(
            f"incomplete init directory {init_dir}; expected cam_K.txt and "
            "rgb/depth/masks/000000.png"
        )
    if depth_raw.dtype != np.uint16:
        raise RuntimeError(f"initialization depth is not uint16 millimetres: {depth_path}")
    return (
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        depth_png_to_metres(depth_raw),
        mask_raw.astype(bool),
        np.loadtxt(K_path, dtype=float).reshape(3, 3),
    )


def load_foundationpose_runtime(foundationpose_root: Path):
    """Lazily import the complete external FoundationPose installation."""
    root = foundationpose_root.expanduser().resolve()
    if not (root / "estimater.py").is_file():
        raise RuntimeError(f"FoundationPose estimater.py not found under {root}")
    if not (root / "box.obj").is_file():
        raise RuntimeError(f"FoundationPose box mesh not found: {root / 'box.obj'}")

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose offline replay requires CUDA")
    probe = torch.eye(4, device="cuda", dtype=torch.float32)
    _ = torch.linalg.inv(probe)
    torch.cuda.synchronize()
    print("CUDA linalg prewarm: OK")

    sys.path.insert(0, str(root))
    estimater = importlib.import_module("estimater")
    module_path = Path(estimater.__file__).resolve()
    if root != module_path.parent and root not in module_path.parents:
        raise RuntimeError(
            f"loaded estimater from {module_path}, not --foundationpose-root {root}"
        )
    return estimater


def create_estimator(fp, foundationpose_root: Path, output_dir: Path):
    """Construct the same single FoundationPose estimator as the live worker."""
    mesh = fp.trimesh.load(foundationpose_root / "box.obj", force="mesh")
    fp.set_logging_format()
    fp.set_seed(0)
    debug_dir = output_dir / "foundationpose_debug"
    debug_dir.mkdir(exist_ok=True)
    estimator = fp.FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=fp.ScorePredictor(),
        refiner=fp.PoseRefinePredictor(),
        debug_dir=str(debug_dir),
        debug=0,
        glctx=fp.dr.RasterizeCudaContext(),
    )
    return estimator


def flatten_pose(row: dict[str, object], pose: Optional[np.ndarray]) -> None:
    for field in POSE_MATRIX_FIELDS:
        row[field] = ""
    row["raw_camera_T_box_json"] = ""
    if pose is None:
        return
    array = np.asarray(pose)
    row["raw_camera_T_box_json"] = json.dumps(array.tolist(), separators=(",", ":"))
    flat = array.reshape(-1)
    if len(flat) == 16:
        for field, value in zip(POSE_MATRIX_FIELDS, flat):
            row[field] = value


def draw_raw_cube(
    image: np.ndarray, camera_T_box: Optional[np.ndarray], K: np.ndarray, color
) -> bool:
    """Attempt to draw the raw pose without consulting validator state."""
    cv2 = _cv2()
    if camera_T_box is None:
        return False
    pose = np.asarray(camera_T_box, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return False
    points = (pose[:3, :3] @ cube_corners().T).T + pose[:3, 3]
    if np.any(points[:, 2] <= 1e-6):
        return False
    uv = np.column_stack((
        K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
        K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
    ))
    if not np.all(np.isfinite(uv)):
        return False
    uv = np.round(np.clip(uv, -100000, 100000)).astype(np.int32)
    for first, second in BOX_EDGES:
        cv2.line(
            image, tuple(uv[first]), tuple(uv[second]), color, 3, cv2.LINE_AA
        )
    return True


def put_lines(image: np.ndarray, lines: list[str], color) -> None:
    cv2 = _cv2()
    overlay = image.copy()
    height = min(image.shape[0], 24 + 27 * len(lines))
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0, image)
    for index, line in enumerate(lines):
        cv2.putText(
            image, line, (12, 25 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.56, color if index == 1 else (255, 255, 255), 2, cv2.LINE_AA,
        )


def annotate_frame(
    bgr: np.ndarray,
    frame_id: str,
    state: str,
    pose: Optional[np.ndarray],
    K: np.ndarray,
    validity: Optional[PoseValidity],
    fp_time_ms: Optional[float],
    error: str,
) -> np.ndarray:
    image = bgr.copy()
    color = STATE_COLORS.get(state, STATE_COLORS["ERROR"])
    projectable = draw_raw_cube(image, pose, K, color)
    quality = "n/a" if validity is None else f"{validity.quality:.3f}"
    reason = error if validity is None else validity.reason
    timing = "n/a" if fp_time_ms is None else f"{fp_time_ms:.1f} ms"
    lines = [
        f"frame {frame_id} | RAW FoundationPose cube",
        f"{state} | q={quality} | {reason}",
        f"FP track_one: {timing}",
    ]
    if validity is not None:
        lines.extend([
            f"overlap={validity.image_overlap:.3f}  visible_px={validity.projected_visible_area_px:.0f}  depth_coverage={validity.depth_coverage:.3f}",
            f"surface_agreement={validity.surface_agreement:.3f}  agreement_of_depth={validity.agreement_of_valid_depth:.3f}  supported_px={validity.supported_pixels}",
            f"missing={validity.missing_depth_ratio:.3f}  occlusion={validity.occlusion_ratio:.3f}  behind={validity.behind_ratio:.3f}",
        ])
    if pose is not None and not projectable:
        lines.append("raw cube is not projectable into this image")
    put_lines(image, lines, color)
    return image


def inferred_frame_size(K: np.ndarray) -> tuple[int, int]:
    width = max(1, int(round(2.0 * float(K[0, 2]))))
    height = max(1, int(round(2.0 * float(K[1, 2]))))
    return width, height


def output_fps(recording: Path, stride: int) -> float:
    rate = 10.0
    metadata_path = recording / "metadata.json"
    if metadata_path.exists():
        try:
            rate = float(json.loads(metadata_path.read_text()).get("record_hz", rate))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return max(0.1, rate / stride)


def prepare_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    transitions = output_dir / "transitions"
    artifacts = [output_dir / "results.csv", output_dir / "replay.mp4"]
    if any(path.exists() for path in artifacts) or (
        transitions.exists() and any(transitions.iterdir())
    ):
        raise RuntimeError(
            f"output directory already contains replay artifacts; choose a new directory: {output_dir}"
        )
    transitions.mkdir(exist_ok=True)
    return transitions


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), percentage))


def main(argv=None) -> int:
    args = parse_args(argv)
    cv2 = _cv2()
    recording = args.recording.expanduser().resolve()
    fp_root = args.foundationpose_root.expanduser().resolve()
    init_dir = args.init_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    rows = select_rows(
        read_recording_rows(recording), args.start, args.end, args.stride
    )
    if not rows:
        raise RuntimeError("selected frame range is empty")
    K_path = recording / args.camera / "K.txt"
    K = np.loadtxt(K_path, dtype=float).reshape(3, 3)
    transitions_dir = prepare_output(output_dir)

    fp = load_foundationpose_runtime(fp_root)
    estimator = create_estimator(fp, fp_root, output_dir)
    init_rgb, init_depth, init_mask, init_K = load_initialization(init_dir)
    registration_start = time.perf_counter()
    initial_pose = estimator.register(
        K=init_K, rgb=init_rgb, depth=init_depth, ob_mask=init_mask, iteration=5
    )
    registration_ms = (time.perf_counter() - registration_start) * 1000.0
    np.savetxt(output_dir / "initial_pose.txt", np.asarray(initial_pose))
    print(f"FoundationPose registration: {registration_ms:.1f} ms")

    first_rgb_rel, _, _ = camera_row_values(rows[0], args.camera)
    first_bgr = cv2.imread(str(recording / first_rgb_rel), cv2.IMREAD_COLOR)
    if first_bgr is None:
        frame_width, frame_height = inferred_frame_size(K)
    else:
        frame_height, frame_width = first_bgr.shape[:2]
    writer = cv2.VideoWriter(
        str(output_dir / "replay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps(recording, args.stride),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output_dir / 'replay.mp4'}")

    counts = {TRACKING: 0, PARTIAL: 0, LOST: 0}
    processing_times: list[float] = []
    transition_ids: list[str] = []
    previous_state = None
    processed = 0
    error_count = 0

    try:
        with (output_dir / "results.csv").open("w", newline="") as csv_handle:
            csv_writer = csv.DictWriter(csv_handle, fieldnames=RESULT_FIELDS)
            csv_writer.writeheader()
            for selected_index, source_row in enumerate(rows):
                frame_id = source_row["frame"]
                rgb_rel, depth_rel, camera_timestamp = camera_row_values(
                    source_row, args.camera
                )
                result: dict[str, object] = {field: "" for field in RESULT_FIELDS}
                result.update({
                    "recording_frame_id": frame_id,
                    "camera": args.camera,
                    "recorded_camera_timestamp": camera_timestamp,
                    "phase_index": source_row["phase_index"],
                    "phase": source_row["phase"],
                    "rgb_path": rgb_rel,
                    "depth_path": depth_rel,
                })
                bgr = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                raw_pose = None
                validity = None
                fp_time_ms = None
                error = ""
                state = "ERROR"
                try:
                    bgr, rgb = load_rgb(recording / rgb_rel)
                    depth = load_depth(recording / depth_rel)
                    if depth.shape != bgr.shape[:2]:
                        raise RuntimeError(
                            f"RGB/depth shape mismatch: {bgr.shape[:2]} vs {depth.shape}"
                        )
                    fp_start = time.perf_counter()
                    try:
                        # Intentionally run every selected frame synchronously.
                        raw_pose = np.asarray(estimator.track_one(
                            rgb=rgb, depth=depth, K=K, iteration=1
                        ))
                    finally:
                        fp_time_ms = (time.perf_counter() - fp_start) * 1000.0
                        processing_times.append(fp_time_ms)
                    # Same-frame deterministic time: recorded data never ages.
                    try:
                        measurement_time = float(camera_timestamp)
                    except (TypeError, ValueError):
                        measurement_time = float(selected_index)
                    validity = evaluate_camera_pose(
                        raw_pose, measurement_time, measurement_time,
                        depth, K, bgr.shape[:2],
                    )
                    state = validity.state
                    result.update(asdict(validity))
                    counts[state] += 1
                except Exception as exc:  # preserve an explicit row and continue
                    error = f"{type(exc).__name__}: {exc}"
                    result["state"] = "ERROR"
                    result["reason"] = "processing error"
                    result["error"] = error
                    error_count += 1

                result["foundationpose_time_ms"] = (
                    "" if fp_time_ms is None else fp_time_ms
                )
                flatten_pose(result, raw_pose)
                annotated = annotate_frame(
                    bgr, frame_id, state, raw_pose, K, validity, fp_time_ms, error
                )
                if annotated.shape[1::-1] != (frame_width, frame_height):
                    annotated = cv2.resize(
                        annotated, (frame_width, frame_height), interpolation=cv2.INTER_AREA
                    )
                writer.write(annotated)

                if state != previous_state:
                    transition_ids.append(frame_id)
                    safe_frame = int(frame_id)
                    cv2.imwrite(
                        str(transitions_dir / f"frame_{safe_frame:06d}_{state}.jpg"),
                        annotated,
                    )
                    previous_state = state
                csv_writer.writerow(result)
                csv_handle.flush()
                processed += 1
                if processed % 25 == 0 or processed == len(rows):
                    print(f"processed {processed}/{len(rows)} (recording frame {frame_id})")
    finally:
        writer.release()

    print("\nOFFLINE REPLAY SUMMARY")
    print(f"processed frames: {processed}")
    for state in (TRACKING, PARTIAL, LOST):
        percentage = 100.0 * counts[state] / max(processed, 1)
        print(f"{state}: {counts[state]} ({percentage:.2f}%)")
    print(f"ERROR: {error_count} ({100.0 * error_count / max(processed, 1):.2f}%)")
    print("transition frame ids:", ", ".join(transition_ids) if transition_ids else "none")
    if processing_times:
        print(f"FP mean processing time: {statistics.fmean(processing_times):.2f} ms")
        print(f"FP p50 processing time: {percentile(processing_times, 50):.2f} ms")
        print(f"FP p95 processing time: {percentile(processing_times, 95):.2f} ms")
    else:
        print("FP mean/p50/p95 processing time: n/a")
    print("results:", output_dir / "results.csv")
    print("video:", output_dir / "replay.mp4")
    print("transitions:", transitions_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
