#!/usr/bin/env python3
"""Capture aligned RealSense RGB-D data for the FoundationPose box test."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from automatic_object_mask import (
    AutomaticMaskConfig,
    AutomaticMaskError,
    generate_object_mask,
    save_mask_debug,
)


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        roi = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1") from exc
    if len(roi) != 4:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1")
    return roi


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_dir / "demo_data" / "box_test",
        help="New output directory (default: demo_data/box_test).",
    )
    parser.add_argument(
        "--serial",
        help="RealSense serial number. Required when multiple cameras are connected.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List connected RealSense cameras and exit.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
        help="Frames to save after selection; 0 records until Q (default: 300).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames discarded before preview (default: 30).",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("auto", "manual"),
        default="auto",
        help="First-frame mask source (default: auto).",
    )
    parser.add_argument(
        "--workspace-roi",
        type=parse_roi,
        help="Tabletop image crop as x0,y0,x1,y1 (default: full frame).",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--table-ransac-threshold-m", type=float, default=0.008)
    parser.add_argument("--min-object-height-m", type=float, default=0.012)
    parser.add_argument("--max-object-height-m", type=float, default=0.30)
    parser.add_argument("--cluster-tolerance-m", type=float, default=0.010)
    parser.add_argument("--min-cluster-pixels", type=int, default=200)
    parser.add_argument("--mask-morphology-kernel-px", type=int, default=5)
    parser.add_argument(
        "--manual-fallback",
        action="store_true",
        help="Open the manual mask selector if automatic generation fails.",
    )
    return parser.parse_args()


def get_devices(rs) -> list[tuple[str, str]]:
    devices = []
    for device in rs.context().query_devices():
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        devices.append((name, serial))
    return devices


def choose_serial(devices: list[tuple[str, str]], requested: str | None) -> str:
    if not devices:
        raise RuntimeError("No RealSense camera was found.")

    serials = {serial for _, serial in devices}
    if requested is not None:
        if requested not in serials:
            available = ", ".join(sorted(serials))
            raise RuntimeError(
                f"RealSense serial {requested!r} was not found. Available: {available}"
            )
        return requested

    if len(devices) > 1:
        lines = [f"  {name}: {serial}" for name, serial in devices]
        raise RuntimeError(
            "Multiple RealSense cameras are connected. Choose one with --serial:\n"
            + "\n".join(lines)
        )
    return devices[0][1]


def validate_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}\n"
            "Choose a new directory so an earlier capture is not overwritten."
        )


def prepare_output_dir(output_dir: Path) -> None:
    (output_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)


def save_frame(
    output_dir: Path,
    index: int,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    depth_scale: float,
    mask: np.ndarray | None = None,
) -> None:
    name = f"{index:06d}.png"
    depth_mm = np.rint(depth_raw.astype(np.float32) * depth_scale * 1000.0)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    rgb_ok = cv2.imwrite(str(output_dir / "rgb" / name), color_bgr)
    depth_ok = cv2.imwrite(str(output_dir / "depth" / name), depth_mm)
    if not rgb_ok or not depth_ok:
        raise RuntimeError(f"Failed to write RGB-D frame {name}.")

    if mask is not None:
        mask_ok = cv2.imwrite(str(output_dir / "masks" / name), mask)
        if not mask_ok:
            raise RuntimeError(f"Failed to write first-frame mask {name}.")


def select_box_mask(color_bgr: np.ndarray) -> np.ndarray | None:
    window = "Draw a tight rectangle around the box, then press Enter"
    x, y, width, height = map(
        int,
        cv2.selectROI(window, color_bgr, showCrosshair=True, fromCenter=False),
    )
    cv2.destroyWindow(window)
    if width <= 0 or height <= 0:
        return None

    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    mask[y : y + height, x : x + width] = 255
    return mask


def automatic_mask_config(args: argparse.Namespace) -> AutomaticMaskConfig:
    return AutomaticMaskConfig(
        workspace_roi=args.workspace_roi,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        table_ransac_threshold_m=args.table_ransac_threshold_m,
        min_object_height_m=args.min_object_height_m,
        max_object_height_m=args.max_object_height_m,
        cluster_tolerance_m=args.cluster_tolerance_m,
        min_cluster_pixels=args.min_cluster_pixels,
        morphology_kernel_px=args.mask_morphology_kernel_px,
    )


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("Width, height, and FPS must be positive.")
    if args.num_frames < 0 or args.warmup_frames < 0:
        raise SystemExit("Frame counts cannot be negative.")

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this environment.\n"
            "If company policy permits it, run: python -m pip install pyrealsense2"
        ) from exc

    devices = get_devices(rs)
    if args.list_devices:
        if not devices:
            print("No RealSense cameras found.")
            return 1
        for name, serial in devices:
            print(f"{name}: {serial}")
        return 0

    try:
        serial = choose_serial(devices, args.serial)
        validate_output_dir(args.output_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    align = rs.align(rs.stream.color)

    started = False
    recording = False
    frame_index = 0
    try:
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        depth_scale = device.first_depth_sensor().get_depth_scale()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        print(f"Camera: {device_name}")
        print(f"Serial: {serial}")
        print(f"Depth scale: {depth_scale} metres/unit")
        print(f"First-frame mask mode: {args.mask_mode}")
        print("Press S to create the first mask and start recording; press Q to stop.")

        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_raw = np.asanyarray(depth_frame.get_data())
            color_bgr = np.asanyarray(color_frame.get_data())

            if recording:
                save_frame(
                    args.output_dir,
                    frame_index,
                    color_bgr,
                    depth_raw,
                    depth_scale,
                )
                frame_index += 1
                if args.num_frames and frame_index >= args.num_frames:
                    print(f"Saved {frame_index} frames to {args.output_dir}")
                    break

            preview = color_bgr.copy()
            message = (
                f"Recording frame {frame_index} - Q: stop"
                if recording
                else f"S: {args.mask_mode} mask and record - Q: quit"
            )
            cv2.putText(
                preview,
                message,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("RealSense box capture", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                if recording:
                    print(f"Saved {frame_index} frames to {args.output_dir}")
                else:
                    print("Capture cancelled before recording started.")
                break

            if not recording and key in (ord("s"), ord("S")):
                automatic_succeeded = False
                if args.mask_mode == "auto":
                    depth_m = depth_raw.astype(np.float32) * depth_scale
                    try:
                        mask = generate_object_mask(
                            depth_m,
                            camera_matrix,
                            automatic_mask_config(args),
                        )
                        automatic_succeeded = True
                        print(f"Automatic mask contains {np.count_nonzero(mask)} pixels.")
                    except AutomaticMaskError as exc:
                        print(f"Automatic mask failed: {exc}")
                        if not args.manual_fallback:
                            print("Press S to retry, or use --manual-fallback/--mask-mode manual.")
                            continue
                        print("Falling back to manual mask selection.")
                        mask = select_box_mask(color_bgr)
                else:
                    mask = select_box_mask(color_bgr)
                if mask is None:
                    print("No box rectangle selected; returning to preview.")
                    continue

                prepare_output_dir(args.output_dir)
                if automatic_succeeded:
                    debug_path = args.output_dir / "automatic_mask_debug.png"
                    try:
                        save_mask_debug(debug_path, color_bgr, mask)
                    except AutomaticMaskError as exc:
                        raise RuntimeError(str(exc)) from exc
                    print(f"Saved automatic-mask debug image to {debug_path}")
                np.savetxt(args.output_dir / "cam_K.txt", camera_matrix, fmt="%.10f")
                save_frame(
                    args.output_dir,
                    frame_index,
                    color_bgr,
                    depth_raw,
                    depth_scale,
                    mask=mask,
                )
                frame_index += 1
                recording = True
                print(f"Recording to {args.output_dir}")

                if args.num_frames and frame_index >= args.num_frames:
                    print(f"Saved {frame_index} frames to {args.output_dir}")
                    break

    except KeyboardInterrupt:
        print(f"\nStopped. Saved {frame_index} frames to {args.output_dir}")
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
