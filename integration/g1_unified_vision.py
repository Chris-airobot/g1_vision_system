#!/usr/bin/env python3
"""Unified VIVE/root and independent dual-FoundationPose visualization.

This deliberately leaves the legacy scripts unchanged. It imports their
validated FK, frame remapping, tracker reader, and viewer.

Keys: L reload tracker mount TF, R clear trail, H/1/2/3/+/- view, Q quit.
"""

from __future__ import annotations

import argparse
import base64
import threading
import time
import sys
from pathlib import Path

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq

ROOT = Path(__file__).resolve().parents[1]
VIVE_SCRIPTS = ROOT / "vive" / "g1_tracker_system" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VIVE_SCRIPTS))

import g1_common_frame_visualizer_interactive as base  # noqa: E402
import g1_hybrid_tracker_visualizer as hybrid  # noqa: E402
from integration.foundationpose_worker import FoundationPoseWorker  # noqa: E402
from integration.transforms import (  # noqa: E402
    BOX_DIMS_M,
    LOST,
    box_disagreement,
    compose_world_box_poses,
    evaluate_camera_pose,
    fuse_world_poses,
    invert_transform,
    save_latest_transforms,
    tracker_root_and_camera,
    validate_transform,
)


CAMERA_STALE_SEC = 1.0
SAVE_INTERVAL_SEC = 1.0
FP_VALID_HZ = 8.0
FP_INVALID_HZ = 2.0
RESEED_INTERVAL_SEC = 1.0
VALIDATION_HZ = 20.0
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def depth_metres(raw: np.ndarray, scale: float) -> np.ndarray:
    depth = np.asarray(raw, dtype=np.float32) * float(scale)
    depth[(depth < 0.001) | (depth > 10.0)] = 0.0
    return depth


class G1CameraReader:
    """Non-blocking latest RGB-D reader for the onboard ZMQ stream."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.lock = threading.Lock()
        self.rgb = self.depth = None
        self.sequence = 0
        self.timestamp = 0.0
        self.error = ""
        self.stop_flag = False
        threading.Thread(target=self._run, daemon=True, name="g1-camera").start()

    def _run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 250)
        sock.connect(self.endpoint)
        try:
            while not self.stop_flag:
                try:
                    payload = sock.recv()
                    data = msgpack.unpackb(payload, raw=False).get("images", {})
                    decoded = {}
                    for key in ("ego_view", "ego_view_depth"):
                        value = data.get(key)
                        if value is None:
                            continue
                        buf = base64.b64decode(value) if isinstance(value, str) else value
                        decoded[key] = cv2.imdecode(
                            np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED
                        )
                    rgb = decoded.get("ego_view")
                    raw_depth = decoded.get("ego_view_depth")
                    if rgb is None or raw_depth is None:
                        continue
                    with self.lock:
                        self.rgb = rgb
                        self.depth = depth_metres(raw_depth, 0.001)
                        self.sequence += 1
                        self.timestamp = time.monotonic()
                        self.error = ""
                except zmq.Again:
                    continue
                except Exception as exc:
                    with self.lock:
                        self.error = repr(exc)
        finally:
            sock.close(0)
            ctx.term()

    def get(self):
        with self.lock:
            return (
                None if self.rgb is None else self.rgb.copy(),
                None if self.depth is None else self.depth.copy(),
                self.sequence, self.timestamp, self.error,
            )

    def stop(self):
        self.stop_flag = True


class ExternalCameraReader:
    """Own the D435i pipeline and expose aligned RGB-D without blocking UI."""

    def __init__(self, serial: str):
        self.serial = serial
        self.lock = threading.Lock()
        self.bgr = self.depth = self.K = self.D = None
        self.sequence = 0
        self.timestamp = 0.0
        self.error = ""
        self.stop_flag = False
        self.pipeline = None
        threading.Thread(target=self._run, daemon=True, name="external-camera").start()

    def _run(self):
        pipe = rs.pipeline()
        self.pipeline = pipe
        try:
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipe.start(config)
            sensor = profile.get_device().first_depth_sensor()
            scale = sensor.get_depth_scale()
            align = rs.align(rs.stream.color)
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            K = np.array(
                [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                dtype=float,
            )
            D = np.asarray(intr.coeffs, dtype=float)
            with self.lock:
                self.K, self.D = K, D
            while not self.stop_flag:
                try:
                    frames = align.process(pipe.wait_for_frames(250))
                    color, depth = frames.get_color_frame(), frames.get_depth_frame()
                    if not color or not depth:
                        continue
                    with self.lock:
                        self.bgr = np.asanyarray(color.get_data()).copy()
                        self.depth = depth_metres(np.asanyarray(depth.get_data()), scale)
                        self.sequence += 1
                        self.timestamp = time.monotonic()
                        self.error = ""
                except RuntimeError as exc:
                    with self.lock:
                        self.error = repr(exc)
        except Exception as exc:
            with self.lock:
                self.error = repr(exc)
        finally:
            try:
                pipe.stop()
            except Exception:
                pass

    def get(self):
        with self.lock:
            return (
                None if self.bgr is None else self.bgr.copy(),
                None if self.depth is None else self.depth.copy(),
                None if self.K is None else self.K.copy(),
                None if self.D is None else self.D.copy(),
                self.sequence, self.timestamp, self.error,
            )

    def stop(self):
        self.stop_flag = True


def box_corners() -> np.ndarray:
    hx, hy, hz = BOX_DIMS_M * 0.5
    return np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])


def draw_box_world(viewer, image, E_T_box, label, color):
    if E_T_box is None:
        return
    points = (E_T_box[:3, :3] @ box_corners().T).T + E_T_box[:3, 3]
    for a, b in BOX_EDGES:
        viewer.draw_line_e(image, points[a], points[b], color, 3)
    viewer.draw_frame(image, E_T_box, label, 0.11)


def draw_box_camera(image, camera_T_box, K, label, color):
    if image is None or camera_T_box is None or K is None:
        return
    points = (camera_T_box[:3, :3] @ box_corners().T).T + camera_T_box[:3, 3]
    if np.any(points[:, 2] <= 1e-6):
        return
    uv = np.column_stack((
        K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2],
        K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2],
    ))
    uv = np.round(uv).astype(np.int32)
    for a, b in BOX_EDGES:
        cv2.line(image, tuple(uv[a]), tuple(uv[b]), color, 2, cv2.LINE_AA)
    cv2.putText(image, label, tuple(uv[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def placeholder(text: str) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    base.put(image, text, 20, 40, (180, 180, 180), 0.65)
    return image


class ExternalWorldViewer(base.Interactive3DViewer):
    """Use the legacy projection/view controls without drawing a ChArUco board."""

    def draw_board(self, img, E_T_K):
        return None

    def draw_frame(self, img, T, label, length=0.15):
        if label == "CHARUCO / FIXED WORLD K":
            label = "FIXED WORLD / EXTERNAL D435i E"
        elif label == "EXTERNAL D435i E":
            return None
        return super().draw_frame(img, T, label, length)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default=hybrid.DEFAULT_TRACKER)
    parser.add_argument("--tracker-tf", type=Path, default=hybrid.DEFAULT_T_T_B_FILE)
    parser.add_argument("--no-tracker-tf", action="store_true")
    parser.add_argument("--g1-endpoint", default=base.G1_ENDPOINT)
    parser.add_argument("--external-serial", default=base.EXTERNAL_SERIAL)
    parser.add_argument(
        "--foundationpose-root", type=Path, required=True,
        help="Complete FoundationPose runtime root containing estimater.py, weights, and box.obj.",
    )
    parser.add_argument(
        "--g1-init-dir", type=Path, required=True,
        help="Existing G1-camera FoundationPose initialization dataset directory.",
    )
    parser.add_argument(
        "--external-init-dir", type=Path, required=True,
        help="Existing external-camera FoundationPose initialization dataset directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "integration" / "outputs" / "latest"
    )
    parser.add_argument(
        "--external-vive-tf",
        type=Path,
        default=ROOT / "vive/g1_tracker_system/calibration/T_external_from_vive_world.txt",
        help="Saved E_T_V (T_external_from_vive_world); required at runtime.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fp_root = args.foundationpose_root.expanduser().resolve()
    mesh = fp_root / "box.obj"
    try:
        T_T_B = np.eye(4) if args.no_tracker_tf else hybrid.load_T_T_B(args.tracker_tf)
    except (OSError, ValueError, RuntimeError) as exc:
        T_T_B = None
        print(f"G1 world path disabled: cannot load tracker mount transform: {exc}")
    if T_T_B is None:
        print(f"G1 world path disabled: tracker mount transform missing: {args.tracker_tf}")
    try:
        E_T_V = validate_transform(
            np.loadtxt(args.external_vive_tf, dtype=float).reshape(4, 4), "E_T_V"
        )
    except (OSError, ValueError) as exc:
        E_T_V = None
        print(f"G1 world path disabled: cannot load E_T_V: {exc}")

    print("BOARD-FREE RUNTIME: fixed world is external D435i optical frame E")
    if E_T_V is not None:
        print("Loaded E_T_V from", args.external_vive_tf)

    # These are the exact readers/FK and VIVE remapping used by the hybrid runtime.
    try:
        low = base.LowStateReader()
    except Exception as exc:
        low = None
        print(f"G1 world path disabled: cannot start lowstate: {exc}")
    vive = hybrid.ViveReader(args.tracker)
    g1_camera = G1CameraReader(args.g1_endpoint)
    ext_camera = ExternalCameraReader(args.external_serial)
    workers = {
        "external": FoundationPoseWorker(
            "external", fp_root, mesh, args.external_init_dir,
            args.output_dir / "foundationpose_external",
        ),
        "g1": FoundationPoseWorker(
            "g1", fp_root, mesh, args.g1_init_dir,
            args.output_dir / "foundationpose_g1",
        ),
    }
    last_g1_fp_seq = last_ext_fp_seq = -1
    last_g1_submit = last_ext_submit = 0.0
    ext_valid_for_rate = g1_valid_for_rate = False
    last_ext_reseed = last_g1_reseed = -float("inf")
    trajectory = []
    last_save = 0.0
    last_save_signature = None
    last_validation = -float("inf")
    E_T_box = C_T_box = None
    ext_pose_time = g1_pose_time = 0.0
    ext_fp_status = g1_fp_status = "STARTING"
    ext_fp_error = g1_fp_error = ""
    ext_pose_depth = g1_pose_depth = None
    ext_pose_K = g1_pose_K = None
    ext_pose_shape = g1_pose_shape = None
    ext_check = evaluate_camera_pose(None, 0.0, 0.0, None, None, None)
    g1_check = evaluate_camera_pose(None, 0.0, 0.0, None, None, None)

    cv2.namedWindow("Unified Cameras", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Unified 3D World", cv2.WINDOW_NORMAL)
    viewer = ExternalWorldViewer(width=1100, height=760)
    cv2.setMouseCallback("Unified 3D World", viewer.mouse_callback)

    try:
        while True:
            now = time.monotonic()
            g1_rgb, g1_depth, g1_seq, g1_time, g1_error = g1_camera.get()
            ext_bgr, ext_depth, K_EXT, D_EXT, ext_seq, ext_time, ext_error = ext_camera.get()
            g1_fresh = g1_rgb is not None and now - g1_time < CAMERA_STALE_SEC
            ext_fresh = ext_bgr is not None and now - ext_time < CAMERA_STALE_SEC
            g1_bgr = (
                cv2.cvtColor(g1_rgb, cv2.COLOR_RGB2BGR)
                if g1_rgb is not None else placeholder("G1 CAMERA: NO INPUT")
            )

            g1_interval = 1.0 / (FP_VALID_HZ if g1_valid_for_rate else FP_INVALID_HZ)
            ext_interval = 1.0 / (FP_VALID_HZ if ext_valid_for_rate else FP_INVALID_HZ)
            if (
                g1_fresh and g1_depth is not None and g1_seq != last_g1_fp_seq
                and now - last_g1_submit >= g1_interval
            ):
                workers["g1"].submit(g1_rgb, g1_depth, base.K_G1, g1_time)
                last_g1_fp_seq = g1_seq
                last_g1_submit = now
            if (
                ext_fresh and ext_depth is not None and K_EXT is not None
                and ext_seq != last_ext_fp_seq and now - last_ext_submit >= ext_interval
            ):
                workers["external"].submit(
                    cv2.cvtColor(ext_bgr, cv2.COLOR_BGR2RGB), ext_depth, K_EXT, ext_time
                )
                last_ext_fp_seq = ext_seq
                last_ext_submit = now

            q, mode_machine, low_time = low.get() if low is not None else (None, None, 0.0)
            low_ok = q is not None and now - low_time < hybrid.LOWSTATE_STALE_SEC
            V_T_T, vive_status, vive_hz, vive_dev, vive_time, vive_error = vive.get()
            vive_ok = (
                V_T_T is not None and vive_status == "OK"
                and now - vive_time < hybrid.VIVE_STALE_SEC and vive_dev == args.tracker
            )

            B_T_C = None
            if low_ok:
                # Preserve the legacy ROS optical convention and exact rev1.0 URDF FK.
                B_T_C = base.pelvis_T_d435(q) @ base.D_T_C_ROS

            E_T_B = E_T_C = E_T_T = None
            if vive_ok and E_T_V is not None and T_T_B is not None:
                E_T_B, E_T_C, E_T_T = tracker_root_and_camera(
                    E_T_V, V_T_T, T_T_B, B_T_C
                )
                position = E_T_B[:3, 3].copy()
                if not trajectory or np.linalg.norm(position - trajectory[-1]) > 0.005:
                    trajectory.append(position); trajectory = trajectory[-300:]

            # Evaluate the pose with the exact depth/K frame consumed by that
            # worker. Cap the CPU ray-cast rate so rendering remains responsive.
            if now - last_validation >= 1.0 / VALIDATION_HZ:
                (
                    E_T_box, ext_pose_time, ext_fp_status, ext_fp_error,
                    ext_pose_depth, ext_pose_K, ext_pose_shape,
                ) = workers["external"].get_validation_inputs()
                (
                    C_T_box, g1_pose_time, g1_fp_status, g1_fp_error,
                    g1_pose_depth, g1_pose_K, g1_pose_shape,
                ) = workers["g1"].get_validation_inputs()
                ext_check = evaluate_camera_pose(
                    E_T_box, ext_pose_time, now,
                    ext_pose_depth, ext_pose_K, ext_pose_shape,
                )
                g1_check = evaluate_camera_pose(
                    C_T_box, g1_pose_time, now,
                    g1_pose_depth, g1_pose_K, g1_pose_shape,
                )
                last_validation = now
            E_T_box_ext, E_T_box_g1, composed_E_T_C = compose_world_box_poses(
                E_T_V, V_T_T if vive_ok else None, T_T_B, B_T_C,
                E_T_box, C_T_box,
            )
            # The external contribution is camera-local world E and never
            # depends on VIVE, lowstate, FK, or G1 visibility.
            ext_valid = ext_check.valid and E_T_box_ext is not None
            g1_valid = g1_check.valid and E_T_box_g1 is not None
            ext_state = ext_check.state if ext_valid else LOST
            g1_state = g1_check.state if g1_valid else LOST
            ext_valid_for_rate, g1_valid_for_rate = ext_valid, g1_valid

            fusion = fuse_world_poses(
                E_T_box_ext, ext_valid, ext_check.quality,
                E_T_box_g1, g1_valid, g1_check.quality,
            )
            disagreement = (
                box_disagreement(E_T_box_ext, E_T_box_g1)
                if ext_valid and g1_valid else None
            )

            # Reseed only from the other valid camera, at a bounded rate. The
            # worker converts original-mesh pose to FoundationPose pose_last.
            if (
                fusion.source == "G1" and ext_fresh
                and now - last_ext_reseed >= RESEED_INTERVAL_SEC
            ):
                workers["external"].request_reseed(fusion.pose)
                last_ext_reseed = now
            if (
                fusion.source == "EXTERNAL" and g1_fresh and composed_E_T_C is not None
                and now - last_g1_reseed >= RESEED_INTERVAL_SEC
            ):
                C_T_box_seed = invert_transform(composed_E_T_C, "E_T_C") @ fusion.pose
                workers["g1"].request_reseed(C_T_box_seed)
                last_g1_reseed = now

            world = viewer.render(
                E_T_B, E_T_C, np.eye(4), trajectory, vive_ok,
            )
            if E_T_T is not None:
                viewer.draw_frame(world, E_T_T, "VIVE TRACKER T", 0.13)
            if ext_valid:
                draw_box_world(viewer, world, E_T_box_ext, "BOX [EXTERNAL]", (0, 215, 255))
            if g1_valid:
                draw_box_world(viewer, world, E_T_box_g1, "BOX [G1 CAMERA]", (255, 80, 255))
            if fusion.valid:
                draw_box_world(viewer, world, fusion.pose, "BOX [FUSED]", (80, 255, 80))
            metric_lines = [
                f"EXT: {ext_state} q={ext_check.quality:.2f} ({ext_check.reason})",
                f"G1: {g1_state} q={g1_check.quality:.2f} "
                f"({'world chain unavailable' if g1_check.valid and not g1_valid else g1_check.reason})",
                f"FUSED SOURCE: {fusion.source}",
                f"BOX: {'TRACKED' if fusion.valid else 'LOST'}",
            ]
            if disagreement is not None:
                metric_lines += [
                    f"translation disagreement: {disagreement.translation_mm:.1f} mm",
                    f"symmetry-aware disagreement: {disagreement.rotation_symmetry_deg:.1f} deg",
                ]
            for index, text in enumerate(metric_lines):
                base.put(world, text, 15, 625 + 25 * index, (255, 255, 255), 0.46)
            cv2.imshow("Unified 3D World", world)

            ext_vis = ext_bgr.copy() if ext_bgr is not None else placeholder("EXTERNAL D435i: NO INPUT")
            g1_vis = g1_bgr.copy()
            if ext_valid:
                draw_box_camera(
                    ext_vis, E_T_box, K_EXT, "BOX [EXTERNAL]", (0, 215, 255)
                )
            if g1_valid:
                draw_box_camera(
                    g1_vis, C_T_box, base.K_G1, "BOX [G1 CAMERA]", (255, 80, 255)
                )
            if ext_error:
                base.put(ext_vis, f"CAMERA: {ext_error[:72]}", 15, 435, (0, 0, 255), 0.38)
            if g1_error:
                base.put(g1_vis, f"CAMERA: {g1_error[:72]}", 15, 435, (0, 0, 255), 0.38)
            base.put(ext_vis, f"EXT: {ext_state} q={ext_check.quality:.2f}", 15, 460, (0, 215, 255), 0.48)
            base.put(g1_vis, f"G1: {g1_state} q={g1_check.quality:.2f}", 15, 460, (255, 80, 255), 0.48)
            if ext_fp_error:
                base.put(ext_vis, ext_fp_error[:72], 190, 460, (0, 0, 255), 0.34)
            if g1_fp_error:
                base.put(g1_vis, g1_fp_error[:72], 190, 460, (0, 0, 255), 0.34)
            cv2.imshow("Unified Cameras", np.hstack((ext_vis, g1_vis)))

            save_signature = (ext_valid, g1_valid, fusion.valid, fusion.source)
            if (
                now - last_save >= SAVE_INTERVAL_SEC
                or save_signature != last_save_signature
            ):
                save_latest_transforms(
                    args.output_dir / "transforms",
                    E_T_box=E_T_box if ext_valid else None,
                    C_T_box=C_T_box if g1_valid else None,
                    E_T_box_ext=E_T_box_ext if ext_valid else None,
                    E_T_box_g1=E_T_box_g1 if g1_valid else None,
                    E_T_box_fused=fusion.pose if fusion.valid else None,
                )
                last_save = now
                last_save_signature = save_signature

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("l"), ord("L")) and not args.no_tracker_tf:
                try:
                    T_T_B = hybrid.load_T_T_B(args.tracker_tf)
                    print("Reloaded tracker transform:", args.tracker_tf)
                except Exception as exc:
                    print("Tracker transform reload failed:", exc)
            elif key in (ord("r"), ord("R")):
                trajectory.clear()
            elif key in (ord("h"), ord("H")):
                viewer.reset_view()
            elif key == ord("1"):
                viewer.top_view()
            elif key == ord("2"):
                viewer.side_view()
            elif key == ord("3"):
                viewer.perspective_view()
            elif key in (ord("+"), ord("=")):
                viewer.zoom(0.85)
            elif key in (ord("-"), ord("_")):
                viewer.zoom(1.18)
            time.sleep(0.002)
    finally:
        for worker in workers.values():
            worker.stop()
        g1_camera.stop(); ext_camera.stop(); vive.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
