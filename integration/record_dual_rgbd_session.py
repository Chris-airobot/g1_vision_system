#!/usr/bin/env python3

import base64
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq

ROOT = Path("/home/samsung/Chris/g1_box_tracking")
VIVE_SCRIPTS = ROOT / "vive/g1_tracker_system/scripts"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VIVE_SCRIPTS))

import g1_common_frame_visualizer_interactive as base
import g1_hybrid_tracker_visualizer as hybrid


G1_ENDPOINT = "tcp://192.168.123.164:5555"
EXT_SERIAL = "262322070500"
RECORD_HZ = 10.0

OUT_ROOT = Path.home() / "Chris/g1_box_recordings"

PHASES = [
    "00_baseline_both_visible_cube_still",
    "01_both_visible_move_and_rotate_cube",
    "02_cube_still_turn_g1_left_right",
    "03_external_partial_occlusion",
    "04_external_fully_blocked_g1_only",
    "05_external_returns_cube_unchanged",
    "06_g1_partial_loss_edge_of_fov",
    "07_g1_fully_away_external_only",
    "08_g1_returns_cube_unchanged",
    "09_both_lose_cube_cube_still",
    "10_external_returns_first_g1_still_lost",
    "11_both_return_same_cube_pose",
    "12_both_lost_move_cube_while_hidden",
    "13_external_reacquires_cube_new_pose",
    "14_g1_joins_cube_new_pose",
    "15_external_blocked_move_cube_g1_only",
    "16_g1_away_move_cube_external_only",
    "17_dynamic_partial_occlusions_while_moving",
    "18_actual_carry_like_motion",
    "19_final_baseline_both_visible_cube_still",
]


class G1Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self.rgb = None
        self.depth = None
        self.seq = 0
        self.ts = 0.0
        self.error = ""
        self.stop_flag = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 250)
        sock.connect(G1_ENDPOINT)

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

                        buf = (
                            base64.b64decode(value)
                            if isinstance(value, str)
                            else value
                        )

                        decoded[key] = cv2.imdecode(
                            np.frombuffer(buf, np.uint8),
                            cv2.IMREAD_UNCHANGED,
                        )

                    rgb = decoded.get("ego_view")
                    depth_raw = decoded.get("ego_view_depth")

                    if rgb is None or depth_raw is None:
                        continue

                    depth_m = depth_raw.astype(np.float32) * 0.001
                    depth_m[
                        (depth_m < 0.001) | (depth_m > 10.0)
                    ] = 0.0

                    with self.lock:
                        self.rgb = rgb
                        self.depth = depth_m
                        self.seq += 1
                        self.ts = time.monotonic()
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
                self.seq,
                self.ts,
                self.error,
            )

    def stop(self):
        self.stop_flag = True


class ExternalCamera:
    def __init__(self):
        self.lock = threading.Lock()
        self.bgr = None
        self.depth = None
        self.K = None
        self.D = None
        self.seq = 0
        self.ts = 0.0
        self.error = ""
        self.stop_flag = False

        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        pipe = rs.pipeline()

        try:
            cfg = rs.config()
            cfg.enable_device(EXT_SERIAL)
            cfg.enable_stream(
                rs.stream.color,
                640, 480,
                rs.format.bgr8,
                30,
            )
            cfg.enable_stream(
                rs.stream.depth,
                640, 480,
                rs.format.z16,
                30,
            )

            profile = pipe.start(cfg)

            sensor = profile.get_device().first_depth_sensor()
            scale = sensor.get_depth_scale()

            align = rs.align(rs.stream.color)

            cp = (
                profile
                .get_stream(rs.stream.color)
                .as_video_stream_profile()
            )

            intr = cp.get_intrinsics()

            K = np.array([
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=float)

            D = np.asarray(intr.coeffs, dtype=float)

            with self.lock:
                self.K = K
                self.D = D

            while not self.stop_flag:
                try:
                    frames = align.process(
                        pipe.wait_for_frames(250)
                    )

                    color = frames.get_color_frame()
                    depth = frames.get_depth_frame()

                    if not color or not depth:
                        continue

                    bgr = np.asanyarray(color.get_data()).copy()

                    depth_m = (
                        np.asanyarray(depth.get_data())
                        .astype(np.float32)
                        * scale
                    )

                    depth_m[
                        (depth_m < 0.001) |
                        (depth_m > 10.0)
                    ] = 0.0

                    with self.lock:
                        self.bgr = bgr
                        self.depth = depth_m
                        self.seq += 1
                        self.ts = time.monotonic()
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
                self.seq,
                self.ts,
                self.error,
            )

    def stop(self):
        self.stop_flag = True


def depth_to_png(depth_m):
    return np.clip(
        np.round(depth_m * 1000.0),
        0,
        65535,
    ).astype(np.uint16)


def json_matrix(value):
    if value is None:
        return ""

    return json.dumps(
        np.asarray(value, dtype=float).tolist(),
        separators=(",", ":"),
    )


def save_calibration_snapshot(out):
    cal_out = out / "calibration"
    cal_out.mkdir(parents=True, exist_ok=True)

    paths = [
        ROOT / "vive/g1_tracker_system/calibration/T_external_from_vive_world.txt",
        ROOT / "vive/g1_tracker_system/calibration/T_tracker_from_g1_root.txt",
        ROOT / "vive/g1_tracker_system/calibration/dual_camera_charuco_results/T_external_from_g1_camera.txt",
        ROOT / "vive/g1_tracker_system/calibration/dual_camera_charuco_results/T_g1_camera_from_external.txt",
    ]

    for path in paths:
        if path.exists():
            shutil.copy2(path, cal_out / path.name)


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"dual_rgbd_{stamp}"

    for camera in ("external", "g1"):
        (out / camera / "rgb").mkdir(
            parents=True,
            exist_ok=True,
        )
        (out / camera / "depth").mkdir(
            parents=True,
            exist_ok=True,
        )

    save_calibration_snapshot(out)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"

    g1 = G1Camera()
    ext = ExternalCamera()

    try:
        low = base.LowStateReader()
    except Exception as exc:
        print("LowState unavailable:", exc)
        low = None

    vive = hybrid.ViveReader("0d:e1:7b:f0")

    print()
    print("Waiting for both RGB-D cameras...")

    while True:
        g1_rgb, g1_depth, *_ = g1.get()
        ext_bgr, ext_depth, K_EXT, D_EXT, *_ = ext.get()

        if (
            g1_rgb is not None
            and g1_depth is not None
            and ext_bgr is not None
            and ext_depth is not None
            and K_EXT is not None
        ):
            break

        time.sleep(0.05)

    np.savetxt(
        out / "g1/K.txt",
        np.asarray(base.K_G1, dtype=float),
        fmt="%.10f",
    )

    np.savetxt(
        out / "external/K.txt",
        K_EXT,
        fmt="%.10f",
    )

    np.savetxt(
        out / "external/D.txt",
        D_EXT,
        fmt="%.10f",
    )

    metadata = {
        "created": datetime.now().isoformat(),
        "git_commit": git_commit,
        "record_hz": RECORD_HZ,
        "g1_endpoint": G1_ENDPOINT,
        "external_serial": EXT_SERIAL,
        "g1_interface": "enx98fc84e54eda",
        "vive_tracker": "0d:e1:7b:f0",
        "box_dimensions_m": [0.30, 0.30, 0.30],
        "depth_unit": "saved PNG values are millimetres",
        "rgb_format": "JPEG quality 92",
        "timestamp_note":
            "camera timestamps are host monotonic receive times; pair_dt_ms records difference",
        "phases": PHASES,
    }

    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    frame_csv = open(
        out / "frames.csv",
        "w",
        newline="",
    )

    event_csv = open(
        out / "events.csv",
        "w",
        newline="",
    )

    frame_writer = csv.writer(frame_csv)
    event_writer = csv.writer(event_csv)

    frame_writer.writerow([
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
    ])

    event_writer.writerow([
        "wall_time",
        "monotonic",
        "event",
        "phase_index",
        "phase",
    ])

    phase = 0
    recording = False
    frame_id = 0
    next_save = time.monotonic()

    last_g1_seq = -1
    last_ext_seq = -1

    cv2.namedWindow(
        "RAW DUAL CAMERA RECORDER",
        cv2.WINDOW_NORMAL,
    )

    print()
    print("==============================================")
    print("RAW DUAL RGB-D RECORDER")
    print("==============================================")
    print("R = START recording")
    print("N = NEXT test phase")
    print("P = PREVIOUS phase")
    print("Q = STOP + save")
    print()
    print("Initial phase:")
    print(PHASES[phase])
    print("==============================================")

    try:
        while True:
            now = time.monotonic()

            (
                g1_rgb,
                g1_depth,
                g1_seq,
                g1_ts,
                g1_err,
            ) = g1.get()

            (
                ext_bgr,
                ext_depth,
                K_EXT,
                D_EXT,
                ext_seq,
                ext_ts,
                ext_err,
            ) = ext.get()

            if g1_rgb is not None:
                g1_bgr = cv2.cvtColor(
                    g1_rgb,
                    cv2.COLOR_RGB2BGR,
                )
            else:
                g1_bgr = np.zeros(
                    (480, 640, 3),
                    dtype=np.uint8,
                )

            if ext_bgr is None:
                ext_view = np.zeros(
                    (480, 640, 3),
                    dtype=np.uint8,
                )
            else:
                ext_view = ext_bgr.copy()

            g1_view = g1_bgr.copy()

            cv2.putText(
                ext_view,
                "EXTERNAL D435i",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                g1_view,
                "G1 CAMERA",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            view = np.hstack(
                [ext_view, g1_view]
            )

            state = (
                "RECORDING"
                if recording
                else "READY - PRESS R"
            )

            cv2.putText(
                view,
                state,
                (15, 445),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255)
                if recording
                else (0, 255, 255),
                2,
            )

            phase_text = (
                f"{phase:02d}/{len(PHASES)-1:02d} "
                f"{PHASES[phase]}"
            )

            cv2.putText(
                view,
                phase_text,
                (15, 472),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            cv2.imshow(
                "RAW DUAL CAMERA RECORDER",
                view,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("r"), ord("R")):
                if not recording:
                    recording = True
                    next_save = now

                    event_writer.writerow([
                        datetime.now().isoformat(),
                        now,
                        "START_RECORDING",
                        phase,
                        PHASES[phase],
                    ])
                    event_csv.flush()

                    print()
                    print("RECORDING STARTED")
                    print("PHASE:", PHASES[phase])

            if key in (ord("n"), ord("N")):
                if phase < len(PHASES) - 1:
                    phase += 1

                    event_writer.writerow([
                        datetime.now().isoformat(),
                        now,
                        "PHASE",
                        phase,
                        PHASES[phase],
                    ])
                    event_csv.flush()

                    print()
                    print("================================")
                    print("NEXT PHASE:")
                    print(PHASES[phase])
                    print("================================")

            if key in (ord("p"), ord("P")):
                if phase > 0:
                    phase -= 1

                    event_writer.writerow([
                        datetime.now().isoformat(),
                        now,
                        "PHASE",
                        phase,
                        PHASES[phase],
                    ])
                    event_csv.flush()

                    print()
                    print("PHASE:", PHASES[phase])

            if not recording or now < next_save:
                time.sleep(0.002)
                continue

            next_save += 1.0 / RECORD_HZ

            if (
                g1_rgb is None
                or g1_depth is None
                or ext_bgr is None
                or ext_depth is None
            ):
                continue

            # Avoid writing the exact same frame pair repeatedly.
            if (
                g1_seq == last_g1_seq
                and ext_seq == last_ext_seq
            ):
                continue

            last_g1_seq = g1_seq
            last_ext_seq = ext_seq

            name = f"{frame_id:06d}"

            g1_rgb_rel = (
                f"g1/rgb/{name}.jpg"
            )
            g1_depth_rel = (
                f"g1/depth/{name}.png"
            )

            ext_rgb_rel = (
                f"external/rgb/{name}.jpg"
            )
            ext_depth_rel = (
                f"external/depth/{name}.png"
            )

            cv2.imwrite(
                str(out / g1_rgb_rel),
                g1_bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    92,
                ],
            )

            cv2.imwrite(
                str(out / ext_rgb_rel),
                ext_bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    92,
                ],
            )

            cv2.imwrite(
                str(out / g1_depth_rel),
                depth_to_png(g1_depth),
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    1,
                ],
            )

            cv2.imwrite(
                str(out / ext_depth_rel),
                depth_to_png(ext_depth),
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    1,
                ],
            )

            if low is not None:
                try:
                    q, mode_machine, low_ts = low.get()
                except Exception:
                    q = None
                    mode_machine = None
                    low_ts = 0.0
            else:
                q = None
                mode_machine = None
                low_ts = 0.0

            B_T_C = None

            if q is not None:
                try:
                    B_T_C = (
                        base.pelvis_T_d435(q)
                        @ base.D_T_C_ROS
                    )
                except Exception:
                    B_T_C = None

            try:
                (
                    V_T_T,
                    vive_status,
                    vive_hz,
                    vive_dev,
                    vive_ts,
                    vive_error,
                ) = vive.get()
            except Exception:
                V_T_T = None
                vive_status = "ERROR"
                vive_ts = 0.0

            pair_dt_ms = (
                abs(g1_ts - ext_ts) * 1000.0
            )

            frame_writer.writerow([
                frame_id,
                phase,
                PHASES[phase],
                now,
                datetime.now().isoformat(),
                g1_seq,
                g1_ts,
                ext_seq,
                ext_ts,
                f"{pair_dt_ms:.3f}",
                g1_rgb_rel,
                g1_depth_rel,
                ext_rgb_rel,
                ext_depth_rel,
                mode_machine,
                (
                    f"{(now-low_ts)*1000:.3f}"
                    if low_ts
                    else ""
                ),
                (
                    json.dumps(
                        np.asarray(q).tolist(),
                        separators=(",", ":"),
                    )
                    if q is not None
                    else ""
                ),
                json_matrix(B_T_C),
                vive_status,
                (
                    f"{(now-vive_ts)*1000:.3f}"
                    if vive_ts
                    else ""
                ),
                json_matrix(V_T_T),
            ])

            frame_csv.flush()

            if frame_id % 20 == 0:
                print(
                    f"saved {frame_id:06d} | "
                    f"phase={phase:02d} | "
                    f"pair dt={pair_dt_ms:.1f} ms"
                )

            frame_id += 1

    finally:
        event_writer.writerow([
            datetime.now().isoformat(),
            time.monotonic(),
            "STOP_RECORDING",
            phase,
            PHASES[phase],
        ])

        frame_csv.close()
        event_csv.close()

        g1.stop()
        ext.stop()

        try:
            vive.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

    print()
    print("==============================================")
    print("RECORDING COMPLETE")
    print("==============================================")
    print("Frames:", frame_id)
    print("Saved to:")
    print(out)
    print()
    print("DO NOT PUT THIS RECORDING IN GIT.")
    print("==============================================")


if __name__ == "__main__":
    main()
