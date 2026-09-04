#!/usr/bin/env python3

import base64
import msgpack
import shutil
import cv2
import numpy as np
import pyrealsense2 as rs
import zmq
from pathlib import Path

G1_ENDPOINT = "tcp://192.168.123.164:5555"
EXT_SERIAL = "262322070500"

FP = Path("/home/samsung/Chris/FoundationPose")
EXT_OUT = FP / "g1/data/external_live_init"
G1_OUT = FP / "g1/data/live_init"

K_G1 = np.array([
    [604.9285, 0.0, 329.4290],
    [0.0, 605.6724, 246.9297],
    [0.0, 0.0, 1.0],
], dtype=float)


def polygon_mask(image, title):
    original = image.copy()
    display = image.copy()
    points = []

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            display = original.copy()

            if len(points) > 1:
                cv2.polylines(
                    display,
                    [np.asarray(points, np.int32)],
                    False,
                    (0, 255, 0),
                    2,
                )

            for p in points:
                cv2.circle(display, p, 4, (0, 0, 255), -1)

    cv2.setMouseCallback(title, mouse)

    while True:
        shown = display.copy()
        cv2.putText(
            shown,
            "Click cube | ENTER accept | R reset | ESC cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        cv2.imshow(title, shown)

        k = cv2.waitKey(20) & 0xFF

        if k == 13:
            if len(points) < 3:
                print("Need at least 3 points.")
                continue

            mask = np.zeros(original.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255)
            cv2.destroyWindow(title)
            return mask

        if k in (ord("r"), ord("R")):
            points.clear()
            display = original.copy()

        if k == 27:
            cv2.destroyWindow(title)
            return None


def decode_g1(payload):
    data = msgpack.unpackb(payload, raw=False)["images"]

    result = {}
    for key in ("ego_view", "ego_view_depth"):
        value = data.get(key)
        if value is None:
            continue

        buf = base64.b64decode(value) if isinstance(value, str) else value
        result[key] = cv2.imdecode(
            np.frombuffer(buf, np.uint8),
            cv2.IMREAD_UNCHANGED,
        )

    return result


def save_init(root, bgr, depth_mm, mask, K):
    if root.exists():
        shutil.rmtree(root)

    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "masks").mkdir()

    cv2.imwrite(str(root / "rgb/000000.png"), bgr)
    cv2.imwrite(str(root / "depth/000000.png"), depth_mm)
    cv2.imwrite(str(root / "masks/000000.png"), mask)
    np.savetxt(root / "cam_K.txt", K, fmt="%.10f")


def main():
    # ---------------- G1 ----------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(G1_ENDPOINT)

    # ---------------- External D435i ----------------
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(EXT_SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    sensor = profile.get_device().first_depth_sensor()
    depth_scale = sensor.get_depth_scale()

    color_profile = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
    )
    intr = color_profile.get_intrinsics()

    K_EXT = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ])

    cv2.namedWindow("Dual Cube Initialization", cv2.WINDOW_NORMAL)

    print()
    print("================================================")
    print("DUAL CAMERA CUBE INITIALIZATION")
    print("================================================")
    print("Adjust the cube until BOTH cameras see it well.")
    print("S = freeze both cameras and draw masks")
    print("Q = quit")
    print("================================================")
    print()

    try:
        while True:
            # G1
            g1_data = decode_g1(sock.recv())
            if (
                "ego_view" not in g1_data
                or "ego_view_depth" not in g1_data
            ):
                continue

            # Keep same convention as existing G1 code.
            g1_bgr = cv2.cvtColor(
                g1_data["ego_view"],
                cv2.COLOR_RGB2BGR,
            )
            g1_depth = g1_data["ego_view_depth"]

            # External
            frames = align.process(pipe.wait_for_frames())
            c = frames.get_color_frame()
            d = frames.get_depth_frame()

            if not c or not d:
                continue

            ext_bgr = np.asanyarray(c.get_data()).copy()
            ext_depth_raw = np.asanyarray(d.get_data())

            # Save explicitly in millimetres because FP init loader
            # converts PNG depth using *0.001.
            ext_depth_mm = np.round(
                ext_depth_raw.astype(np.float32)
                * depth_scale
                * 1000.0
            ).astype(np.uint16)

            left = ext_bgr.copy()
            right = g1_bgr.copy()

            cv2.putText(
                left, "EXTERNAL D435i",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                right, "G1 CAMERA",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2,
            )

            view = np.hstack([left, right])

            cv2.putText(
                view,
                "Adjust cube | S = freeze both | Q = quit",
                (15, 465),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2,
            )

            cv2.imshow("Dual Cube Initialization", view)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                return

            if key in (ord("s"), ord("S")):
                frozen_ext = ext_bgr.copy()
                frozen_g1 = g1_bgr.copy()
                frozen_ext_depth = ext_depth_mm.copy()
                frozen_g1_depth = g1_depth.copy()

                print()
                print("Draw EXTERNAL camera cube mask.")
                ext_mask = polygon_mask(
                    frozen_ext,
                    "External Cube Mask",
                )

                if ext_mask is None:
                    print("Cancelled. Returning to live preview.")
                    continue

                print("Draw G1 camera cube mask.")
                g1_mask = polygon_mask(
                    frozen_g1,
                    "G1 Cube Mask",
                )

                if g1_mask is None:
                    print("Cancelled. Returning to live preview.")
                    continue

                save_init(
                    EXT_OUT,
                    frozen_ext,
                    frozen_ext_depth,
                    ext_mask,
                    K_EXT,
                )

                save_init(
                    G1_OUT,
                    frozen_g1,
                    frozen_g1_depth,
                    g1_mask,
                    K_G1,
                )

                print()
                print("==============================================")
                print("DUAL INITIALIZATION SAVED")
                print("==============================================")
                print("External:", EXT_OUT)
                print("G1:      ", G1_OUT)
                print()
                print("DO NOT MOVE THE CUBE YET.")
                print("==============================================")
                return

    finally:
        pipe.stop()
        sock.close(0)
        ctx.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
