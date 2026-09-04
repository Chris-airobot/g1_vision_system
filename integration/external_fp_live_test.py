#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path("/home/samsung/Chris/g1_box_tracking")
FP_ROOT = Path("/home/samsung/Chris/FoundationPose")
INIT_DIR = FP_ROOT / "g1/data/external_live_init"
OUT_DIR = ROOT / "integration/outputs/external_live_test"

sys.path.insert(0, str(ROOT))

from integration.foundationpose_worker import FoundationPoseWorker


SERIAL = "262322070500"
BOX = np.array([0.30, 0.30, 0.30], dtype=float)

EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


def corners():
    hx, hy, hz = BOX / 2.0
    return np.array([
        [-hx,-hy,-hz],
        [ hx,-hy,-hz],
        [ hx, hy,-hz],
        [-hx, hy,-hz],
        [-hx,-hy, hz],
        [ hx,-hy, hz],
        [ hx, hy, hz],
        [-hx, hy, hz],
    ], dtype=float)


def draw_box(img, T, K):
    if T is None:
        return img

    pts = corners()
    pts = (T[:3,:3] @ pts.T).T + T[:3,3]

    if np.any(pts[:,2] <= 0.01):
        return img

    uv = np.column_stack([
        K[0,0] * pts[:,0] / pts[:,2] + K[0,2],
        K[1,1] * pts[:,1] / pts[:,2] + K[1,2],
    ])
    uv = np.round(uv).astype(int)

    for a,b in EDGES:
        cv2.line(
            img,
            tuple(uv[a]),
            tuple(uv[b]),
            (0,255,0),
            3,
            cv2.LINE_AA,
        )

    # object frame axes
    origin = T[:3,3]
    axes = np.array([
        origin,
        origin + T[:3,0] * 0.15,
        origin + T[:3,1] * 0.15,
        origin + T[:3,2] * 0.15,
    ])

    uv_axes = np.column_stack([
        K[0,0] * axes[:,0] / axes[:,2] + K[0,2],
        K[1,1] * axes[:,1] / axes[:,2] + K[1,2],
    ])
    uv_axes = np.round(uv_axes).astype(int)

    o = tuple(uv_axes[0])
    cv2.line(img, o, tuple(uv_axes[1]), (0,0,255), 3)
    cv2.line(img, o, tuple(uv_axes[2]), (0,255,0), 3)
    cv2.line(img, o, tuple(uv_axes[3]), (255,0,0), 3)

    return img


def main():
    worker = FoundationPoseWorker(
        name="external",
        foundationpose_root=FP_ROOT,
        mesh_path=FP_ROOT / "box.obj",
        init_dir=INIT_DIR,
        output_dir=OUT_DIR,
        track_iterations=1,
        register_iterations=5,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = cp.get_intrinsics()

    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ], dtype=float)

    print("External D435i live FoundationPose")
    print("Serial:", SERIAL)
    print("Q = quit")
    print("Green = 30x30x30 cm FoundationPose cube")

    last_pose = None
    last_submit = 0
    frames = 0
    t0 = time.time()

    try:
        while True:
            fs = pipeline.wait_for_frames()
            fs = align.process(fs)

            cf = fs.get_color_frame()
            df = fs.get_depth_frame()
            if not cf or not df:
                continue

            bgr = np.asanyarray(cf.get_data()).copy()
            depth_raw = np.asanyarray(df.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale
            depth_m[(depth_m < 0.001) | (depth_m > 10.0)] = 0

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # latest-frame-only worker
            worker.submit(rgb, depth_m, K)

            pose, pose_time, status, error = worker.get()
            if pose is not None:
                last_pose = pose
                np.savetxt(OUT_DIR / "latest_E_T_box.txt", pose)

            vis = bgr.copy()
            vis = draw_box(vis, last_pose, K)

            frames += 1
            fps = frames / max(time.time() - t0, 1e-6)

            cv2.putText(
                vis,
                f"FP: {status} | camera {fps:.1f} FPS",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2,
            )

            if error:
                cv2.putText(
                    vis,
                    error[:80],
                    (15, 455),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,0,255),
                    1,
                )

            cv2.imshow("External D435i FoundationPose LIVE", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    finally:
        worker.stop()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
