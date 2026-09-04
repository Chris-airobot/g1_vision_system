#!/usr/bin/python3
"""G1 hybrid root tracking: ChArUco initialization -> VIVE tracker runtime.

    python g1_hybrid_tracker_visualizer.py                      # formal: needs T_tracker_from_g1_root file
    python g1_hybrid_tracker_visualizer.py --no-tracker-tf      # TEMP MODE: tracker frame == root frame
    python g1_hybrid_tracker_visualizer.py --tracker 95:65:2e:67

Keys: I init/lock alignment, X clear, L reload TF, R clear trail, F d435 frame mode,
      H/1/2/3/+/- view, Q quit.
"""
import argparse
import base64
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # g1_config / base 在任意 cwd 下可导入

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq

from scipy.spatial.transform import Rotation

from viva_vive_ultimate import Dongle, device_id_str
from viva_vive_ultimate.frames import (
    parse_axis_remap,
    quat_to_matrix,
    remap_body_quat,
    to_z_up_position,
    to_z_up_quat,
)
from viva_vive_ultimate.meshio import BODY_AXES_REMAP
from viva_vive_ultimate.handeye import (
    CameraIntrinsics,
    CharucoEstimator,
)

import g1_config as cfg
import g1_common_frame_visualizer_interactive as base


#: 默认 tracker device id，来自 g1_config（环境变量 VIVE_TRACKER_ID）；命令行 --tracker 覆盖
DEFAULT_TRACKER = cfg.TARGET_TRACKER

# Tracker body frame T = "remapped body" of viva_vive_ultimate:
# Z-up world + BODY_AXES_REMAP="Y,-X,Z"
# (X along thickness toward the mount face, Y = 79 mm edge,
# Z = 59 mm edge). This is the same frame as vive_tracker_link
# in model/g1_29dof_with_hand_rev_1_0.urdf and the T used by
# viva_vive_ultimate.handeye, so T_T_B files must use this convention.
R_BODY_REMAP = parse_axis_remap(BODY_AXES_REMAP)

# Convention:
#
# A_T_B maps B coordinates into A.
#
# The TF file must contain a 4x4
#
#     T_T_B = T_tracker_from_g1_root
#
# in the T frame described above. Nominal value from the URDF
# mount: rotation = identity, translation = (0.06, 0, 0.08);
# the real value comes from hand-eye calibration.
DEFAULT_T_T_B_FILE = cfg.TRACKER_TF_FILE

VIVE_STALE_SEC = 0.25
LOWSTATE_STALE_SEC = 0.5
INIT_FRAMES = 30


# ============================================================
# UTILITIES
# ============================================================

def average_T(Ts):

    t = np.mean(
        [T[:3, 3] for T in Ts],
        axis=0,
    )

    M = np.sum(
        [T[:3, :3] for T in Ts],
        axis=0,
    )

    U, _, Vt = np.linalg.svd(M)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return base.T_rt(
        R,
        t,
    )


def pose_error(A, B):

    D = (
        base.invT(A)
        @ B
    )

    mm = (
        np.linalg.norm(
            D[:3, 3]
        )
        * 1000.0
    )

    deg = np.degrees(
        np.linalg.norm(
            Rotation
            .from_matrix(
                D[:3, :3]
            )
            .as_rotvec()
        )
    )

    return (
        float(mm),
        float(deg),
    )


def load_T_T_B(path):

    path = Path(path)

    if not path.exists():
        return None

    T = np.loadtxt(
        path,
        dtype=float,
    )

    if (
        T.shape != (4, 4)
        or not np.allclose(
            T[3],
            [0, 0, 0, 1],
            atol=1e-5,
        )
    ):
        raise RuntimeError(
            f"{path} must be a valid 4x4 transform"
        )

    R = T[:3, :3]

    if (
        not np.allclose(R @ R.T, np.eye(3), atol=1e-4)
        or abs(np.linalg.det(R) - 1.0) > 1e-4
    ):
        raise RuntimeError(
            f"{path}: rotation block is not a proper rotation"
            " (R R^T != I or det != +1)"
        )

    return T


# ============================================================
# VIVE TRACKER 4
# ============================================================

class ViveReader:

    def __init__(self, device_id):

        self.device_id = device_id

        self.lock = threading.Lock()

        self.T = None
        self.status = "WAITING"
        self.hz = 0.0
        self.device = ""

        self.last_good = 0.0
        self.error = ""

        self.stop_flag = False

        threading.Thread(
            target=self._run,
            daemon=True,
        ).start()

    def _run(self):

        count = 0
        tick = time.monotonic()

        try:

            with Dongle() as d:

                for report, pose in d.stream(
                    timeout=0.2
                ):

                    if self.stop_flag:
                        return

                    dev = device_id_str(
                        report.mac
                    )

                    # Selected tracker ONLY
                    if dev != self.device_id:
                        continue

                    now = time.monotonic()

                    count += 1

                    if now - tick >= 1.0:

                        hz = (
                            count
                            / (now - tick)
                        )

                        count = 0
                        tick = now

                    else:

                        with self.lock:
                            hz = self.hz

                    with self.lock:

                        self.device = dev
                        self.status = (
                            pose.status_name
                        )
                        self.hz = hz

                    # Do not update pose using stale
                    # RECENTLY_LOST packets.
                    if pose.status_name != "OK":
                        continue

                    p = np.asarray(
                        to_z_up_position(
                            pose.pos
                        ),
                        dtype=float,
                    )

                    # Z-up world, then remap body axes (Y,-X,Z)
                    # so T matches the URDF vive_tracker_link frame.
                    q = remap_body_quat(
                        to_z_up_quat(
                            np.asarray(
                                pose.rot,
                                dtype=float,
                            )
                        ),
                        R_BODY_REMAP,
                    )

                    T = base.T_rt(
                        quat_to_matrix(q),
                        p,
                    )

                    with self.lock:

                        self.T = T

                        self.last_good = now

        except Exception as e:

            with self.lock:

                self.status = "ERROR"

                self.error = repr(e)

    def get(self):

        with self.lock:

            return (
                None
                if self.T is None
                else self.T.copy(),

                self.status,
                self.hz,
                self.device,
                self.last_good,
                self.error,
            )

    def stop(self):

        self.stop_flag = True


# ============================================================
# MAIN
# ============================================================

def parse_args(argv=None):

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tracker",
        default=DEFAULT_TRACKER,
        help=f"tracker device id (last 4 MAC bytes), default {DEFAULT_TRACKER}",
    )
    ap.add_argument(
        "--tracker-tf",
        type=Path,
        default=DEFAULT_T_T_B_FILE,
        help=f"4x4 T_tracker_from_g1_root text file, default {DEFAULT_T_T_B_FILE}",
    )
    ap.add_argument(
        "--no-tracker-tf",
        action="store_true",
        help="TEMP MODE: treat tracker frame as root frame (T_T_B = identity)",
    )
    return ap.parse_args(argv)


def main(argv=None):

    args = parse_args(argv)
    temp_mode = args.no_tracker_tf

    print("=" * 72)
    print(
        "G1 HYBRID: ChArUco initialization"
        " -> VIVE tracker runtime"
    )
    print(
        "tracker:",
        args.tracker,
    )
    print(
        "external D435i:",
        base.EXTERNAL_SERIAL,
    )
    print(
        "URDF:",
        base.URDF,
    )
    print("=" * 72)

    if temp_mode:

        # TEMPORARY TEST MODE: tracker frame == root frame.
        # K_T_V = K_T_B0 @ inv(V_T_T0) absorbs the fixed offset;
        # runtime K_T_B = K_T_V @ V_T_T. Translation direction is
        # right because the verified mount rotation is identity;
        # rotating the robot still shows the lever-arm error.
        T_T_B = np.eye(4)

        print()
        print("TEMP MODE: tracker pose is treated as root pose")
        print("No tracker->root transform is used (T_T_B = identity).")
        print()

    else:

        T_T_B = load_T_T_B(args.tracker_tf)

    if T_T_B is None:

        print()
        print(
            "Missing tracker->root transform:"
        )
        print(
            args.tracker_tf
        )

        print(
            "Program can run, but VIVE alignment"
            " cannot be locked yet."
        )

        print(
            "After creating the file,"
            " press L to reload it."
        )
        print()

    else:

        print()
        print(
            "Loaded T_tracker_from_root:"
        )
        print(T_T_B)
        print()

    # --------------------------------------------------------
    # G1 state
    # --------------------------------------------------------

    low = base.LowStateReader()

    # --------------------------------------------------------
    # VIVE
    # --------------------------------------------------------

    vive = ViveReader(args.tracker)

    # --------------------------------------------------------
    # G1 onboard camera
    # --------------------------------------------------------

    ctx = zmq.Context()

    sock = ctx.socket(
        zmq.SUB
    )

    sock.setsockopt_string(
        zmq.SUBSCRIBE,
        "",
    )

    sock.setsockopt(
        zmq.CONFLATE,
        1,
    )

    sock.setsockopt(
        zmq.RCVTIMEO,
        2000,
    )

    sock.connect(
        base.G1_ENDPOINT
    )

    # --------------------------------------------------------
    # External D435i
    # --------------------------------------------------------

    pipe = rs.pipeline()

    cfg = rs.config()

    cfg.enable_device(
        base.EXTERNAL_SERIAL
    )

    cfg.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        30,
    )

    profile = pipe.start(
        cfg
    )

    intr = (
        profile
        .get_stream(
            rs.stream.color
        )
        .as_video_stream_profile()
        .get_intrinsics()
    )

    K_EXT = np.array([
        [
            intr.fx,
            0,
            intr.ppx,
        ],
        [
            0,
            intr.fy,
            intr.ppy,
        ],
        [
            0,
            0,
            1,
        ],
    ], dtype=float)

    D_EXT = np.asarray(
        intr.coeffs,
        dtype=float,
    )

    # --------------------------------------------------------
    # ChArUco
    # --------------------------------------------------------

    g1_intr = CameraIntrinsics(
        width=640,
        height=480,
        camera_matrix=(
            base.K_G1.tolist()
        ),
        distortion=(
            base.D_G1.tolist()
        ),
        source="g1_rgb",
    )

    ext_intr = CameraIntrinsics(
        width=640,
        height=480,
        camera_matrix=(
            K_EXT.tolist()
        ),
        distortion=(
            D_EXT.tolist()
        ),
        source="external_d435i",
    )

    estimator = CharucoEstimator(
        base.BOARD,
        min_corners=base.MIN_CORNERS,
    )

    frame_mode = "ros_optical"

    # ========================================================
    # HYBRID STATE
    # ========================================================

    # K_T_V gets locked during initialization.
    K_T_V = None

    # Fixed external-camera pose in ChArUco frame.
    K_T_E_fixed = None
    E_T_K_fixed = None

    init_candidates = []
    init_EK = []

    init_collecting = False

    # Automatically initialize once Vision + VIVE have both
    # remained valid for 1 second.
    auto_ready_since = None

    last_B = None
    last_C = None

    trajectory = []

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    cv2.namedWindow(
        "Hybrid Cameras",
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        "Interactive 3D World",
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        "Interactive 3D World",
        1000,
        700,
    )

    viewer = (
        base.Interactive3DViewer(
            width=1000,
            height=700,
        )
    )

    cv2.setMouseCallback(
        "Interactive 3D World",
        viewer.mouse_callback,
    )

    try:

        while True:

            now = time.monotonic()

            # =================================================
            # G1 RGB
            # =================================================

            try:

                payload = sock.recv()

            except zmq.Again:

                continue

            data = msgpack.unpackb(
                payload,
                raw=False,
            )

            v = (
                data
                .get("images", {})
                .get("ego_view")
            )

            if v is None:
                continue

            buf = (
                base64.b64decode(v)
                if isinstance(v, str)
                else v
            )

            g1 = cv2.imdecode(
                np.frombuffer(
                    buf,
                    np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )

            if g1 is None:
                continue

            g1 = cv2.cvtColor(
                g1,
                cv2.COLOR_RGB2BGR,
            )

            # =================================================
            # EXTERNAL RGB
            # =================================================

            try:

                frames = pipe.wait_for_frames(
                    2000
                )

            except RuntimeError as exc:

                # External D435i hiccup: keep the locked
                # alignment alive instead of crashing.
                print(
                    "External D435i timeout:",
                    exc,
                )

                continue

            cf = (
                frames
                .get_color_frame()
            )

            if not cf:
                continue

            ext = np.asanyarray(
                cf.get_data()
            ).copy()

            # =================================================
            # CHARUCO
            # =================================================

            dg = estimator.detect(
                g1,
                g1_intr,
            )

            de = estimator.detect(
                ext,
                ext_intr,
            )

            g1_vis = estimator.draw(
                g1,
                dg,
                g1_intr,
            )

            ext_vis = estimator.draw(
                ext,
                de,
                ext_intr,
            )

            # =================================================
            # G1 LOWSTATE
            # =================================================

            q, mode_machine, low_t = (
                low.get()
            )

            low_ok = (
                q is not None
                and
                now - low_t
                < LOWSTATE_STALE_SEC
            )

            # Vision root needs only the G1 camera + FK:
            # K_T_B = inv(C_T_K) @ inv(B_T_C). The external
            # camera cancels out of K_T_B; it is used only to
            # project results into its own image.
            vision_ok = (
                low_ok
                and dg is not None
                and dg.C_T_K is not None
            )

            ext_ok = (
                de is not None
                and de.C_T_K is not None
            )

            # =================================================
            # VIVE
            # =================================================

            (
                V_T_T,
                vive_status,
                vive_hz,
                vive_dev,
                vive_t,
                vive_err,
            ) = vive.get()

            vive_ok = (
                V_T_T is not None
                and vive_status == "OK"
                and
                now - vive_t
                < VIVE_STALE_SEC
                and
                vive_dev
                == args.tracker
            )

            # =================================================
            # ROBOT CAMERA FK
            # =================================================

            B_T_C = None

            if low_ok:

                B_T_D = (
                    base
                    .pelvis_T_d435(q)
                )

                D_T_C = (
                    base.D_T_C_ROS
                    if
                    frame_mode
                    == "ros_optical"
                    else
                    base.D_T_C_IDENTITY
                )

                B_T_C = (
                    B_T_D
                    @ D_T_C
                )

            # =================================================
            # INDEPENDENT VISION ROOT
            # =================================================

            K_T_B_vision = None
            K_T_C_vision = None

            K_T_E_live = None
            E_T_K_live = None

            E_T_B_vision = None

            if vision_ok:

                C_T_K = np.asarray(
                    dg.C_T_K,
                    dtype=float,
                )

                K_T_C_vision = base.invT(C_T_K)

                K_T_B_vision = (
                    K_T_C_vision
                    @ base.invT(B_T_C)
                )

            if ext_ok:

                E_T_K_live = np.asarray(
                    de.C_T_K,
                    dtype=float,
                )

                K_T_E_live = base.invT(E_T_K_live)

                if K_T_B_vision is not None:

                    # Only for drawing in the external image.
                    E_T_B_vision = (
                        E_T_K_live
                        @ K_T_B_vision
                    )

            # =================================================
            # AUTO INITIALIZATION
            #
            # No keyboard focus is required.
            # Wait until Vision + VIVE are continuously valid
            # for 1 second, then automatically collect 30 frames.
            # =================================================

            if (
                K_T_V is None
                and not init_collecting
                and T_T_B is not None
                and vision_ok
                and vive_ok
            ):

                if auto_ready_since is None:
                    auto_ready_since = now

                elif now - auto_ready_since >= 1.0:

                    init_candidates.clear()
                    init_EK.clear()
                    init_collecting = True
                    auto_ready_since = None

                    print(
                        f"AUTO INIT: collecting {INIT_FRAMES} frames. "
                        "KEEP ROBOT STILL..."
                    )

            else:

                if not init_collecting:
                    auto_ready_since = None


            # =================================================
            # INITIALIZATION
            #
            # K_T_V =
            #
            # K_T_B
            # @ inv(T_T_B)
            # @ inv(V_T_T)
            #
            # Average 30 frames while robot is stationary.
            # =================================================

            if (
                init_collecting
                and T_T_B is not None
                and vision_ok
                and vive_ok
            ):

                candidate = (
                    K_T_B_vision
                    @ base.invT(T_T_B)
                    @ base.invT(V_T_T)
                )

                init_candidates.append(
                    candidate
                )

                if ext_ok:

                    init_EK.append(
                        E_T_K_live.copy()
                    )

                if (
                    len(init_candidates)
                    >= INIT_FRAMES
                ):

                    K_T_V = average_T(
                        init_candidates
                    )

                    # Average ONE transform and invert it so the
                    # pair is exactly mutually inverse (the mean
                    # of inverses is not the inverse of the mean).
                    if init_EK:

                        E_T_K_fixed = average_T(
                            init_EK
                        )

                        K_T_E_fixed = base.invT(
                            E_T_K_fixed
                        )

                    else:

                        E_T_K_fixed = None
                        K_T_E_fixed = None

                        print(
                            "External camera never saw the board"
                            " during init: no external overlay."
                        )

                    init_candidates.clear()
                    init_EK.clear()

                    init_collecting = False

                    trajectory.clear()

                    print()
                    print("=" * 72)
                    print(
                        "ALIGNMENT LOCKED"
                    )
                    print(
                        "K_T_V:"
                    )
                    print(K_T_V)
                    print("=" * 72)
                    print()

            # =================================================
            # TRACKER ROOT
            #
            # K_T_B =
            # K_T_V @ V_T_T @ T_T_B
            # =================================================

            K_T_B_tracker = None
            K_T_C_tracker = None

            if (
                K_T_V is not None
                and T_T_B is not None
                and vive_ok
            ):

                K_T_B_tracker = (
                    K_T_V
                    @ V_T_T
                    @ T_T_B
                )

                if B_T_C is not None:

                    K_T_C_tracker = (
                        K_T_B_tracker
                        @ B_T_C
                    )

                last_B = (
                    K_T_B_tracker.copy()
                )

                if (
                    K_T_C_tracker
                    is not None
                ):

                    last_C = (
                        K_T_C_tracker.copy()
                    )

                p = (
                    K_T_B_tracker[
                        :3,
                        3,
                    ].copy()
                )

                if (
                    not trajectory
                    or
                    np.linalg.norm(
                        p
                        - trajectory[-1]
                    )
                    > 0.005
                ):

                    trajectory.append(
                        p
                    )

                    trajectory = (
                        trajectory[-300:]
                    )

            # =================================================
            # PRIMARY VISUALIZATION SOURCE
            # =================================================

            if K_T_B_tracker is not None:

                primary_B = (
                    K_T_B_tracker
                )

                primary_C = (
                    K_T_C_tracker
                )

            elif K_T_V is not None:

                # Momentary VIVE loss:
                # show last pose, but status becomes non-OK.
                primary_B = last_B
                primary_C = last_C

            else:

                # Before tracker initialization:
                # use vision.
                primary_B = (
                    K_T_B_vision
                )

                primary_C = (
                    K_T_C_vision
                )

            external_pose = (
                K_T_E_fixed
                if
                K_T_E_fixed
                is not None
                else
                K_T_E_live
            )

            # =================================================
            # INTERACTIVE WORLD
            # =================================================

            world = viewer.render(
                primary_B,
                primary_C,
                external_pose,
                trajectory,
                (
                    vive_ok
                    if K_T_V is not None
                    else vision_ok
                ),
            )

            # Show BOTH whenever vision available.
            if K_T_B_vision is not None:

                viewer.draw_frame(
                    world,
                    K_T_B_vision,
                    "ROOT [VISION]",
                    0.14,
                )

            if K_T_B_tracker is not None:

                viewer.draw_frame(
                    world,
                    K_T_B_tracker,
                    "ROOT [TRACKER]",
                    0.20,
                )

            # =================================================
            # TRACKER VS VISION ERROR
            # =================================================

            err = None

            if (
                K_T_B_tracker
                is not None
                and
                K_T_B_vision
                is not None
            ):

                err = pose_error(
                    K_T_B_vision,
                    K_T_B_tracker,
                )

            # =================================================
            # HUD
            # =================================================

            y = 82

            base.put(
                world,
                (
                    f"VIVE 4: "
                    f"{vive_status}  "
                    f"{vive_hz:.1f} Hz  "
                    f"id={vive_dev or '--'}"
                ),
                15,
                y,
                (
                    (0, 255, 0)
                    if vive_ok
                    else (0, 140, 255)
                ),
                0.48,
            )

            y += 25

            base.put(
                world,
                (
                    "VISION (G1 cam+FK): "
                    + (
                        "VISIBLE"
                        if vision_ok
                        else "NOT VISIBLE"
                    )
                    + " | EXT cam: "
                    + (
                        "VISIBLE"
                        if ext_ok
                        else "NOT VISIBLE"
                    )
                ),
                15,
                y,
                (
                    (0, 255, 0)
                    if vision_ok
                    else (0, 140, 255)
                ),
                0.48,
            )

            y += 25

            if K_T_V is not None:

                txt = (
                    "ALIGNMENT: LOCKED | "
                    "ROOT SOURCE: VIVE"
                )

                col = (
                    0,
                    255,
                    0,
                )

            elif init_collecting:

                txt = (
                    f"ALIGNMENT: "
                    f"{len(init_candidates)}"
                    f"/{INIT_FRAMES}"
                    " - KEEP ROBOT STILL"
                )

                col = (
                    0,
                    220,
                    255,
                )

            else:

                txt = (
                    "ALIGNMENT: NOT LOCKED | "
                    "press I when VISION+VIVE valid"
                )

                col = (
                    0,
                    140,
                    255,
                )

            base.put(
                world,
                txt,
                15,
                y,
                col,
                0.48,
            )

            y += 25

            if err is not None:

                base.put(
                    world,
                    (
                        "TRACKER vs VISION: "
                        f"{err[0]:.1f} mm | "
                        f"{err[1]:.2f} deg"
                    ),
                    15,
                    y,
                    (
                        255,
                        255,
                        255,
                    ),
                    0.50,
                )

                y += 25

            if temp_mode:

                base.put(
                    world,
                    "TEMP MODE: TRACKER FRAME = ROOT FRAME",
                    15,
                    y,
                    (0, 220, 255),
                    0.46,
                )

            elif T_T_B is None:

                base.put(
                    world,
                    (
                        "NO T_tracker_from_root: "
                        "create file then press L"
                    ),
                    15,
                    y,
                    (
                        0,
                        0,
                        255,
                    ),
                    0.48,
                )

            cv2.imshow(
                "Interactive 3D World",
                world,
            )

            # =================================================
            # PROJECT TRACKER RESULT BACK INTO
            # EXTERNAL CAMERA IMAGE
            # =================================================

            if (
                E_T_K_fixed is not None
                and
                K_T_B_tracker
                is not None
            ):

                E_T_B_tracker = (
                    E_T_K_fixed
                    @ K_T_B_tracker
                )

                base.draw_pose_axes(
                    ext_vis,
                    E_T_B_tracker,
                    K_EXT,
                    D_EXT,
                    "ROOT TRACKER",
                    0.15,
                )

                if (
                    K_T_C_tracker
                    is not None
                ):

                    base.draw_pose_axes(
                        ext_vis,
                        (
                            E_T_K_fixed
                            @ K_T_C_tracker
                        ),
                        K_EXT,
                        D_EXT,
                        "G1 CAM TRACKER",
                        0.10,
                    )

            if E_T_B_vision is not None:

                base.draw_pose_axes(
                    ext_vis,
                    E_T_B_vision,
                    K_EXT,
                    D_EXT,
                    "ROOT VISION",
                    0.12,
                )

            base.put(
                ext_vis,
                "EXTERNAL D435i",
                15,
                28,
                (
                    255,
                    255,
                    255,
                ),
                0.60,
            )

            base.put(
                g1_vis,
                "G1 ONBOARD RGB",
                15,
                28,
                (
                    255,
                    255,
                    255,
                ),
                0.60,
            )

            if vive_err:

                base.put(
                    ext_vis,
                    (
                        "VIVE ERROR: "
                        + vive_err[:65]
                    ),
                    15,
                    460,
                    (
                        0,
                        0,
                        255,
                    ),
                    0.42,
                )

            cv2.imshow(
                "Hybrid Cameras",
                np.hstack([
                    ext_vis,
                    g1_vis,
                ]),
            )

            # =================================================
            # CONTROLS
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
            ):

                break

            # -----------------------------------------------
            # Initialize VIVE alignment
            # -----------------------------------------------

            elif key in (
                ord("i"),
                ord("I"),
            ):

                if T_T_B is None:

                    print(
                        "Cannot initialize:"
                        " missing mentor"
                        " T_tracker_from_root."
                    )

                elif not vision_ok:

                    print(
                        "Cannot initialize:"
                        " board must be visible in the G1"
                        " camera and lowstate must be fresh."
                    )

                elif not vive_ok:

                    print(
                        "Cannot initialize:"
                        " tracker 4 must be OK."
                    )

                else:

                    init_candidates.clear()
                    init_EK.clear()

                    init_collecting = True

                    print(
                        f"Collecting "
                        f"{INIT_FRAMES} frames."
                        " KEEP ROBOT STILL..."
                    )

            # -----------------------------------------------
            # Clear tracker/world alignment
            # -----------------------------------------------

            elif key in (
                ord("x"),
                ord("X"),
            ):

                K_T_V = None

                K_T_E_fixed = None
                E_T_K_fixed = None

                init_collecting = False

                init_candidates.clear()
                init_EK.clear()

                last_B = None
                last_C = None

                trajectory.clear()

                print(
                    "Alignment cleared."
                )

            # -----------------------------------------------
            # Reload mentor transform
            # -----------------------------------------------

            elif key in (
                ord("l"),
                ord("L"),
            ):

                if temp_mode:

                    print(
                        "TEMP MODE: no tracker->root TF is being used."
                    )

                    continue

                try:

                    T_T_B = (
                        load_T_T_B(args.tracker_tf)
                    )

                    if T_T_B is None:

                        print(
                            "Still missing:",
                            args.tracker_tf,
                        )

                    else:

                        print(
                            "Loaded "
                            "T_tracker_from_root:"
                        )

                        print(
                            T_T_B
                        )

                except Exception as e:

                    print(
                        "Reload failed:",
                        e,
                    )

            elif key in (
                ord("r"),
                ord("R"),
            ):

                trajectory.clear()

            elif key in (
                ord("f"),
                ord("F"),
            ):

                frame_mode = (
                    "identity"
                    if
                    frame_mode
                    == "ros_optical"
                    else
                    "ros_optical"
                )

                print(
                    "D435 frame mode:",
                    frame_mode,
                )

                # B_T_C changed: a locked K_T_V (or a collection
                # in progress) is no longer consistent with it.
                if K_T_V is not None or init_collecting:

                    K_T_V = None
                    K_T_E_fixed = None
                    E_T_K_fixed = None
                    init_collecting = False
                    init_candidates.clear()
                    init_EK.clear()
                    last_B = None
                    last_C = None
                    trajectory.clear()

                    print(
                        "Alignment cleared (frame mode changed):"
                        " press I to re-initialize."
                    )

            elif key in (
                ord("h"),
                ord("H"),
            ):

                viewer.reset_view()

            elif key == ord("1"):

                viewer.top_view()

            elif key == ord("2"):

                viewer.side_view()

            elif key == ord("3"):

                viewer.perspective_view()

            elif key in (
                ord("+"),
                ord("="),
            ):

                viewer.zoom(
                    0.85
                )

            elif key in (
                ord("-"),
                ord("_"),
            ):

                viewer.zoom(
                    1.18
                )

    finally:

        vive.stop()

        pipe.stop()

        sock.close(0)

        ctx.term()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
