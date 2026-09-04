#!/usr/bin/python3

import base64
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 让 g1_config 在任意 cwd 下可导入

import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2

try:
    cv2.setLogLevel(0)  # SILENT
except Exception:
    pass
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq

from scipy.spatial.transform import Rotation

from viva_vive_ultimate.frames import parse_axis_remap
from viva_vive_ultimate.handeye import (
    BoardConfig,
    CameraIntrinsics,
    CharucoEstimator,
)

# ============================================================
# CONFIG
# ============================================================

import g1_config as cfg

# 所有常量集中在 g1_config.py，这里保留同名别名供其它脚本 base.XXX 引用
G1_IFACE = cfg.G1_IFACE
G1_ENDPOINT = cfg.G1_ENDPOINT
EXTERNAL_SERIAL = cfg.EXTERNAL_SERIAL
URDF = str(cfg.URDF)

if cfg.UNITREE_SDK:
    sys.path.insert(0, cfg.UNITREE_SDK)

try:
    from unitree_sdk2py.core.channel import (
        ChannelSubscriber,
        ChannelFactoryInitialize,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
except ImportError as exc:
    raise ImportError(
        "unitree_sdk2py 不可导入。装到当前环境，或把 UNITREE_SDK2_PY "
        "环境变量指到 unitree_sdk2_python 仓库目录。"
    ) from exc

K_G1 = cfg.K_G1
D_G1 = cfg.D_G1

BOARD = BoardConfig(
    dictionary=cfg.BOARD_DICT,
    squares_x=cfg.SQUARES_X,
    squares_y=cfg.SQUARES_Y,
    square_length_m=cfg.SQUARE_LENGTH_M,
    marker_length_m=cfg.MARKER_LENGTH_M,
)

MIN_CORNERS = cfg.MIN_CORNERS
STALE_SEC = 0.5

# ============================================================
# TRANSFORM UTILITIES
#
# A_T_B means:
# coordinates in B -> coordinates in A
# ============================================================

def T_rt(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]

    X = np.eye(4)
    X[:3, :3] = R.T
    X[:3, 3] = -R.T @ t
    return X


def xyz_rpy_T(xyz, rpy):
    return T_rt(
        Rotation.from_euler("xyz", rpy).as_matrix(),
        xyz,
    )


def axis_T(axis, q):
    return T_rt(
        Rotation.from_rotvec(
            np.asarray(axis, dtype=float) * float(q)
        ).as_matrix(),
        [0, 0, 0],
    )


def transform_rpy_deg(T):
    return Rotation.from_matrix(
        T[:3, :3]
    ).as_euler("xyz", degrees=True)


# ============================================================
# LOAD CORRECT MODE_MACHINE=5 REV1.0 URDF
#
# pelvis
#   -> waist_yaw
#   -> waist_roll
#   -> waist_pitch
#   -> torso
#   -> d435_link
# ============================================================

if not Path(URDF).exists():
    raise FileNotFoundError(
        f"URDF 不存在: {URDF}（用环境变量 G1_URDF 指定）"
    )

root = ET.parse(URDF).getroot()


def vec3(text):
    return np.array(
        [float(x) for x in (text or "0 0 0").split()],
        dtype=float,
    )


def get_joint(name):
    j = root.find(f".//joint[@name='{name}']")

    if j is None:
        raise RuntimeError(
            f"Missing URDF joint: {name}"
        )

    origin = j.find("origin")
    axis = j.find("axis")

    return {
        "name": name,
        "type": j.get("type"),
        "parent": j.find("parent").get("link"),
        "child": j.find("child").get("link"),
        "xyz": vec3(
            origin.get("xyz")
            if origin is not None
            else None
        ),
        "rpy": vec3(
            origin.get("rpy")
            if origin is not None
            else None
        ),
        "axis": (
            None
            if axis is None
            else vec3(axis.get("xyz"))
        ),
    }


JY = get_joint("waist_yaw_joint")
JR = get_joint("waist_roll_joint")
JP = get_joint("waist_pitch_joint")
JD = get_joint("d435_joint")


def joint_T(j, q=None):

    T = xyz_rpy_T(
        j["xyz"],
        j["rpy"],
    )

    if j["type"] in ("revolute", "continuous"):

        if q is None:
            raise RuntimeError(
                f"Missing q for {j['name']}"
            )

        T = T @ axis_T(
            j["axis"],
            q,
        )

    return T


def pelvis_T_d435(q):

    return (
        joint_T(JY, q[0])
        @ joint_T(JR, q[1])
        @ joint_T(JP, q[2])
        @ joint_T(JD)
    )


# Standard ROS camera link -> optical frame convention:
#
# d435_link:
#   x forward
#   y left
#   z up
#
# optical:
#   x right
#   y down
#   z forward
#
# D_T_C maps RGB optical -> d435_link
R_D_FROM_C_ROS = np.array([
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
], dtype=float)

D_T_C_ROS = T_rt(
    R_D_FROM_C_ROS,
    [0, 0, 0],
)

D_T_C_IDENTITY = np.eye(4)


# ============================================================
# LIVE G1 LOWSTATE
# ============================================================

class LowStateReader:

    def __init__(self):

        self.lock = threading.Lock()

        self.q = None
        self.mode_machine = None
        self.last_t = 0.0

        self.error = ""
        self.n_errors = 0

        ChannelFactoryInitialize(
            0,
            G1_IFACE,
        )

        self.sub = ChannelSubscriber(
            "rt/lowstate",
            LowState_,
        )

        self.sub.Init(
            self.callback,
            10,
        )

    def callback(self, msg):

        try:

            q = np.array([
                msg.motor_state[12].q,
                msg.motor_state[13].q,
                msg.motor_state[14].q,
            ], dtype=float)

            with self.lock:

                self.q = q
                self.mode_machine = int(
                    msg.mode_machine
                )
                self.last_t = time.monotonic()

        except Exception as exc:

            # Do not stay silent: q would remain None forever
            # and the UI would only say "tracking lost".
            with self.lock:
                self.n_errors += 1
                self.error = repr(exc)
                first = self.n_errors == 1

            if first:
                print(
                    "LowStateReader: cannot read waist joints "
                    f"(motor_state[12..14]) / mode_machine: {exc!r}",
                    file=sys.stderr,
                )

    def get(self):

        with self.lock:

            return (
                None
                if self.q is None
                else self.q.copy(),

                self.mode_machine,
                self.last_t,
            )


# ============================================================
# DRAWING HELPERS
# ============================================================

def put(
    img,
    text,
    x,
    y,
    color=(230, 230, 230),
    scale=0.55,
    thickness=2,
):

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_pose_axes(
    img,
    T,
    K,
    D,
    label,
    length=0.10,
):

    if T is None:
        return

    # Do not project poses behind external camera
    if T[2, 3] <= 0.01:
        return

    try:

        rvec, _ = cv2.Rodrigues(
            T[:3, :3]
        )

        tvec = T[:3, 3].reshape(3, 1)

        cv2.drawFrameAxes(
            img,
            K,
            D,
            rvec,
            tvec,
            length,
            2,
        )

        origin, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=float),
            rvec,
            tvec,
            K,
            D,
        )

        u, v = np.round(
            origin.reshape(2)
        ).astype(int)

        if (
            0 <= u < img.shape[1]
            and 0 <= v < img.shape[0]
        ):

            put(
                img,
                label,
                u + 8,
                v - 8,
                (255, 255, 255),
                0.50,
            )

    except Exception:
        pass


# ============================================================
# PANEL SIZE (used for the preview / info panel layout)
# ============================================================

WORLD_W = 800
WORLD_H = 400


# ============================================================
# INTERACTIVE 3D VIEWER
#
# No Open3D / matplotlib required.
#
# Mouse:
#   Left drag  = orbit
#   Right drag = pan
#   Wheel      = zoom
#
# Keyboard:
#   H = auto-fit/reset
#   1 = top view
#   2 = side view
#   3 = perspective view
#   + / - = zoom
#
# Display coordinates are chosen only for intuitive viewing:
#
#   visual X = External X  (right)
#   visual Y = External Z  (forward)
#   visual Z = -External Y (up)
#
# The underlying transforms remain in the external D435i
# optical frame E.
# ============================================================

class Interactive3DViewer:

    def __init__(self, width=1000, height=700):

        self.width = width
        self.height = height

        self.yaw = 40.0
        self.pitch = 25.0
        self.distance = 2.5

        self.target = np.zeros(3, dtype=float)

        self.drag_mode = None
        self.last_mouse = None

        self.need_fit = True

        # Interactive viewer fixed frame = ChArUco K.
        #
        # The view frame is K under a proper axis permutation
        # (cfg.VIS_AXES_FROM_K, right-handed, det = +1, checked
        # by parse_axis_remap). View +Z is "up" for the orbit
        # camera. The previous diag(1, 1, -1) was a reflection
        # and mirrored the whole scene (rotation sense inverted,
        # left/right swapped). This affects VISUALIZATION ONLY.
        self.axes_spec = cfg.VIS_AXES_FROM_K

        # parse_axis_remap: columns = view axes expressed in K,
        # i.e. v_K = R @ v_view. We need v_view = R.T @ v_K.
        self.R_VIS_FROM_E = parse_axis_remap(
            self.axes_spec
        ).T

    # --------------------------------------------------------

    def e_to_vis(self, p):

        p = np.asarray(
            p,
            dtype=float,
        ).reshape(3)

        return (
            self.R_VIS_FROM_E
            @ p
        )

    # --------------------------------------------------------

    def camera_basis(self):

        yaw = np.radians(
            self.yaw
        )

        pitch = np.radians(
            self.pitch
        )

        # Camera position orbiting target
        offset = self.distance * np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ])

        cam = (
            self.target
            + offset
        )

        forward = (
            self.target
            - cam
        )

        n = np.linalg.norm(
            forward
        )

        if n < 1e-9:
            forward = np.array(
                [0.0, 1.0, 0.0]
            )
        else:
            forward /= n

        world_up = np.array([
            0.0,
            0.0,
            1.0,
        ])

        right = np.cross(
            forward,
            world_up,
        )

        n = np.linalg.norm(
            right
        )

        if n < 1e-6:

            world_up = np.array([
                0.0,
                1.0,
                0.0,
            ])

            right = np.cross(
                forward,
                world_up,
            )

            n = np.linalg.norm(
                right
            )

        right /= max(
            n,
            1e-9,
        )

        up = np.cross(
            right,
            forward,
        )

        up /= max(
            np.linalg.norm(up),
            1e-9,
        )

        return (
            cam,
            right,
            up,
            forward,
        )

    # --------------------------------------------------------

    def project_vis(self, p):

        p = np.asarray(
            p,
            dtype=float,
        ).reshape(3)

        (
            cam,
            right,
            up,
            forward,
        ) = self.camera_basis()

        rel = p - cam

        xc = float(
            np.dot(rel, right)
        )

        yc = float(
            np.dot(rel, up)
        )

        zc = float(
            np.dot(rel, forward)
        )

        if zc <= 0.02:
            return None

        f = (
            0.82
            * min(
                self.width,
                self.height,
            )
        )

        u = (
            self.width * 0.5
            + f * xc / zc
        )

        v = (
            self.height * 0.5
            - f * yc / zc
        )

        return np.array([
            int(round(u)),
            int(round(v)),
        ])

    # --------------------------------------------------------

    def project_e(self, p):

        return self.project_vis(
            self.e_to_vis(p)
        )

    # --------------------------------------------------------

    def draw_line_e(
        self,
        img,
        a,
        b,
        color,
        thickness=1,
    ):

        pa = self.project_e(a)
        pb = self.project_e(b)

        if (
            pa is None
            or pb is None
        ):
            return

        cv2.line(
            img,
            tuple(pa),
            tuple(pb),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------

    def draw_frame(
        self,
        img,
        T,
        label,
        length=0.15,
    ):

        if T is None:
            return

        T = np.asarray(
            T,
            dtype=float,
        )

        o = T[:3, 3]

        po = self.project_e(o)

        if po is None:
            return

        cv2.circle(
            img,
            tuple(po),
            5,
            (245, 245, 245),
            -1,
        )

        # OpenCV BGR:
        # X red, Y green, Z blue
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]

        labels = [
            "X",
            "Y",
            "Z",
        ]

        for i in range(3):

            p = (
                o
                + T[:3, :3][:, i]
                * length
            )

            pp = self.project_e(p)

            if pp is None:
                continue

            cv2.line(
                img,
                tuple(po),
                tuple(pp),
                colors[i],
                3,
                cv2.LINE_AA,
            )

            put(
                img,
                labels[i],
                int(pp[0]) + 3,
                int(pp[1]) - 3,
                colors[i],
                0.38,
                1,
            )

        put(
            img,
            label,
            int(po[0]) + 8,
            int(po[1]) - 8,
            (255, 255, 255),
            0.48,
            1,
        )

    # --------------------------------------------------------

    def draw_camera_frustum(
        self,
        img,
        E_T_C,
        depth=0.20,
    ):

        if E_T_C is None:
            return

        # Camera optical coordinates:
        #
        # x right
        # y down
        # z forward

        w = depth * 0.60
        h = depth * 0.45

        local = np.array([
            [-w, -h, depth],
            [ w, -h, depth],
            [ w,  h, depth],
            [-w,  h, depth],
        ])

        R = E_T_C[:3, :3]
        t = E_T_C[:3, 3]

        corners = [
            R @ p + t
            for p in local
        ]

        origin = t

        for p in corners:

            self.draw_line_e(
                img,
                origin,
                p,
                (210, 210, 210),
                1,
            )

        for i in range(4):

            self.draw_line_e(
                img,
                corners[i],
                corners[
                    (i + 1) % 4
                ],
                (210, 210, 210),
                1,
            )

    # --------------------------------------------------------

    def draw_board(
        self,
        img,
        E_T_K,
    ):

        if E_T_K is None:
            return

        # Physical ChArUco board:
        # 7 * 40 mm = 0.28 m

        size = 0.28

        local = np.array([
            [0.0,  0.0, 0.0],
            [size, 0.0, 0.0],
            [size, size, 0.0],
            [0.0,  size, 0.0],
        ])

        R = E_T_K[:3, :3]
        t = E_T_K[:3, 3]

        pts = [
            R @ p + t
            for p in local
        ]

        for i in range(4):

            self.draw_line_e(
                img,
                pts[i],
                pts[
                    (i + 1) % 4
                ],
                (0, 220, 220),
                2,
            )

    # --------------------------------------------------------

    def draw_trajectory(
        self,
        img,
        trajectory,
    ):

        if len(trajectory) < 2:
            return

        traj = trajectory[-250:]

        for a, b in zip(
            traj[:-1],
            traj[1:],
        ):

            self.draw_line_e(
                img,
                a,
                b,
                (0, 180, 255),
                2,
            )

    # --------------------------------------------------------

    def scene_points(
        self,
        E_T_B,
        E_T_C,
        E_T_K,
        trajectory,
    ):

        pts = [
            np.zeros(3),
        ]

        for T in (
            E_T_B,
            E_T_C,
            E_T_K,
        ):

            if T is not None:

                pts.append(
                    np.asarray(
                        T[:3, 3],
                        dtype=float,
                    )
                )

        if trajectory:

            pts.extend(
                trajectory[-100:]
            )

        return np.array(
            [
                self.e_to_vis(p)
                for p in pts
            ],
            dtype=float,
        )

    # --------------------------------------------------------

    def fit_scene(
        self,
        E_T_B,
        E_T_C,
        E_T_K,
        trajectory,
    ):

        pts = self.scene_points(
            E_T_B,
            E_T_C,
            E_T_K,
            trajectory,
        )

        if len(pts) == 0:
            return

        mn = pts.min(
            axis=0
        )

        mx = pts.max(
            axis=0
        )

        self.target = (
            mn + mx
        ) * 0.5

        extent = float(
            np.max(
                mx - mn
            )
        )

        self.distance = max(
            0.8,
            extent * 2.2 + 0.5,
        )

        self.need_fit = False

    # --------------------------------------------------------

    def reset_view(self):

        self.yaw = 40.0
        self.pitch = 25.0
        self.need_fit = True

    # --------------------------------------------------------

    def top_view(self):

        self.yaw = 0.0
        self.pitch = 89.0

    # --------------------------------------------------------

    def side_view(self):

        self.yaw = 0.0
        self.pitch = 0.0

    # --------------------------------------------------------

    def perspective_view(self):

        self.yaw = 40.0
        self.pitch = 25.0

    # --------------------------------------------------------

    def zoom(self, factor):

        self.distance *= float(
            factor
        )

        self.distance = np.clip(
            self.distance,
            0.15,
            30.0,
        )

    # --------------------------------------------------------

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.drag_mode = "orbit"

            self.last_mouse = (
                x,
                y,
            )

        elif event == cv2.EVENT_RBUTTONDOWN:

            self.drag_mode = "pan"

            self.last_mouse = (
                x,
                y,
            )

        elif event in (
            cv2.EVENT_LBUTTONUP,
            cv2.EVENT_RBUTTONUP,
        ):

            self.drag_mode = None
            self.last_mouse = None

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and self.drag_mode is not None
            and self.last_mouse is not None
        ):

            lx, ly = self.last_mouse

            dx = x - lx
            dy = y - ly

            self.last_mouse = (
                x,
                y,
            )

            if self.drag_mode == "orbit":

                self.yaw -= (
                    dx * 0.35
                )

                self.pitch += (
                    dy * 0.35
                )

                self.pitch = np.clip(
                    self.pitch,
                    -85.0,
                    85.0,
                )

            elif self.drag_mode == "pan":

                (
                    cam,
                    right,
                    up,
                    forward,
                ) = self.camera_basis()

                scale = (
                    self.distance
                    * 0.0015
                )

                self.target += (
                    -dx
                    * scale
                    * right
                    + dy
                    * scale
                    * up
                )

        elif event == cv2.EVENT_MOUSEWHEEL:

            try:

                delta = (
                    cv2.getMouseWheelDelta(
                        flags
                    )
                )

            except Exception:

                delta = (
                    1
                    if flags > 0
                    else -1
                )

            if delta > 0:

                self.zoom(
                    0.88
                )

            else:

                self.zoom(
                    1.14
                )

    # --------------------------------------------------------

    def render(
        self,
        E_T_B,
        E_T_C,
        E_T_K,
        trajectory,
        tracking_ok,
    ):

        if self.need_fit:

            self.fit_scene(
                E_T_B,
                E_T_C,
                E_T_K,
                trajectory,
            )

        img = np.zeros(
            (
                self.height,
                self.width,
                3,
            ),
            dtype=np.uint8,
        )

        # ----------------------------------------------------
        # Root trajectory
        # ----------------------------------------------------

        self.draw_trajectory(
            img,
            trajectory,
        )

        # ----------------------------------------------------
        # Relationships
        # ----------------------------------------------------

        if E_T_B is not None:

            self.draw_line_e(
                img,
                [0, 0, 0],
                E_T_B[:3, 3],
                (80, 80, 80),
                1,
            )

        if (
            E_T_B is not None
            and E_T_C is not None
        ):

            self.draw_line_e(
                img,
                E_T_B[:3, 3],
                E_T_C[:3, 3],
                (150, 150, 150),
                2,
            )

        # ----------------------------------------------------
        # Board
        # ----------------------------------------------------

        # ChArUco is the fixed visualization world.
        self.draw_board(
            img,
            np.eye(4),
        )

        # ----------------------------------------------------
        # Coordinate frames
        # ----------------------------------------------------

        self.draw_frame(
            img,
            np.eye(4),
            "CHARUCO / FIXED WORLD K",
            0.18,
        )

        # E_T_K argument now carries K_T_E for visualization.
        self.draw_frame(
            img,
            E_T_K,
            "EXTERNAL D435i E",
            0.14,
        )

        self.draw_frame(
            img,
            E_T_B,
            "G1 ROOT / PELVIS B",
            0.18,
        )

        self.draw_frame(
            img,
            E_T_C,
            "G1 CAMERA C",
            0.12,
        )

        # Camera frustum
        self.draw_camera_frustum(
            img,
            E_T_C,
        )

        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        put(
            img,
            "INTERACTIVE COMMON FRAME",
            15,
            28,
            (255, 255, 255),
            0.62,
        )

        put(
            img,
            (
                "Fixed frame = ChArUco K | "
                f"view X,Y,Z = K {self.axes_spec} | view Z up"
            ),
            15,
            54,
            (180, 180, 180),
            0.45,
        )

        put(
            img,
            (
                "Left drag orbit | "
                "Right drag pan | Wheel zoom"
            ),
            15,
            self.height - 48,
            (190, 190, 190),
            0.45,
        )

        put(
            img,
            (
                "H reset | 1 top | "
                "2 side | 3 perspective | +/- zoom"
            ),
            15,
            self.height - 22,
            (190, 190, 190),
            0.45,
        )

        status = (
            "TRACKING OK"
            if tracking_ok
            else "TRACKING WAITING"
        )

        status_color = (
            (0, 255, 0)
            if tracking_ok
            else (0, 140, 255)
        )

        put(
            img,
            status,
            self.width - 190,
            28,
            status_color,
            0.52,
        )

        return img



# ============================================================
# INFO PANEL
# ============================================================

def fmt_pose(T):

    if T is None:
        return None, None

    xyz = T[:3, 3]
    rpy = transform_rpy_deg(T)

    return xyz, rpy


def render_info(
    E_T_B,
    E_T_C,
    E_T_K,
    q,
    mode_machine,
    tracking_ok,
    frame_mode,
    g1_corners,
    ext_corners,
):

    panel = np.zeros(
        (WORLD_H, 480, 3),
        dtype=np.uint8,
    )

    put(
        panel,
        "COMMON FRAME STATUS",
        15,
        28,
        (255, 255, 255),
        0.62,
    )

    mm_ok = (
        mode_machine == cfg.EXPECTED_MODE_MACHINE
    )

    put(
        panel,
        (
            f"mode_machine: {mode_machine} "
            f"({'CORRECT rev1.0' if mm_ok else 'EXPECTED 5'})"
        ),
        15,
        60,
        (
            (0, 255, 0)
            if mm_ok
            else (0, 0, 255)
        ),
        0.48,
    )

    put(
        panel,
        f"d435 frame mode: {frame_mode}",
        15,
        86,
        (220, 220, 220),
        0.48,
    )

    put(
        panel,
        (
            f"Charuco corners: "
            f"G1={g1_corners} EXT={ext_corners}"
        ),
        15,
        112,
        (
            (0, 255, 0)
            if tracking_ok
            else (0, 180, 255)
        ),
        0.48,
    )

    y = 148

    if q is not None:

        qdeg = np.degrees(q)

        put(
            panel,
            (
                "waist rad: "
                f"{q[0]:+.3f} "
                f"{q[1]:+.3f} "
                f"{q[2]:+.3f}"
            ),
            15,
            y,
            (220, 220, 220),
            0.47,
        )

        y += 24

        put(
            panel,
            (
                "waist deg: "
                f"{qdeg[0]:+.1f} "
                f"{qdeg[1]:+.1f} "
                f"{qdeg[2]:+.1f}"
            ),
            15,
            y,
            (220, 220, 220),
            0.47,
        )

        y += 34

    xyz_B, rpy_B = fmt_pose(
        E_T_B
    )

    if xyz_B is not None:

        put(
            panel,
            "G1 ROOT in External E:",
            15,
            y,
            (255, 255, 255),
            0.50,
        )

        y += 25

        put(
            panel,
            (
                "xyz [m]: "
                f"{xyz_B[0]:+.3f} "
                f"{xyz_B[1]:+.3f} "
                f"{xyz_B[2]:+.3f}"
            ),
            15,
            y,
            (220, 220, 220),
            0.46,
        )

        y += 23

        put(
            panel,
            (
                "rpy [deg]: "
                f"{rpy_B[0]:+.1f} "
                f"{rpy_B[1]:+.1f} "
                f"{rpy_B[2]:+.1f}"
            ),
            15,
            y,
            (220, 220, 220),
            0.46,
        )

        y += 34

    xyz_C, rpy_C = fmt_pose(
        E_T_C
    )

    if xyz_C is not None:

        put(
            panel,
            "G1 CAMERA in External E:",
            15,
            y,
            (255, 255, 255),
            0.50,
        )

        y += 25

        put(
            panel,
            (
                "xyz [m]: "
                f"{xyz_C[0]:+.3f} "
                f"{xyz_C[1]:+.3f} "
                f"{xyz_C[2]:+.3f}"
            ),
            15,
            y,
            (220, 220, 220),
            0.46,
        )

        y += 23

        put(
            panel,
            (
                "rpy [deg]: "
                f"{rpy_C[0]:+.1f} "
                f"{rpy_C[1]:+.1f} "
                f"{rpy_C[2]:+.1f}"
            ),
            15,
            y,
            (220, 220, 220),
            0.46,
        )

    put(
        panel,
        "P print matrices | R clear trail",
        15,
        WORLD_H - 42,
        (170, 170, 170),
        0.43,
    )

    put(
        panel,
        "F frame mode | Q quit",
        15,
        WORLD_H - 18,
        (170, 170, 170),
        0.43,
    )

    return panel


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 68)
    print("G1 + EXTERNAL CAMERA COMMON-FRAME VISUALIZER")
    print("=" * 68)
    print("World frame E = external D435i optical frame")
    print()
    print("Robot model:")
    print(URDF)
    print()
    print("Expected mode_machine = 5")
    print()
    print("Keep external D435i and ChArUco board fixed.")
    print("The G1 may move naturally.")
    print()
    print("P = print transforms")
    print("R = clear root trajectory")
    print("F = toggle d435 optical-frame convention")
    print("Q = quit")
    print("=" * 68)
    print()

    print("Loaded rev1.0 geometry:")
    print(
        " waist_roll:",
        JR["xyz"],
    )
    print(
        " waist_pitch:",
        JP["xyz"],
    )
    print(
        " d435:",
        JD["xyz"],
    )
    print()

    low = LowStateReader()

    # --------------------------------------------------------
    # G1 onboard camera ZMQ
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
        G1_ENDPOINT
    )

    # --------------------------------------------------------
    # External D435i
    # --------------------------------------------------------

    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_device(
        EXTERNAL_SERIAL
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

    vp = (
        profile
        .get_stream(rs.stream.color)
        .as_video_stream_profile()
    )

    intr = vp.get_intrinsics()

    K_EXT = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
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
        camera_matrix=K_G1.tolist(),
        distortion=D_G1.tolist(),
        source="g1_rgb",
    )

    ext_intr = CameraIntrinsics(
        width=640,
        height=480,
        camera_matrix=K_EXT.tolist(),
        distortion=D_EXT.tolist(),
        source="external_d435i",
    )

    estimator = CharucoEstimator(
        BOARD,
        min_corners=MIN_CORNERS,
    )

    # --------------------------------------------------------

    frame_mode = "ros_optical"

    last_E_T_B = None
    last_E_T_C = None
    last_E_T_K = None

    trajectory = []

    cv2.namedWindow(
        "G1 Common Frame",
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

    viewer3d = Interactive3DViewer(
        width=1000,
        height=700,
    )

    cv2.setMouseCallback(
        "Interactive 3D World",
        viewer3d.mouse_callback,
    )

    try:

        while True:

            # =================================================
            # G1 RGB FRAME
            # =================================================

            try:

                payload = sock.recv()

            except zmq.Again:

                print(
                    "Waiting for G1 camera..."
                )

                continue

            data = msgpack.unpackb(
                payload,
                raw=False,
            )

            value = (
                data
                .get("images", {})
                .get("ego_view")
            )

            if value is None:
                continue

            buf = (
                base64.b64decode(value)
                if isinstance(value, str)
                else value
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

            # Keep same convention as your already-working script
            g1 = cv2.cvtColor(
                g1,
                cv2.COLOR_RGB2BGR,
            )

            # =================================================
            # EXTERNAL CAMERA FRAME
            # =================================================

            try:

                frames = pipe.wait_for_frames(
                    2000
                )

            except RuntimeError as exc:

                # External D435i hiccup: keep running instead
                # of taking the whole program down.
                print(
                    "External D435i timeout:",
                    exc,
                )

                continue

            cf = frames.get_color_frame()

            if not cf:
                continue

            ext = np.asanyarray(
                cf.get_data()
            ).copy()

            # =================================================
            # DETECT BOARD
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

            q, mode_machine, low_t = low.get()

            now = time.monotonic()

            low_ok = (
                q is not None
                and now - low_t < STALE_SEC
            )

            g1_ok = (
                dg is not None
                and dg.C_T_K is not None
            )

            ext_ok = (
                de is not None
                and de.C_T_K is not None
            )

            tracking_ok = (
                low_ok
                and g1_ok
                and ext_ok
            )

            # =================================================
            # COMMON FRAME TRANSFORMS
            # =================================================

            if tracking_ok:

                # Board pose in external-camera frame
                E_T_K = np.asarray(
                    de.C_T_K,
                    dtype=float,
                )

                # Board pose in G1 optical-camera frame
                C_T_K = np.asarray(
                    dg.C_T_K,
                    dtype=float,
                )

                # ------------------------------------------------
                # G1 optical camera pose in external world
                #
                # E_T_C = E_T_K * inverse(C_T_K)
                # ------------------------------------------------

                E_T_C = (
                    E_T_K
                    @ invT(C_T_K)
                )

                # ------------------------------------------------
                # pelvis/root -> d435_link
                # using CORRECT rev1.0 URDF + live waist
                # ------------------------------------------------

                B_T_D = pelvis_T_d435(
                    q
                )

                if frame_mode == "ros_optical":

                    D_T_C = D_T_C_ROS

                else:

                    D_T_C = D_T_C_IDENTITY

                B_T_C = (
                    B_T_D
                    @ D_T_C
                )

                # ------------------------------------------------
                # G1 pelvis/root pose in external world
                #
                # E_T_B = E_T_C * inverse(B_T_C)
                # ------------------------------------------------

                E_T_B = (
                    E_T_C
                    @ invT(B_T_C)
                )

                last_E_T_B = (
                    E_T_B.copy()
                )

                last_E_T_C = (
                    E_T_C.copy()
                )

                last_E_T_K = (
                    E_T_K.copy()
                )

                # Root trajectory
                p = E_T_B[:3, 3].copy()

                if (
                    len(trajectory) == 0
                    or np.linalg.norm(
                        p - trajectory[-1]
                    ) > 0.005
                ):

                    trajectory.append(
                        p
                    )

                    trajectory = (
                        trajectory[-250:]
                    )

            # =================================================
            # 2D IMAGE OVERLAYS
            # =================================================

            if last_E_T_K is not None:

                draw_pose_axes(
                    ext_vis,
                    last_E_T_K,
                    K_EXT,
                    D_EXT,
                    "BOARD",
                    0.08,
                )

            if last_E_T_B is not None:

                draw_pose_axes(
                    ext_vis,
                    last_E_T_B,
                    K_EXT,
                    D_EXT,
                    "G1 ROOT",
                    0.15,
                )

            if last_E_T_C is not None:

                draw_pose_axes(
                    ext_vis,
                    last_E_T_C,
                    K_EXT,
                    D_EXT,
                    "G1 CAM",
                    0.10,
                )

            if (
                dg is not None
                and dg.C_T_K is not None
            ):

                draw_pose_axes(
                    g1_vis,
                    np.asarray(
                        dg.C_T_K,
                        dtype=float,
                    ),
                    K_G1,
                    D_G1,
                    "BOARD",
                    0.08,
                )

            put(
                g1_vis,
                "G1 ONBOARD RGB",
                15,
                28,
                (0, 255, 0),
                0.65,
            )

            put(
                ext_vis,
                "EXTERNAL D435i = WORLD E",
                15,
                28,
                (0, 255, 0),
                0.62,
            )

            if not tracking_ok:

                put(
                    ext_vis,
                    "COMMON-FRAME TRACKING LOST",
                    15,
                    460,
                    (0, 0, 255),
                    0.62,
                )

            # =================================================
            # 3D WORLD + INFO
            # =================================================

            # =================================================
            # RViz-style visualization fixed frame = ChArUco K
            #
            # IMPORTANT:
            # These K-frame transforms are ONLY for visualization.
            # last_E_T_* remain unchanged for the real pipeline.
            # =================================================

            K_T_E = None
            K_T_B = None
            K_T_C = None
            trajectory_K = []

            if last_E_T_K is not None:

                K_T_E = invT(last_E_T_K)

                if last_E_T_B is not None:
                    K_T_B = (
                        K_T_E
                        @ last_E_T_B
                    )

                if last_E_T_C is not None:
                    K_T_C = (
                        K_T_E
                        @ last_E_T_C
                    )

                R_KE = K_T_E[:3, :3]
                t_KE = K_T_E[:3, 3]

                trajectory_K = [
                    R_KE @ np.asarray(p) + t_KE
                    for p in trajectory
                ]

            interactive_world = viewer3d.render(
                K_T_B,
                K_T_C,
                K_T_E,
                trajectory_K,
                tracking_ok,
            )

            cv2.imshow(
                "Interactive 3D World",
                interactive_world,
            )

            # Small synchronized preview inside main window
            world = cv2.resize(
                interactive_world,
                (WORLD_W, WORLD_H),
                interpolation=cv2.INTER_AREA,
            )

            info = render_info(
                last_E_T_B,
                last_E_T_C,
                last_E_T_K,
                q,
                mode_machine,
                tracking_ok,
                frame_mode,
                (
                    0
                    if dg is None
                    else dg.corner_count
                ),
                (
                    0
                    if de is None
                    else de.corner_count
                ),
            )

            top = np.hstack([
                ext_vis,
                g1_vis,
            ])

            bottom = np.hstack([
                world,
                info,
            ])

            gui = np.vstack([
                top,
                bottom,
            ])

            cv2.imshow(
                "G1 Common Frame",
                gui,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # =================================================
            # CONTROLS
            # =================================================

            if key in (
                ord("q"),
                ord("Q"),
            ):

                break

            elif key in (
                ord("r"),
                ord("R"),
            ):

                trajectory.clear()

                print(
                    "Root trajectory cleared."
                )

            elif key in (
                ord("f"),
                ord("F"),
            ):

                if frame_mode == "ros_optical":

                    frame_mode = "identity"

                else:

                    frame_mode = "ros_optical"

                trajectory.clear()

                print(
                    "D435 frame convention:",
                    frame_mode,
                )

            elif key in (
                ord("h"),
                ord("H"),
            ):

                viewer3d.reset_view()

            elif key == ord("1"):

                viewer3d.top_view()

            elif key == ord("2"):

                viewer3d.side_view()

            elif key == ord("3"):

                viewer3d.perspective_view()

            elif key in (
                ord("+"),
                ord("="),
            ):

                viewer3d.zoom(
                    0.85
                )

            elif key in (
                ord("-"),
                ord("_"),
            ):

                viewer3d.zoom(
                    1.18
                )

            elif key in (
                ord("p"),
                ord("P"),
            ):

                print()
                print("=" * 60)
                print("CURRENT COMMON-FRAME POSES")
                print("=" * 60)

                if last_E_T_B is not None:

                    print()
                    print(
                        "E_T_G1_ROOT:"
                    )

                    print(
                        last_E_T_B
                    )

                if last_E_T_C is not None:

                    print()
                    print(
                        "E_T_G1_CAMERA:"
                    )

                    print(
                        last_E_T_C
                    )

                if last_E_T_K is not None:

                    print()
                    print(
                        "E_T_CHARUCO:"
                    )

                    print(
                        last_E_T_K
                    )

                print()
                print("=" * 60)

    finally:

        pipe.stop()

        sock.close()

        ctx.term()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
