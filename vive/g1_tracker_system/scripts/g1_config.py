"""G1 tracker system 的集中配置。

所有路径相对工程目录（``g1_tracker_system/``）解析，可用环境变量覆盖：

    G1_IFACE              G1 DDS 网卡名
    G1_ENDPOINT           G1 板载相机 ZMQ 地址
    EXTERNAL_D435I_SERIAL 外部 D435i 序列号
    G1_URDF               FK 用的 URDF
    UNITREE_SDK2_PY       unitree_sdk2_python 仓库目录（unitree_sdk2py 不在 PYTHONPATH 时用）
    VIVE_TRACKER_ID       tracker device id（MAC 后 4 字节，如 0d:e1:7b:f0）
    G1_TRACKER_TF         T_tracker_from_g1_root 4x4 文本文件
    G1_DUAL_CAM_RESULTS   双相机标定输出目录
    G1_VIS_AXES_FROM_K    3D 视图显示系相对 ChArUco K 的轴置换，默认 -X,Y,-Z（须右手系）
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "model"
CALIB_DIR = PROJECT_DIR / "calibration"


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else Path(default)


# ---------------------------------------------------------------- 机器人 / 网络
G1_IFACE = os.environ.get("G1_IFACE", "enx349971ea2929")
G1_ENDPOINT = os.environ.get("G1_ENDPOINT", "tcp://192.168.123.164:5555")
EXTERNAL_SERIAL = os.environ.get("EXTERNAL_D435I_SERIAL", "262322070500")

# ---------------------------------------------------------------- 模型
#: 唯一的 URDF：FK（pelvis -> waist_yaw/roll/pitch -> torso -> d435_joint）+ vive_tracker_link，
#: mode_machine=5 对应 rev1.0
URDF = _env_path("G1_URDF", MODEL_DIR / "g1_29dof_with_hand_rev_1_0.urdf")
EXPECTED_MODE_MACHINE = 5

#: unitree_sdk2py 所在目录；为空表示已在 PYTHONPATH
UNITREE_SDK = os.environ.get("UNITREE_SDK2_PY", "")

# ---------------------------------------------------------------- G1 板载 RGB 内参（640x480）
G1_IMAGE_SIZE = (640, 480)
K_G1 = np.array([
    [604.9285, 0.0, 329.4290],
    [0.0, 605.6724, 246.9297],
    [0.0, 0.0, 1.0],
], dtype=float)
#: G1 端没有给畸变系数，按 0 处理
D_G1 = np.zeros(5, dtype=float)

# ---------------------------------------------------------------- ChArUco 板
BOARD_DICT = "DICT_6X6_250"
SQUARES_X = 7
SQUARES_Y = 7
SQUARE_LENGTH_M = 0.040
MARKER_LENGTH_M = 0.030
MIN_CORNERS = 8

# ---------------------------------------------------------------- VIVE tracker
#: 默认 tracker device id；会随设备更换，优先用命令行 --tracker 或环境变量
TARGET_TRACKER = os.environ.get("VIVE_TRACKER_ID", "0d:e1:7b:f0")
#: T_tracker_from_g1_root（T 系 = Z-up + BODY_AXES_REMAP "Y,-X,Z"，与 URDF vive_tracker_link 一致）
TRACKER_TF_FILE = _env_path("G1_TRACKER_TF", CALIB_DIR / "T_tracker_from_g1_root.txt")
#: pelvis -> vive_tracker_link 的安装位置，与 URDF vive_tracker_joint 一致（rpy = 0，2026-09-02 实测验证旋转）
TRACKER_MOUNT_XYZ = (-0.06, 0.0, -0.08)

# ---------------------------------------------------------------- 可视化
#: 交互式 3D 视图的显示系相对 ChArUco K 系的轴置换（spec 含义见 frames.parse_axis_remap：
#: 显示系 X/Y/Z 分别指向 K 的哪根轴）。必须是右手系（det=+1），parse_axis_remap 会检查。
#: 显示系 +Z 是轨道相机的"上"。K 的 Z 朝下时用 "-X,Y,-Z"（绕 Y 转 180°）。
#: 旧代码用 diag(1,1,-1) 是反射，画面是镜像的。
VIS_AXES_FROM_K = os.environ.get("G1_VIS_AXES_FROM_K", "-X,Y,-Z")

# ---------------------------------------------------------------- 标定输出
DUAL_CAM_RESULTS_DIR = _env_path("G1_DUAL_CAM_RESULTS", CALIB_DIR / "dual_camera_charuco_results")
