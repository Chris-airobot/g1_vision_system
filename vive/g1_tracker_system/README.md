# Unitree G1 Root Tracking System

实时估计并可视化 Unitree G1 的 Root / Pelvis 位姿：用 ChArUco 双相机做初始化，
之后由贴在骨盆背面的 VIVE Ultimate Tracker 持续驱动。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/g1_config.py` | **所有常量与路径**（网卡、ZMQ、序列号、URDF、内参、板参数、tracker id、输出目录），可用环境变量覆盖 |
| `scripts/g1_common_frame_visualizer_interactive.py` | G1 rev1.0 FK、坐标变换、`LowStateReader`、交互式 3D 视图；也是其它脚本的公共模块 `base` |
| `scripts/g1_hybrid_tracker_visualizer.py` | **主运行脚本**：双相机 ChArUco 初始化 → tracker 持续跟踪；`--no-tracker-tf` 为临时模式 |
| `scripts/dual_camera_charuco_calib.py` | 双相机 ChArUco 一致性检查与验证 |
| `scripts/check_tracker_mount.py` | 核对 tracker 输出机体系与 URDF `vive_tracker_link` 是否一致（static / push） |
| `scripts/make_tracker_mesh.py` | 把 Ultimate Tracker CAD（OBJ，mm）转成 URDF 用的 STL |
| `model/g1_29dof_with_hand_rev_1_0.urdf` | 唯一的 URDF：FK 用（mode_machine = 5），含 `vive_tracker_link` |
| `model/meshes/` | 网格，含 `vive_ultimate_tracker.STL` |
| `calibration/dual_camera_charuco_results/` | 双相机标定输出（`T_external_from_g1_camera.txt`、`validation_runs.csv` 等；`validation.csv` 是旧格式，无 run_id） |
| `calibration/T_tracker_from_g1_root.txt` | `T_T_B` 4x4（默认路径，见下） |

## 环境

Python 解释器：conda env `vivavive`（`/home/ubuntu/anaconda3/envs/vivavive/bin/python`），
本包 `viva_vive_ultimate` 已装在里面。运行脚本还需要 `opencv-contrib-python`、`pyzmq`、
`msgpack`、`pyrealsense2`、`scipy`、`unitree_sdk2py`；`unitree_sdk2py` 不在 PYTHONPATH 时用
`UNITREE_SDK2_PY=/path/to/unitree_sdk2_python`。

脚本都从 `scripts/` 目录导入 `g1_config` 和 `base`，在任意 cwd 下运行都可以。

## 坐标系

| 记号 | 含义 |
|---|---|
| `K` | ChArUco 板系（OpenCV 原生），**固定世界** |
| `E` | 外部 D435i color optical frame |
| `C` | G1 板载相机 optical frame |
| `B` | G1 root / pelvis |
| `V` | tracker 的 SLAM 地图系，已转 Z-up |
| `T` | tracker 机体系 = Z-up + `BODY_AXES_REMAP="Y,-X,Z"`（X 沿厚度指向安装面、Y 79 mm 边、Z 59 mm 边） |

`A_T_B` 把 B 系坐标变到 A 系。`T` 与 URDF 的 `vive_tracker_link` 是同一个系，`T_T_B` 文件必须按它给。

Tracker 装在 pelvis 背面，`vive_tracker_joint` origin xyz = (-0.06, 0, -0.08)，rpy = 0。
2026-09-02 用 `check_tracker_mount.py` 实测：静态 body +Z 竖直偏 0.8°，前推位移在 body +X 偏 3.0°，
**旋转已验证**。SLAM 原点相对安装面中心的平移未验证，要靠手眼标定；名义 `T_T_B` 平移 = (0.06, 0, 0.08)。

## 流程

```bash
cd viva_vive_ultimate/g1_tracker_system/scripts
PY=/home/ubuntu/anaconda3/envs/vivavive/bin/python

# 0. tracker 启流（只发 ATM-1 + ATM20，SAFE；每次仍需用户明确要求）
vvu-monitor --enable-tracking

# 1. tracker 安装核对（先 static，再 push；tracker 可先放地上、朝向与装机一致）
$PY check_tracker_mount.py --mode static --device-id <id>
$PY check_tracker_mount.py --mode push   --device-id <id>

# 2. 双相机一致性检查（C 标定 / V 验证 / R 重置 / Q 退出）
$PY dual_camera_charuco_calib.py

# 3. 主脚本
$PY g1_hybrid_tracker_visualizer.py --tracker <id> --no-tracker-tf     # 临时模式
$PY g1_hybrid_tracker_visualizer.py --tracker <id> --tracker-tf ../calibration/T_tracker_from_g1_root.txt
```

主脚本初始化：确保 `VISION: VISIBLE` 与 `VIVE: OK`，机器人静止按 `I`，出现
`ALIGNMENT: LOCKED | ROOT SOURCE: VIVE` 后即可移动机器人。板再次可见时 HUD 显示
`TRACKER vs VISION` 误差。按键：`I` 初始化、`X` 清除对齐、`L` 重载 TF、`R` 清轨迹、
`F` 切 d435 光学系约定、`H/1/2/3/+/-` 视角、`Q` 退出。

临时模式（`--no-tracker-tf`）把 `T_T_B` 当单位阵：由于安装旋转已验证为单位阵，
**平移方向是对的**；机器人旋转时会有 tracker 到 pelvis 的杆臂误差（约 10 cm × sin θ）。

## 已修问题（2026-09-02）

评审发现并已修改，真机上还没跑过，第一次运行时留意：

1. 3D 视图原来用 `diag(1,1,-1)`（反射，画面镜像）。现在用 `g1_config.VIS_AXES_FROM_K`（默认 `-X,Y,-Z`，绕 Y 转 180°）经 `frames.parse_axis_remap` 生成真旋转，det=+1 有检查。显示系 +Z 朝上；若你的板 Z 实际朝上，把它改成 `X,Y,Z`。
2. `E_T_K_fixed` 只平均一次，`K_T_E_fixed` 取其逆，两者严格互逆。
3. `F` 键切换 d435 光学系约定时若已锁定或正在采集，会清除对齐并提示重新按 `I`。
4. 外部 D435i `wait_for_frames` 超时改为打印后继续，不再让程序退出、丢失对齐。base 与 hybrid 都改了。
5. `LowStateReader` 读不到 `motor_state[12..14]` 时首次打印异常到 stderr，并计数到 `error` / `n_errors`。
6. 标定验证改写到 `validation_runs.csv`，多了 `run_id`（标定完成时刻）和 `timestamp` 列；旧的 `validation.csv` 不再写入。
7. 视觉根位姿只需 G1 相机 + FK（`K_T_B = inv(C_T_K) · inv(B_T_C)`），外部相机不再是初始化前提，只用于在它自己的画面里投影；HUD 分别显示 `VISION (G1 cam+FK)` 和 `EXT cam`。初始化期间外部相机一次也没看到板时，没有外部画面 overlay，其它功能不受影响。
8. 标定脚本收不到 G1 图像时（`zmq.Again`）打印等待而不是崩溃。
9. `load_T_T_B` 额外校验旋转块正交且 det=+1。

## 未改的注意事项

1. 关节角 q、图像、tracker 位姿没有时间对齐；只在静止时读误差数字。
2. `SOLVEPNP_ITERATIVE` 对平面板有 Z 翻转二义性，无检查。
3. `T_T_B` 的真实值：用包里的 `solve_handeye`，把 `B_T_K = B_T_C · C_T_K` 当作 `C_T_K` 喂入即可，不必等外部提供。

已淘汰的 `g1_29dof_fakehand_freebase.urdf` 的 `d435_joint` z 比 rev_1_0 低 1 cm（0.41987 对 0.42987），
以 rev_1_0 为准。
