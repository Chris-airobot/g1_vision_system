"""把多台 tracker 各自独立的世界坐标系对齐到一起。

为什么需要
----------
每台 tracker 自己建图、自己定原点（见 docs/protocol.md「地图」一节）。
tracker A 报的位姿在 ``W_A`` 里，tracker B 在 ``W_B`` 里，**两者不可直接比较**。

HuMI 论文里「放一台在地上当 z=0 参考系」的做法解决的是**共享地图内**的原点问题，
并不能合并两张独立的地图 —— 那台地面 tracker 只知道自己在自己那张图里的位姿。

做法：共位标定
--------------
录制开始前把几台 tracker **刚性绑在一起**（或攥成一把）移动 30-40 秒。
它们走的是同一条物理轨迹，于是可以从两条轨迹反解出 ``W_A <- W_B`` 的固定变换：

    p_A(t) ≈ R · p_B(t) + t

这就是 Kabsch/Umeyama 问题。标定完把它们分开戴上，之后所有位姿都能换算到同一系。

两个精度要点
------------
- **要有平移**。只旋转不平移解不出来（轨迹退化成一个点）。
- **杆臂误差**：两台 tracker 物理上相隔 d，旋转时各自轨迹差最多 d。
  所以标定动作**以平移为主、少旋转**，或者把它们贴得越近越好。
  :func:`estimate_transform` 返回的 ``rms`` 直接反映这个误差。

若两台的世界系都重力对齐，则 ``R`` 只差一个偏航角。
:func:`rotation_axis_angle` 可以检验这一点 —— 转轴接近竖直就说明成立。
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence


class Transform(NamedTuple):
    """``p_a = R @ p_b + t``，即把 B 系的点搬到 A 系。"""

    R: "object"          #: (3,3) numpy 数组
    t: "object"          #: (3,) numpy 数组
    rms_m: float         #: 拟合残差 RMS（米），反映杆臂与噪声
    samples: int
    scale: float         #: Umeyama 尺度。应该 ≈1；明显偏离说明配错了

    def apply(self, points):
        """把 B 系下的点批量换算到 A 系。points 形如 (N,3) 或 (3,)。"""
        np = _np()
        p = np.asarray(points, dtype=float)
        return (self.R @ p.T).T + self.t if p.ndim == 2 else self.R @ p + self.t

    def inverse(self) -> "Transform":
        np = _np()
        Rt = self.R.T
        return Transform(Rt, -Rt @ self.t, self.rms_m, self.samples, 1.0 / self.scale)


def _np():
    try:
        import numpy as np
    except ImportError as exc:                        # pragma: no cover
        raise ImportError('frames 需要 numpy：pip install -e ".[analysis]"') from exc
    return np


def estimate_transform(pos_a: Sequence, pos_b: Sequence,
                       allow_scale: bool = False,
                       min_samples: int = 10) -> Transform:
    """Kabsch/Umeyama：从两条同步轨迹解出 ``p_a = R @ p_b + t``。

    :param pos_a, pos_b: 形如 (N,3) 的同步位置序列（同一时刻一一对应）
    :param allow_scale: 是否同时解尺度。两台 tracker 尺度本应相同，
        打开只用于诊断 —— 解出的尺度明显偏离 1 说明数据配错了。
    :param min_samples: 最少样本数。连续轨迹拟合用默认的 10；
        **静止点标定**（每个位置停一段、取均值）只需 4 个点即可，
        传 ``min_samples=4``。数学上 3 个非共线点就能定解，但 4 个才能
        留一交叉验证。

    ⚠️ 点必须**非共面**才能定住全部 6 个自由度。全在一条直线或一个平面上时
    :attr:`Transform.rms_m` 会很小但解是病态的 —— 用 :func:`condition_number`
    检查。
    """
    np = _np()
    A = np.asarray(pos_a, dtype=float)
    B = np.asarray(pos_b, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("pos_a / pos_b 需要形状相同的 (N,3) 数组")
    if len(A) < min_samples:
        raise ValueError(f"样本太少（{len(A)} < {min_samples}）")
    if len(A) < 3:
        raise ValueError("至少需要 3 个点")

    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    # 平移量太小时问题病态 —— 轨迹几乎退化成一个点
    spread = float(np.sqrt((A0 ** 2).sum(axis=1)).mean())
    if spread < 0.02:
        raise ValueError(f"轨迹范围只有 {spread*100:.1f} cm，太小 —— 标定时要真的平移起来")

    H = B0.T @ A0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))            # 防止解出镜像
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    scale = 1.0
    if allow_scale:
        var_b = float((B0 ** 2).sum() / len(B0))
        scale = float((S * np.array([1.0, 1.0, d])).sum() / len(B0) / var_b)

    t = ca - scale * (R @ cb)
    resid = A - (scale * (R @ B.T).T + t)
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return Transform(R=R, t=t, rms_m=rms, samples=len(A), scale=scale)


def average_quaternion(quats) -> "object":
    """一组四元数求平均（先统一符号再归一化）。

    四元数 q 和 -q 表示同一旋转，直接平均会互相抵消，所以要先对齐符号。
    静止采样时姿态聚集得很紧，这个简单做法足够；姿态分散时应改用
    协方差矩阵最大特征向量法。
    """
    np = _np()
    Q = np.asarray(quats, dtype=float)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    ref = Q[0]
    Q = Q * np.sign(np.sum(Q * ref, axis=1))[:, None]   # 统一符号
    m = Q.mean(axis=0)
    return m / np.linalg.norm(m)


def quat_multiply(a, b):
    """四元数乘法，(x, y, z, w) 顺序。"""
    np = _np()
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw,
                     aw*bw - ax*bx - ay*by - az*bz])


def quat_conjugate(q):
    np = _np()
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_to_matrix(q):
    np = _np()
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def decompose_yaw_tilt(q, up_axis: int = 1):
    """把旋转拆成「绕世界竖直轴的偏航」和「倾斜」两部分。

    用标准 swing-twist 分解，约定 ``q = q_yaw ⊗ q_tilt`` ——
    偏航在**世界系**里施加（左乘），倾斜是剩下的部分。

    为什么重要：设备的世界系是**重力对齐**的，但**偏航每次开机都不同**
    （地图每次上电现场重建，原点朝向取决于建图那一刻的水平朝向）。所以：

    - **倾斜部分跨会话可复现** —— 由重力定死，这才是真正的安装偏置
    - **偏航部分只在本次会话有效** —— 每次上电要重新归零

    :returns: ``(yaw_deg, q_tilt)``，满足 ``q = q_yaw ⊗ q_tilt``
    """
    np = _np()
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    # twist：q 中绕 up 轴的分量
    twist = np.zeros(4)
    twist[up_axis] = q[up_axis]
    twist[3] = q[3]
    n = np.linalg.norm(twist)
    if n < 1e-9:                       # 退化：旋转 180° 且完全垂直于 up
        twist = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        twist = twist / n
    q_tilt = quat_multiply(quat_conjugate(twist), q)     # q = twist ⊗ q_tilt
    yaw = math.degrees(2.0 * math.atan2(twist[up_axis], twist[3]))
    if yaw > 180.0:
        yaw -= 360.0
    elif yaw < -180.0:
        yaw += 360.0
    return yaw, q_tilt


# ---------------------------------------------------------------- 约定转换
#
# 设备的世界系是 **+Y 朝上**（重力对齐，实测确认）。而机器人 / IMU 的常规约定
# 是 **+Z 朝上**（REP-103）。两者差一个绕 X 轴 +90° 的旋转：
#
#     (x, y, z)  ->  (x, -z, y)
#
# 换算后，设备平放时本体 Z 轴指向新世界 +Z，与普通 IMU 一致。

_S45 = math.sin(math.radians(45.0))

#: Y 朝上 -> Z 朝上 的四元数，(x, y, z, w) 顺序
Q_Z_UP_FROM_Y_UP = (_S45, 0.0, 0.0, _S45)


def to_z_up_position(pos):
    """位置：设备的 Y 朝上世界系 -> Z 朝上世界系。"""
    np = _np()
    p = np.asarray(pos, dtype=float)
    if p.ndim == 1:
        return np.array([p[0], -p[2], p[1]])
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def to_z_up_quat(q):
    """姿态：设备的 Y 朝上世界系 -> Z 朝上世界系（左乘约定旋转）。"""
    return quat_multiply(Q_Z_UP_FROM_Y_UP, q)


def to_z_up_matrix(R):
    """旋转矩阵版本。"""
    np = _np()
    C = quat_to_matrix(np.asarray(Q_Z_UP_FROM_Y_UP, dtype=float))
    return C @ np.asarray(R, dtype=float)


def parse_axis_remap(spec: str):
    """把 ``"Z,-X,-Y"`` 这样的轴重映射规格解析成旋转矩阵。

    规格的三项依次是**新的 X / Y / Z 轴在旧机体系里指向哪个轴**。
    例如 ``"Z,-X,-Y"`` 表示：新 X = 旧 Z，新 Y = −旧 X，新 Z = −旧 Y。

    返回 ``R``，列向量是新轴在旧系中的表示，满足 ``v_old = R @ v_new``。
    姿态换算用 ``R_world_new = R_world_old @ R``。

    会检查结果是右手系（det = +1），左手系（轴选错导致镜像）直接报错。
    """
    np = _np()
    parts = [t.strip().upper() for t in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("规格需要三项，如 'Z,-X,-Y'")
    cols = []
    for t in parts:
        sign = -1.0 if t.startswith("-") else 1.0
        name = t.lstrip("+-")
        if name not in ("X", "Y", "Z"):
            raise ValueError(f"看不懂的轴 {t!r}，只能是 X/Y/Z 带可选正负号")
        v = np.zeros(3)
        v["XYZ".index(name)] = sign
        cols.append(v)
    R = np.stack(cols, axis=1)
    if abs(np.linalg.det(R) - 1.0) > 1e-9:
        raise ValueError(f"{spec!r} 不是右手系（det = {np.linalg.det(R):+.0f}）—— "
                         "检查正负号，或调换两个轴")
    return R


def all_axis_remaps() -> list:
    """列出全部 24 种右手系轴置换的规格字符串，如 ``"Z,-X,-Y"``。

    CAD 坐标系与位姿本体系通常只差一个轴置换（两边都是轴对齐建模的），
    所以对齐问题的解一定在这 24 个里面 —— 穷举比拖滑块试快得多。
    """
    import itertools
    np = _np()
    axes = ["X", "-X", "Y", "-Y", "Z", "-Z"]

    def vec(t):
        v = np.zeros(3)
        v["XYZ".index(t.lstrip("-"))] = -1.0 if t.startswith("-") else 1.0
        return v

    out = []
    for combo in itertools.permutations(axes, 3):
        if len({t.lstrip("-") for t in combo}) != 3:
            continue
        R = np.stack([vec(t) for t in combo], axis=1)
        if abs(np.linalg.det(R) - 1.0) < 1e-9:
            out.append(",".join(combo))
    return out


def remap_body_quat(q, remap):
    """把姿态换到重映射后的机体系。

    ``remap`` 是 :func:`parse_axis_remap` 的矩阵。机体系换轴是**右乘**：
    ``R_world_new = R_world_old @ remap``。位置不受影响 —— 位置表达在世界系里，
    换机体系的轴不改变设备在空间中的位置。
    """
    np = _np()
    return quat_multiply(np.asarray(q, dtype=float), matrix_to_quat(remap))


def matrix_to_quat(R):
    """旋转矩阵 -> 四元数 (x, y, z, w)。"""
    np = _np()
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
             (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
             (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def condition_number(points) -> float:
    """点云的几何条件数：最大主轴长度 / 最小主轴长度。

    静止点标定时用来判断点摆得够不够开。全在一条线上 -> inf；
    全在一个平面上 -> 很大。经验上 **< 10 才算健康**。
    """
    np = _np()
    P = np.asarray(points, dtype=float)
    P = P - P.mean(axis=0)
    sv = np.linalg.svd(P, compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")


def rotation_axis_angle(R) -> tuple["object", float]:
    """把旋转矩阵拆成 (单位转轴, 角度弧度)。用来检验是不是纯偏航。"""
    np = _np()
    R = np.asarray(R, dtype=float)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:                                       # 角度接近 pi，退化
        w, v = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = v[:, int(np.argmax(w))]
    else:
        axis = axis / n
    return axis, angle


def is_yaw_only(R, up_axis: int = 1, tol_deg: float = 5.0) -> tuple[bool, float]:
    """判断旋转是否只绕竖直轴（即两个世界系都重力对齐）。

    :param up_axis: 哪个世界轴朝上。本设备实测是 Y（索引 1）。
    :returns: (是否, 转轴偏离竖直的角度/度)
    """
    np = _np()
    axis, angle = rotation_axis_angle(R)
    up = np.zeros(3)
    up[up_axis] = 1.0
    dev = math.degrees(math.acos(min(1.0, abs(float(np.dot(axis, up))))))
    if angle < math.radians(1.0):                      # 几乎没转，无从判断
        return True, 0.0
    return dev <= tol_deg, dev


def estimate_yaw_transform(pos_a: Sequence, pos_b: Sequence,
                           up_axis: int = 1) -> Transform:
    """只解偏航 + 平移（4 自由度）。两系都重力对齐时用这个，比全 6 自由度更稳。"""
    np = _np()
    A = np.asarray(pos_a, dtype=float)
    B = np.asarray(pos_b, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("pos_a / pos_b 需要形状相同的 (N,3) 数组")
    horiz = [i for i in range(3) if i != up_axis]
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    # 水平面内的最优旋转有闭式解
    u, v = horiz
    num = float((A0[:, v] * B0[:, u] - A0[:, u] * B0[:, v]).sum())
    den = float((A0[:, u] * B0[:, u] + A0[:, v] * B0[:, v]).sum())
    theta = math.atan2(num, den)
    c, s = math.cos(theta), math.sin(theta)
    R = np.eye(3)
    R[u, u], R[u, v] = c, -s
    R[v, u], R[v, v] = s, c
    t = ca - R @ cb
    resid = A - ((R @ B.T).T + t)
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    return Transform(R=R, t=t, rms_m=rms, samples=len(A), scale=1.0)


def resample_to(t_ref: Sequence, t_src: Sequence, pos_src: Sequence):
    """把 B 的轨迹插值到 A 的时间戳上（两台 tracker 帧不对齐时必需）。

    ``t_ref`` 落在 ``t_src`` 范围之外的位置返回 **NaN** —— 不做外推。
    ``np.interp`` 默认会把边界值 clamp 出去，那会把外推值当成真实数据混进标定，
    用 :func:`common_window` 或直接丢掉 NaN 行。
    """
    np = _np()
    t_ref = np.asarray(t_ref, dtype=float)
    t_src = np.asarray(t_src, dtype=float)
    pos_src = np.asarray(pos_src, dtype=float)
    out = np.stack([np.interp(t_ref, t_src, pos_src[:, i]) for i in range(3)], axis=1)
    outside = (t_ref < t_src[0]) | (t_ref > t_src[-1])
    out[outside] = np.nan
    return out


def common_window(t_a: Sequence, pos_a: Sequence, t_b: Sequence, pos_b: Sequence):
    """把两台 tracker 的轨迹对齐到公共时间窗，返回 ``(pos_a, pos_b)``，行一一对应。

    以 A 的时间戳为基准，把 B 插值过来，并丢掉重叠区之外的样本。
    """
    np = _np()
    t_a = np.asarray(t_a, dtype=float)
    pos_a = np.asarray(pos_a, dtype=float)
    resampled = resample_to(t_a, t_b, pos_b)
    keep = ~np.isnan(resampled).any(axis=1)
    if keep.sum() < 10:
        raise ValueError("两台 tracker 的时间窗重叠太少")
    return pos_a[keep], resampled[keep]


class Pivot(NamedTuple):
    """绕固定支点转出来的杆臂解。

    上报的位置是设备自己那个内部参考点，不是你关心的那个点。两者差一个常向量
    ``r_body``（机体系下）。机器人原地转身时这个差直接变成里程计误差：转 180°
    会凭空多出 ``2·|r|`` 的位移。
    """

    r_body: "object"     #: 杆臂，机体系，米。支点位置 = p + R·r_body
    c_world: "object"    #: 支点在世界系的位置
    rms_m: float         #: 拟合残差 RMS。支点没固定住、或者姿态读数有误差就会大
    samples: int
    condition: float     #: 设计矩阵条件数。姿态转得不够散会爆，见下
    spread_deg: float    #: 姿态相对均值的最大夹角，反映转得够不够开


def estimate_lever_arm(positions: Sequence, quats: Sequence,
                       min_samples: int = 50) -> Pivot:
    """把设备绕一个**固定的物理支点**转，解出那个支点在机体系里的位置。

    body 系下位于 ``r`` 的点，其世界坐标是 ``p + R·r``。支点被按住不动，所以对
    每一帧都有::

        p_i + R_i · r = c          （c = 支点的世界坐标，也是未知量）

    对 ``(r, c)`` 是**线性**的，直接最小二乘。每帧给 3 个方程、共 6 个未知量。

    **姿态必须转得散。** 只绕一个轴转的话，``r`` 沿该轴的分量恒等地进了 ``c``，
    根本观测不到 —— 这时 :attr:`Pivot.condition` 会爆掉。所以要绕不同方向都转，
    别只是原地打转。
    """
    np = _np()
    P = np.asarray(positions, dtype=float)
    Q = np.asarray(quats, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("positions 要是 (N,3)")
    if len(P) != len(Q):
        raise ValueError(f"位置 {len(P)} 帧、姿态 {len(Q)} 帧，对不上")
    if len(P) < min_samples:
        raise ValueError(f"只有 {len(P)} 帧，少于 {min_samples}，解不可靠")

    n = len(P)
    R = np.stack([quat_to_matrix(q) for q in Q])          # (n,3,3)
    A = np.empty((3 * n, 6))
    A[:, :3] = R.reshape(3 * n, 3)
    A[:, 3:] = -np.tile(np.eye(3), (n, 1))
    b = -P.reshape(3 * n)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    r, c = x[:3], x[3:]
    resid = P + np.einsum("nij,j->ni", R, r) - c
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))

    q_mean = average_quaternion(Q)
    dots = np.abs(Q @ q_mean / np.linalg.norm(Q, axis=1))
    spread = float(np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))).max())
    return Pivot(r_body=r, c_world=c, rms_m=rms, samples=n,
                 condition=float(np.linalg.cond(A)), spread_deg=spread)


def apply_lever_arm(positions, quats, r_body):
    """把位置从设备参考点搬到杆臂末端：``p' = p + R·r``。"""
    np = _np()
    P = np.asarray(positions, dtype=float)
    r = np.asarray(r_body, dtype=float)
    if P.ndim == 1:
        return P + quat_to_matrix(quats) @ r
    R = np.stack([quat_to_matrix(q) for q in np.asarray(quats, dtype=float)])
    return P + np.einsum("nij,j->ni", R, r)
