"""四元数 / 欧拉角 / 旋转矩阵。纯标准库，结果与 scipy 逐位一致（见 tests）。"""

from __future__ import annotations

import math

Quat = tuple[float, float, float, float]   # (x, y, z, w)
Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


def normalize(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q) if n > 1e-9 else IDENTITY  # type: ignore[return-value]


def norm(q: Quat) -> float:
    return math.sqrt(sum(c * c for c in q))


def conjugate(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def multiply(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def relative_to(reference: Quat, q: Quat) -> Quat:
    """q 相对 reference 的旋转（把 reference 当成零点）。"""
    return normalize(multiply(conjugate(normalize(reference)), normalize(q)))


def to_euler_deg(q: Quat) -> Vec3:
    """ZYX 内旋，返回 (roll绕X, pitch绕Y, yaw绕Z)，单位度。等价 scipy 的 as_euler("xyz")。"""
    x, y, z, w = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))  # type: ignore[return-value]


def to_matrix(q: Quat) -> Mat3:
    """3x3 旋转矩阵。第 i **列** = 本体第 i 轴在世界系下的单位向量。"""
    x, y, z, w = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def body_axes(q: Quat) -> tuple[Vec3, Vec3, Vec3]:
    """本体 X/Y/Z 轴在世界系下的指向。用来实测坐标系映射。"""
    m = to_matrix(q)
    return tuple((m[0][j], m[1][j], m[2][j]) for j in range(3))  # type: ignore[return-value]


def from_axis_angle(axis: Vec3, degrees: float) -> Quat:
    half = math.radians(degrees) / 2.0
    n = math.sqrt(sum(c * c for c in axis)) or 1.0
    s = math.sin(half)
    return (axis[0] / n * s, axis[1] / n * s, axis[2] / n * s, math.cos(half))


def magnitude(v: Vec3) -> float:
    return math.sqrt(sum(c * c for c in v))
