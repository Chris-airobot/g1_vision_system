"""每台 tracker 的运行时统计：速率、丢包、行程、零点。"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

from . import protocol as p
from . import transforms as tf


@dataclass(frozen=True)
class PoseValidity:
    """一帧 pose 的协议有效性和保守里程计有效性。"""

    position_valid: bool
    odometry_valid: bool
    fresh: bool
    pose_finite: bool
    motion_finite: bool
    quaternion_valid: bool
    quaternion_norm: float
    age_ms: float
    reasons: tuple[str, ...]


def evaluate_pose_validity(pose: p.Pose, *, age_s: float,
                           stale_after_s: float = 0.2) -> PoseValidity:
    """解释 Tracker 状态位，并给出适合里程计使用的保守 valid。

    ``position_valid`` 完全遵循设备协议：``OK`` 和 ``RECENTLY_LOST`` 都为真。
    但后者实测会冻结位置，所以 ``odometry_valid`` 只接受新鲜的 ``OK`` pose，
    同时检查位置/四元数数值和四元数范数。加速度/角速度异常会单独显示，但不影响
    位置姿态的 valid。
    """
    age_s = max(0.0, float(age_s))
    pose_values = tuple(pose.pos) + tuple(pose.rot)
    motion_values = tuple(pose.acc) + tuple(pose.rot_vel)
    pose_finite = all(math.isfinite(v) for v in pose_values)
    motion_finite = all(math.isfinite(v) for v in motion_values)
    quaternion_norm = math.sqrt(sum(v * v for v in pose.rot)) \
        if all(math.isfinite(v) for v in pose.rot) else float("nan")
    quaternion_valid = math.isfinite(quaternion_norm) and 0.9 <= quaternion_norm <= 1.1
    fresh = age_s <= stale_after_s

    reasons: list[str] = []
    if pose.status != p.POSE_OK:
        reasons.append(f"status={pose.status_name}")
    if not fresh:
        reasons.append(f"数据超时 {age_s * 1000:.1f} ms")
    if not pose_finite:
        reasons.append("位置/四元数含 NaN 或 Inf")
    if not quaternion_valid:
        reasons.append(f"四元数范数异常 {quaternion_norm:.4f}")
    odometry_valid = (pose.status == p.POSE_OK and fresh and pose_finite
                      and quaternion_valid)
    return PoseValidity(
        position_valid=pose.position_valid,
        odometry_valid=odometry_valid,
        fresh=fresh,
        pose_finite=pose_finite,
        motion_finite=motion_finite,
        quaternion_valid=quaternion_valid,
        quaternion_norm=quaternion_norm,
        age_ms=age_s * 1000.0,
        reasons=tuple(reasons),
    )


class TrackerState:
    """累计一台 tracker 的状态。喂 :class:`~.protocol.Pose` 进来即可。"""

    def __init__(self, index: int, mac: bytes, rate_window: int = 200) -> None:
        self.index = index
        self.mac = mac
        self.mac_str = p.mac_str(mac)
        self.pose: p.Pose | None = None
        self.count = 0
        self.lost = 0
        self.duplicates = 0
        self.stalls = 0
        self.stalled_ticks = 0
        self._prev_idx: int | None = None
        self._prev_device_time: int | None = None
        self._stamps: deque[float] = deque(maxlen=rate_window)
        self.zero_pos: tuple[float, float, float] | None = None
        self.zero_quat: tf.Quat | None = None
        self.lo = [math.inf] * 3
        self.hi = [-math.inf] * 3
        self.path_length = 0.0
        self._prev_pos: tuple[float, float, float] | None = None
        self.first_seen = time.monotonic()
        self.last_seen = self.first_seen
        self.last_pkt_idx: int | None = None

    # ------------------------------------------------------------- 更新

    def update(self, pose: p.Pose, *, pkt_idx: int | None = None,
               recv_monotonic: float | None = None) -> None:
        self.pose = pose
        self.count += 1
        now = time.monotonic() if recv_monotonic is None else float(recv_monotonic)
        self._stamps.append(now)
        self.last_seen = now
        if pkt_idx is not None:
            self.last_pkt_idx = int(pkt_idx)

        if self._prev_idx is not None:
            gap = p.idx_gap(self._prev_idx, pose.idx)
            if gap > 0:
                self.lost += gap
            elif gap == -1:
                # dongle 偶发重复投递同一帧（整包除主机时间戳外完全相同）。
                # 不计入有效帧数。
                self.duplicates += 1
            elif gap == 0 and self._prev_device_time is not None:
                # idx 连续但设备时间跳了好几个帧周期 = 设备侧停顿。
                # 基于 idx 的丢包统计看不见这种缺口。
                dt = (pose.device_time - self._prev_device_time) & 0xFFFF
                if dt > p.DEVICE_STALL_TICKS:
                    self.stalls += 1
                    self.stalled_ticks += dt
        self._prev_idx = pose.idx
        self._prev_device_time = pose.device_time

        for i in range(3):
            self.lo[i] = min(self.lo[i], pose.pos[i])
            self.hi[i] = max(self.hi[i], pose.pos[i])
        if self._prev_pos is not None:
            self.path_length += math.dist(pose.pos, self._prev_pos)
        self._prev_pos = pose.pos

    # ------------------------------------------------------------- 查询

    @property
    def hz(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    @property
    def loss_percent(self) -> float:
        total = self.count + self.lost
        return 100.0 * self.lost / total if total else 0.0

    @property
    def stalled_seconds(self) -> float:
        """设备侧停顿累计时长（秒）。"""
        return self.stalled_ticks * p.DEVICE_TIME_TICK_SECONDS

    @property
    def uptime(self) -> float:
        return time.monotonic() - self.first_seen

    def pose_age_s(self, now: float | None = None) -> float:
        """最新 pose 距当前主机单调时钟的时间。"""
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.last_seen)

    def validity(self, *, stale_after_s: float = 0.2,
                 now: float | None = None) -> PoseValidity | None:
        """返回当前 pose 的两层 valid；尚未收到 pose 时返回 ``None``。"""
        if self.pose is None:
            return None
        return evaluate_pose_validity(
            self.pose, age_s=self.pose_age_s(now), stale_after_s=stale_after_s)

    @property
    def relative_pos(self) -> tuple[float, float, float]:
        if self.pose is None:
            return (0.0, 0.0, 0.0)
        if self.zero_pos is None:
            return self.pose.pos
        return tuple(self.pose.pos[i] - self.zero_pos[i] for i in range(3))  # type: ignore[return-value]

    @property
    def relative_quat(self) -> tf.Quat:
        if self.pose is None:
            return tf.IDENTITY
        q = tf.normalize(self.pose.rot)
        return q if self.zero_quat is None else tf.relative_to(self.zero_quat, q)

    def span(self, axis: int) -> float:
        if self.hi[axis] <= self.lo[axis]:
            return 0.0
        return self.hi[axis] - self.lo[axis]

    # ------------------------------------------------------------- 控制

    def set_zero(self) -> None:
        if self.pose is not None:
            self.zero_pos = self.pose.pos
            self.zero_quat = tf.normalize(self.pose.rot)

    def clear_zero(self) -> None:
        self.zero_pos = None
        self.zero_quat = None

    def reset_stats(self) -> None:
        self.lo = [math.inf] * 3
        self.hi = [-math.inf] * 3
        self.path_length = 0.0
        self._prev_pos = None
        self.count = 0
        self.lost = 0
        self.duplicates = 0
        self.stalls = 0
        self.stalled_ticks = 0
        self._prev_idx = None
        self._prev_device_time = None
        self.first_seen = time.monotonic()
