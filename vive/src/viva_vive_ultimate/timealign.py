"""用互相关估计两个传感器之间的**常量时间偏移**。

为什么需要
----------
:mod:`~viva_vive_ultimate.clocksync` 能把 tracker 的 ``device_time`` 对齐到主机时钟，
但它对齐的是**包到达的下包络**。设备内部从「相机曝光 / IMU 采样」到「RF 发包」
之间还有一段固定延迟，这一段**软件观测不到** —— 无论拟合多准都会整体偏这么多。

唯一的办法是物理标定：把 tracker 和相机**刚性绑在一起**同时晃动，两边都会看到
同一段角速度曲线，互相关的峰值位置就是那个常量偏移。

（这和 Kalibr 估计 IMU-相机时间偏移是同一个思路。）

用法
----
    from viva_vive_ultimate.timealign import estimate_offset

    r = estimate_offset(utk_t, utk_gyro_mag, cam_t, cam_gyro_mag)
    print(r.offset_s, r.correlation)
    # utk 的时间戳加上 offset_s 之后与相机对齐

需要 numpy：``pip install -e ".[analysis]"``
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


class AlignResult(NamedTuple):
    """互相关标定结果。"""

    offset_s: float          #: 把 a 的时间戳加上它才与 b 对齐；正数表示 a 早于 b
    correlation: float       #: 峰值处的归一化相关，越接近 1 越可信
    rate_hz: float           #: 重采样栅格
    samples: int             #: 参与相关的样本数
    overlap_s: float         #: 两路时间轴的重叠时长
    search_s: float          #: 搜索范围
    contrast: float          #: 峰值 / 相关函数典型幅度。仅供参考，不作判据 ——
                             #: 实测它区分不出好坏（静止时反而最高）

    @property
    def offset_ms(self) -> float:
        return self.offset_s * 1000.0

    @property
    def trustworthy(self) -> bool:
        """归一化相关够高才算初步可信。

        **这只是必要条件。** 周期性运动（匀速画圈、规律摆动）会让相关函数出现
        等高的次峰，单次估计可能落在错误的周期上。最终一定要用
        :func:`split_half_check` 验证。
        """
        return self.correlation > 0.5


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "timealign 需要 numpy：pip install -e \".[analysis]\""
        ) from exc
    return np


def resample(t: Sequence[float], v: Sequence[float], grid):
    """把不等间隔采样线性插值到统一栅格。"""
    np = _require_numpy()
    return np.interp(grid, np.asarray(t, dtype=float), np.asarray(v, dtype=float))


def estimate_offset(
    t_a: Sequence[float],
    v_a: Sequence[float],
    t_b: Sequence[float],
    v_b: Sequence[float],
    rate_hz: float = 200.0,
    search_s: float = 0.25,
    detrend: bool = True,
) -> AlignResult:
    """互相关求 a 相对 b 的常量时间偏移。

    :param t_a, v_a: A 路（如 tracker）的时间戳（秒）与标量信号（如角速度模长）
    :param t_b, v_b: B 路（如 D435i 陀螺）同上
    :param rate_hz: 重采样栅格，取两路较高采样率即可
    :param search_s: 搜索范围 ±秒
    :param detrend: 去均值并归一化方差（强烈建议开）

    信号用**角速度模长**最稳 —— 它与两个传感器各自的坐标系无关，
    不需要先标定外参。
    """
    np = _require_numpy()
    t_a = np.asarray(t_a, dtype=float)
    t_b = np.asarray(t_b, dtype=float)
    if len(t_a) < 4 or len(t_b) < 4:
        raise ValueError("样本太少")

    lo = max(t_a[0], t_b[0])
    hi = min(t_a[-1], t_b[-1])
    overlap = hi - lo
    if overlap <= 2 * search_s:
        raise ValueError(f"两路重叠只有 {overlap:.2f}s，不足以在 ±{search_s}s 内搜索")

    grid = np.arange(lo, hi, 1.0 / rate_hz)
    a = resample(t_a, v_a, grid)
    b = resample(t_b, v_b, grid)

    if detrend:
        a = a - a.mean()
        b = b - b.mean()
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            raise ValueError("信号几乎没有变化 —— 标定时要真的晃起来")
        a /= sa
        b /= sb

    max_lag = int(round(search_s * rate_hz))
    lags = np.arange(-max_lag, max_lag + 1)
    n = len(grid)
    corr = np.empty(len(lags), dtype=float)
    for i, lag in enumerate(lags):
        if lag >= 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: n + lag]
        corr[i] = float(np.dot(x, y) / len(x)) if len(x) else -1.0

    k = int(np.argmax(corr))
    peak = float(corr[k])

    # 抛物线插值取亚采样精度
    frac = 0.0
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            frac = 0.5 * (y0 - y2) / denom

    # 对比度：峰值相对相关函数的典型幅度。用中位绝对值而不是次峰 ——
    # 周期性运动的次峰本来就高，那是真实歧义，应该由 split_half_check 去抓。
    typical = float(np.median(np.abs(corr)))
    contrast = peak / typical if typical > 1e-9 else float("inf")

    lag_samples = lags[k] + frac
    # b[lag:] 对 a[:-lag] 相关最大 => b 比 a 晚 lag 个格点 => a 要加上 lag 才对齐 b
    offset_s = float(lag_samples / rate_hz)

    return AlignResult(
        offset_s=offset_s,
        correlation=peak,
        rate_hz=rate_hz,
        samples=n,
        overlap_s=float(overlap),
        search_s=search_s,
        contrast=contrast,
    )


class SplitHalfResult(NamedTuple):
    """分半一致性检验。"""

    first: AlignResult
    second: AlignResult
    full: AlignResult
    spread_ms: float         #: 前后半段估计之差的绝对值

    @property
    def consistent(self) -> bool:
        """前后半段估到同一个偏移，才说明它真的是个常量。"""
        return (self.spread_ms < 2.0 and self.first.trustworthy
                and self.second.trustworthy and self.full.trustworthy)

    def summary(self) -> str:
        tag = "一致 ✓" if self.consistent else "不一致 ✗"
        return (f"前半 {self.first.offset_ms:+.2f} ms  后半 {self.second.offset_ms:+.2f} ms  "
                f"全段 {self.full.offset_ms:+.2f} ms   差 {self.spread_ms:.2f} ms   {tag}")


def split_half_check(t_a, v_a, t_b, v_b, **kwargs) -> SplitHalfResult:
    """把录制切成前后两半各估一次，看结果是否一致。

    这是最有力的检验：真正的常量硬件延迟在前后半段应该完全一样。
    若两半差得远，说明要么运动太规律导致周期歧义，要么根本没对上。
    """
    np = _require_numpy()
    t_a = np.asarray(t_a, dtype=float)
    t_b = np.asarray(t_b, dtype=float)
    mid = (max(t_a[0], t_b[0]) + min(t_a[-1], t_b[-1])) / 2.0

    def part(lo_a, hi_a, lo_b, hi_b):
        ma = (t_a >= lo_a) & (t_a <= hi_a)
        mb = (t_b >= lo_b) & (t_b <= hi_b)
        return estimate_offset(t_a[ma], np.asarray(v_a, float)[ma],
                               t_b[mb], np.asarray(v_b, float)[mb], **kwargs)

    first = part(t_a[0], mid, t_b[0], mid)
    second = part(mid, t_a[-1], mid, t_b[-1])
    full = estimate_offset(t_a, v_a, t_b, v_b, **kwargs)
    return SplitHalfResult(first, second, full,
                           abs(first.offset_ms - second.offset_ms))


def magnitude(x: Sequence[float], y: Sequence[float], z: Sequence[float]):
    """三轴 -> 模长。与坐标系无关，做互相关最省事。"""
    np = _require_numpy()
    return np.sqrt(np.asarray(x, float) ** 2 + np.asarray(y, float) ** 2
                   + np.asarray(z, float) ** 2)


def angular_velocity_from_quaternions(t: Sequence[float], quats) -> "object":
    """从四元数序列差分出角速度模长。

    用来**独立校验**：tracker 上报的 ``rot_vel`` 字段万一相对位姿本身有滤波滞后，
    拿它标定出来的偏移就不能用在位姿上。这个函数直接从位姿算，两者对比即可确认。

    ⚠️ 必须用**中心差分**：后向差分 ``angle(q[i-1], q[i]) / dt`` 代表的是区间中点
    ``t_i - dt/2`` 的角速度，会凭空引入半个采样周期的滞后。UTK 是 125 Hz，
    那就是 4 ms —— 比要测的偏移本身还大。

    :param t: 每帧时间戳（秒）
    :param quats: 形如 (N, 4) 的四元数，顺序 (x, y, z, w)
    """
    np = _require_numpy()
    t = np.asarray(t, dtype=float)
    q = np.asarray(quats, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quats 需要形如 (N, 4) 的数组，顺序 (x, y, z, w)")
    if len(t) != len(q) or len(t) < 3:
        raise ValueError("时间戳与四元数数量不匹配，或样本太少")

    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    # 中心差分：q[i-1] 到 q[i+1] 的夹角，除以 t[i+1]-t[i-1]，代表 t[i] 时刻
    dot = np.abs(np.sum(q[:-2] * q[2:], axis=1)).clip(-1.0, 1.0)
    dt = t[2:] - t[:-2]
    w = np.zeros(len(t))
    valid = dt > 1e-9
    w[1:-1][valid] = 2.0 * np.arccos(dot[valid]) / dt[valid]
    w[0], w[-1] = w[1], w[-2]
    return w
