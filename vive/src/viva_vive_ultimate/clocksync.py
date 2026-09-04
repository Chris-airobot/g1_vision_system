"""把 tracker 的 device_time 映射到主机时钟，用于和别的传感器（相机等）对齐。

问题
----
`device_time` 是 u16、10 µs 一 tick，**每 655.36 ms 回绕一次**，而且是设备自己的
晶振，与主机时钟有 ppm 级漂移。主机侧的到达时间又被 RF + USB 排队延迟污染。

做法
----
1. 把 u16 展开成单调递增的 int64 tick。
2. 用**单边稳健拟合**求 ``host_ns ≈ slope * ticks + offset``。

   关键点：到达延迟 ``d_i >= 0`` 恒成立 —— 包只会晚到，不会早到。所以样本点全部
   落在真实直线**上方**，普通最小二乘会被排队延迟系统性抬高。这里用分位数
   IRLS 反复只保留残差最小的一批点，收敛到下包络。

   （WiFi 方案的 ``PeerTimeMapper`` 用 EMA 拟合 ns_per_tick，就有这个偏差问题。）

3. ``align_to_floor`` 打开时再把截距下移到最小残差处，扣掉排队延迟，
   只剩设备内部那个**常量**硬件延迟 —— 那一项软件测不出来，要靠互相关标定。

用法
----
    sync = ClockSync()
    for report, pose in dongle.stream():
        sync.add(pose.device_time, time.monotonic_ns())
        if sync.ready:
            t_ns = sync.host_ns(pose.device_time)   # 该帧在主机时钟下的时刻
"""

from __future__ import annotations

import time
from collections import deque
from typing import NamedTuple

TICK_NS_NOMINAL = 10_000.0        #: 标称 10 µs/tick，实测 10.001
WRAP = 1 << 16
HALF_WRAP = WRAP // 2

#: 相邻样本间隔超过这个值就认为 u16 可能已经绕过一整圈，展开不再可信
MAX_SAFE_GAP_NS = 400_000_000     # 655.36 ms 的 ~60%


class Fit(NamedTuple):
    """一次拟合的结果。"""

    slope_ns_per_tick: float
    offset_ns: float
    samples: int
    span_s: float
    residual_p50_ns: float
    residual_p95_ns: float

    @property
    def ppm(self) -> float:
        """设备晶振相对标称 10 µs 的偏差，单位 ppm。"""
        return (self.slope_ns_per_tick / TICK_NS_NOMINAL - 1.0) * 1e6


class ClockSync:
    """device_time -> 主机时钟的在线估计。

    :param window_s: 参与拟合的时间窗，越长斜率越准、跟漂移越慢
    :param min_samples: 出结果所需的最少样本
    :param keep_quantile: 每轮 IRLS 保留残差最小的这个比例
    :param iterations: IRLS 轮数
    :param align_to_floor: 是否把截距压到下包络（扣掉排队延迟）
    """

    def __init__(self, window_s: float = 30.0, min_samples: int = 200,
                 keep_quantile: float = 0.2, iterations: int = 3,
                 align_to_floor: bool = True) -> None:
        self.window_ns = int(window_s * 1e9)
        self.min_samples = min_samples
        self.keep_quantile = keep_quantile
        self.iterations = iterations
        self.align_to_floor = align_to_floor

        self._samples: deque[tuple[int, int]] = deque()   # (ticks, host_ns)
        self._last_u16: int | None = None
        self._last_host_ns: int | None = None
        self._ticks = 0
        self._fit: Fit | None = None
        self._dirty = True
        self.wraps = 0
        self.resets = 0

    # ------------------------------------------------------------ 展开

    def unwrap(self, device_time: int) -> int:
        """u16 -> 单调递增的 int64 tick。"""
        value = device_time & 0xFFFF
        if self._last_u16 is None:
            self._last_u16 = value
            self._ticks = value
            return self._ticks
        delta = (value - self._last_u16) & 0xFFFF
        if delta > HALF_WRAP:          # 负向跳变 = 乱序，按倒退处理
            delta -= WRAP
        if value < self._last_u16 and delta > 0:
            self.wraps += 1
        self._ticks += delta
        self._last_u16 = value
        return self._ticks

    # ------------------------------------------------------------ 采样

    def add(self, device_time: int, host_ns: int | None = None) -> int:
        """喂一帧。返回展开后的 tick。"""
        if host_ns is None:
            host_ns = time.monotonic_ns()

        # 断流太久就重来：u16 可能绕过整圈，展开不可信
        if self._last_host_ns is not None and host_ns - self._last_host_ns > MAX_SAFE_GAP_NS:
            self.reset()
            self.resets += 1
        self._last_host_ns = host_ns

        ticks = self.unwrap(device_time)
        self._samples.append((ticks, host_ns))
        cutoff = host_ns - self.window_ns
        while self._samples and self._samples[0][1] < cutoff:
            self._samples.popleft()
        self._dirty = True
        return ticks

    def reset(self) -> None:
        self._samples.clear()
        self._last_u16 = None
        self._ticks = 0
        self._fit = None
        self._dirty = True

    # ------------------------------------------------------------ 拟合

    @property
    def ready(self) -> bool:
        return self.fit is not None

    @property
    def fit(self) -> Fit | None:
        if self._dirty:
            self._fit = self._compute_fit()
            self._dirty = False
        return self._fit

    def _compute_fit(self) -> Fit | None:
        n = len(self._samples)
        if n < self.min_samples:
            return None
        pts = list(self._samples)
        # 以窗口首个样本为原点，避免大数相减丢精度
        t0, h0 = pts[0]
        xs = [float(t - t0) for t, _ in pts]
        ys = [float(h - h0) for _, h in pts]

        slope, intercept = _least_squares(xs, ys)
        keep = xs, ys
        for _ in range(self.iterations):
            res = [y - (slope * x + intercept) for x, y in zip(*keep)]
            k = max(2, int(len(res) * self.keep_quantile))
            idx = sorted(range(len(res)), key=lambda i: res[i])[:k]
            keep = ([keep[0][i] for i in idx], [keep[1][i] for i in idx])
            slope, intercept = _least_squares(*keep)

        residuals = sorted(y - (slope * x + intercept) for x, y in zip(xs, ys))
        if self.align_to_floor:
            floor = residuals[0]
            intercept += floor
            residuals = [r - floor for r in residuals]

        # 换回绝对坐标：host = slope * (ticks - t0) + intercept + h0
        offset_ns = h0 + intercept - slope * t0
        span_s = (pts[-1][1] - pts[0][1]) / 1e9
        return Fit(
            slope_ns_per_tick=slope,
            offset_ns=offset_ns,
            samples=n,
            span_s=span_s,
            residual_p50_ns=residuals[len(residuals) // 2],
            residual_p95_ns=residuals[min(len(residuals) - 1, int(len(residuals) * 0.95))],
        )

    # ------------------------------------------------------------ 换算

    def host_ns(self, device_time: int | None = None, ticks: int | None = None) -> float | None:
        """把一帧的 device_time（或已展开的 tick）换成主机时钟纳秒。

        传 ``device_time`` 时按当前展开状态就近解释，只对刚喂进来的那一帧有效；
        批量离线换算请传 ``ticks``。
        """
        f = self.fit
        if f is None:
            return None
        if ticks is None:
            if device_time is None:
                return None
            ticks = self._nearest_ticks(device_time)
        return f.slope_ns_per_tick * ticks + f.offset_ns

    def _nearest_ticks(self, device_time: int) -> int:
        """把 u16 解释成离当前展开值最近的那个 tick。"""
        base = self._ticks
        candidate = (base & ~0xFFFF) | (device_time & 0xFFFF)
        for adjusted in (candidate, candidate + WRAP, candidate - WRAP):
            if abs(adjusted - base) <= HALF_WRAP:
                return adjusted
        return candidate


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return TICK_NS_NOMINAL, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return TICK_NS_NOMINAL, my - TICK_NS_NOMINAL * mx
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def unwrap_series(device_times: list[int]) -> list[int]:
    """离线批量展开一整列 u16 device_time。"""
    out: list[int] = []
    ticks = 0
    last: int | None = None
    for v in device_times:
        v &= 0xFFFF
        if last is None:
            ticks = v
        else:
            d = (v - last) & 0xFFFF
            if d > HALF_WRAP:
                d -= WRAP
            ticks += d
        last = v
        out.append(ticks)
    return out


def fit_offline(ticks: list[int], host_ns: list[int], keep_quantile: float = 0.2,
                iterations: int = 3, align_to_floor: bool = True) -> Fit:
    """对一整段离线数据做同样的单边稳健拟合。"""
    sync = ClockSync(window_s=1e9, min_samples=2, keep_quantile=keep_quantile,
                     iterations=iterations, align_to_floor=align_to_floor)
    sync._samples.extend(zip(ticks, host_ns))
    sync._dirty = True
    f = sync.fit
    assert f is not None
    return f
