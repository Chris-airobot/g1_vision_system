"""静置定原点：把一段静止采样压成一个可复用的基准位姿。

**为什么必须做这一步。** 设备上电即现场重建地图，所以世界系的**原点位置和
偏航朝向每次开机都不一样**（俯仰/横滚不会变 —— 世界系由重力定死）。不归零的话
今天的轨迹和昨天的没法比，两台 tracker 的轨迹也对不上。

归零动作就是：把设备摆在参考位置**别动**，采 30 秒，取平均当基准。

**哪一部分能留到下次？** 只看这一件事：那个量是不是由重力定的。

===========  =====================  ==================================
 量           参照物                 换台机器 / 重新上电 / 装到机器人上
===========  =====================  ==================================
 t_w  位置    地图原点               **没了** —— 地图每次上电现场重建
 q_ow 偏航    地图朝向               **没了** —— 同上，而且没有任何绝对方位参照
 q_bb 倾角    **重力**               **留得住** —— 重力不会变
===========  =====================  ==================================

所以「先标定好再装到机器人上」这条路，只有倾角走得通：**台面上量出来的倾角修正
可以一直用下去，位置和偏航必须装好之后每次上电现场定**。偏航尤其没得商量 ——
绕重力那一个自由度，没有磁力计也没有外部地标，物理上就没有绝对参照。

对应两步走::

    vvu-datum --datum-mode level -o body.json      # 台面上做一次，永久有效
    vvu-datum --reuse body.json -o origin.json     # 装好后每次上电做，10 秒够

台面标定要管用，有个前提：**摆在台面上的姿态，要和装到机器人上站直时的姿态一致**
（连支架一起标就容易做到）。第二步会把这个前提**量出来** —— 见 ``residual_tilt_deg``。

五种归零方式，差别在「除了位置，还抹掉姿态的哪一部分」::

    q_out = q_ow ⊗ q_in ⊗ q_bb        p_out = R_ow · (p_in − t_w)

    模式        q_ow            q_bb          静置时的输出   重力对齐  跨会话
    position    单位            单位          原姿态         是        否
    yaw         conj(q_yaw)     单位          剩安装倾角     是        否
    mount       conj(q_yaw)     conj(q_tilt)  单位           是        否   ← 默认
    full        conj(q_mean)    单位          单位           否        否
    level       单位            conj(q_tilt)  只把姿态摆平   是        **是**

``level`` 就是上面说的台面标定：不碰位置也不碰偏航，只留那个由重力定死的倾角修正，
所以它跨上电周期一直有效。

``mount`` 是给 G1 胯部这种场景的：tracker 粘上去必然有个歪角，你既想让它静置时
读数是单位阵（像普通 IMU 那样），又不能把重力对齐丢掉。办法是把旋转拆开 ——
**偏航从世界系那侧左乘抹掉，倾角从机体系那侧右乘抹掉**。世界系那侧只转了偏航，
所以 Z 轴仍然沿重力。前提是归零那一刻机体确实摆在参考姿态（G1 站直）。

``full`` 把整个姿态都塞进世界系那侧，静置时输出也是单位阵，但输出系的 Z 轴变成
了设备当时的朝向，**重力对齐没了**。留着是因为有人就想要这个，用之前想清楚。

核心链路零第三方依赖，离线机器上直接能跑。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

from . import protocol as p
from . import transforms as tf

#: JSON 格式版本。字段含义变了就 +1，读的时候拒绝不认识的版本。
DATUM_FORMAT = 3

MODES = ("position", "yaw", "mount", "full", "level")

#: 只有这个模式产出的基准跨上电周期有效（它只含重力定死的倾角）
SESSION_FREE_MODES = ("level",)
DEFAULT_MODE = "mount"

# --------------------------------------------------------------- 质量门限
#
# 阈值来自实测：这台设备静置 180 s 漂移 < 0.3 mm、每轴 σ ≈ 0.06 mm。
# 下面的硬性门限比实测值宽 1~2 个数量级，能过说明设备真的没动、SLAM 也没飘。
# 警告线取硬性线的 1/4。

MAX_DRIFT_MM = 2.0            #: 前半段均值 vs 后半段均值的距离
MAX_JITTER_MM = 1.0           #: 单轴标准差
MAX_SPAN_MM = 10.0            #: 单轴峰峰值
MAX_TILT_JITTER_DEG = 0.5     #: 姿态相对均值的 RMS 夹角
MIN_RATE_HZ = 100.0           #: 标称 125 Hz，掉太多说明丢包严重
MIN_STATUS_OK_PERCENT = 99.0  #: 出现 LOST / RECENTLY_LOST 一律不收
MIN_SAMPLES = 200
WARN_RATIO = 0.25


class Sample(NamedTuple):
    """静置期间的一帧。位姿已经是**最终输出约定**（z_up / 机体系换轴都做完了）。"""

    t_ns: int
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    status: int


class StaticReport(NamedTuple):
    """这段采样到底静没静。``problems`` 非空就是不该拿来定原点。"""

    samples: int
    seconds: float
    rate_hz: float
    jitter_mm: tuple[float, float, float]
    span_mm: tuple[float, float, float]
    drift_mm: float
    tilt_jitter_deg: float
    status_ok_percent: float
    bad_status: dict
    problems: list
    warnings: list

    @property
    def ok(self) -> bool:
        return not self.problems

    def describe(self) -> str:
        j = "/".join(f"{v:.2f}" for v in self.jitter_mm)
        s = "/".join(f"{v:.2f}" for v in self.span_mm)
        return (f"{self.samples} 帧 {self.seconds:.1f}s {self.rate_hz:.1f}Hz  "
                f"抖动 σ={j} mm  峰峰 {s} mm  半段漂移 {self.drift_mm:.2f} mm  "
                f"姿态抖动 {self.tilt_jitter_deg:.3f}°  OK {self.status_ok_percent:.1f}%")


class Datum(NamedTuple):
    """一台设备的基准位姿。``apply()`` 把世界系位姿换算到原点系。"""

    device_id: str        #: MAC[2:6]，跨会话稳定，用它认设备
    mac: str
    tracker_index: int    #: 槽位号，只在本次会话有效，仅供排查
    mode: str
    z_up: bool            #: 采基准时是否已经换成 +Z 朝上
    body_axes: str        #: 采基准时的机体系换轴规格，"" = 没换
    up_axis: int          #: 世界竖直轴下标：z_up 时是 2，否则 1
    t_w: tuple            #: 世界系平移（米）
    q_ow: tuple           #: 世界系那侧的旋转，左乘
    q_bb: tuple           #: 机体系那侧的旋转，右乘
    yaw_deg: float        #: 抹掉的偏航
    tilt_deg: float       #: 安装倾角（相对重力）
    tilt_axis: tuple      #: 倾角的转轴，世界系下
    lever_body: tuple     #: 杆臂，机体系，米。把上报位置搬到你真正关心的那个点
    session_scoped: bool  #: True = 含地图相关量（位置/偏航），换次上电就作废
    residual_tilt_deg: float  #: --reuse 时：台面标定没对上的那部分。0 表示完美转移
    reused_from: str      #: --reuse 用的文件，留个出处
    samples: int
    seconds: float
    created: str
    report: dict          #: StaticReport 的快照，事后能回查这次归零干不干净

    # ------------------------------------------------------------- 应用

    def apply(self, pos, rot):
        """世界系位姿 -> 原点系位姿。

        输入必须和采基准时是**同一套约定** —— z_up 和机体系换轴都已经做完。
        约定对不上算出来的东西没有意义，所以 :func:`load` 会去核对。

        顺序：先按杆臂把位置搬到你关心的那个点，再平移到原点、转掉偏航。
        """
        rot = tf.normalize(rot)
        if self.lever_body != (0.0, 0.0, 0.0):
            B = tf.to_matrix(rot)                    # 机体系 -> 世界系
            r = self.lever_body
            pos = tuple(pos[i] + sum(B[i][k] * r[k] for k in range(3)) for i in range(3))
        R = tf.to_matrix(self.q_ow)
        d = (pos[0] - self.t_w[0], pos[1] - self.t_w[1], pos[2] - self.t_w[2])
        out_pos = tuple(R[i][0] * d[0] + R[i][1] * d[1] + R[i][2] * d[2] for i in range(3))
        out_rot = tf.multiply(tf.multiply(self.q_ow, rot), self.q_bb)
        return out_pos, out_rot

    # ------------------------------------------------------------- 序列化

    def to_dict(self) -> dict:
        d = self._asdict()
        d["format"] = DATUM_FORMAT
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Datum":
        fmt = d.get("format", 0)
        if fmt != DATUM_FORMAT:
            raise ValueError(f"基准文件版本 {fmt}，本程序只认 {DATUM_FORMAT}，请重新归零")
        fields = {k: d[k] for k in cls._fields}
        for k in ("t_w", "q_ow", "q_bb", "tilt_axis", "lever_body"):
            fields[k] = tuple(fields[k])
        return cls(**fields)

    def summary(self) -> str:
        tail = (f"  机体系 {self.body_axes}" if self.body_axes else "")
        if self.lever_body != (0.0, 0.0, 0.0):
            tail += ("  杆臂 (" + ", ".join(f"{v*1000:+.1f}" for v in self.lever_body)
                     + ") mm")
        if not self.session_scoped:
            return (f"{self.device_id}  模式 {self.mode}  倾角修正 {self.tilt_deg:.2f}°  "
                    f"（不含位置/偏航，跨上电有效）  "
                    f"{'Z' if self.z_up else 'Y'} 朝上" + tail)
        t = ", ".join(f"{v:+.3f}" for v in self.t_w)
        res = (f"  残余倾角 {self.residual_tilt_deg:.2f}°"
               if self.reused_from else f"  安装倾角 {self.tilt_deg:.2f}°")
        return (f"{self.device_id}  模式 {self.mode}  原点 ({t}) m  "
                f"偏航 {self.yaw_deg:+.2f}°{res}  "
                f"{'Z' if self.z_up else 'Y'} 朝上" + tail)


# ------------------------------------------------------------------ 统计


def _jsonable(v):
    """元组转列表。JSON 没有元组，存回来会变列表 —— 干脆一开始就用列表，
    这样 save/load 走一圈能原样相等。"""
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def mean_position(samples: Sequence[Sample]) -> tuple[float, float, float]:
    return tuple(_mean([s.pos[i] for s in samples]) for i in range(3))  # type: ignore[return-value]


def mean_quaternion(samples_or_quats: Iterable) -> tuple[float, float, float, float]:
    """一组四元数求平均。先统一符号 —— q 和 −q 是同一个旋转，直接平均会互相抵消。

    静置时姿态聚得很紧，这个简单做法够用；姿态分散的场合要改用协方差矩阵
    最大特征向量法。
    """
    quats = [q.rot if isinstance(q, Sample) else q for q in samples_or_quats]
    ref = tf.normalize(quats[0])
    acc = [0.0, 0.0, 0.0, 0.0]
    for q in quats:
        q = tf.normalize(q)
        if sum(a * b for a, b in zip(q, ref)) < 0.0:
            q = tuple(-c for c in q)
        for i in range(4):
            acc[i] += q[i]
    return tf.normalize(tuple(acc))  # type: ignore[arg-type]


def quat_angle_deg(a, b) -> float:
    """两个四元数之间的夹角（度），已处理双覆盖。"""
    d = abs(sum(x * y for x, y in zip(tf.normalize(a), tf.normalize(b))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def decompose_yaw_tilt(q, up_axis: int = 2):
    """swing-twist 分解：``q = q_yaw ⊗ q_tilt``，偏航在**世界系**里施加。

    倾角部分由重力定死，跨会话可复现，那才是真正的安装偏置；偏航部分每次上电
    都不同，只在本次会话有效。
    """
    q = tf.normalize(q)
    twist = [0.0, 0.0, 0.0, 0.0]
    twist[up_axis] = q[up_axis]
    twist[3] = q[3]
    n = math.sqrt(sum(c * c for c in twist))
    if n < 1e-9:                       # 退化：转了 180° 且转轴完全垂直于竖直轴
        twist = [0.0, 0.0, 0.0, 1.0]
    else:
        twist = [c / n for c in twist]
    tw = (twist[0], twist[1], twist[2], twist[3])
    q_tilt = tf.multiply(tf.conjugate(tw), q)
    yaw = math.degrees(2.0 * math.atan2(twist[up_axis], twist[3]))
    if yaw > 180.0:
        yaw -= 360.0
    elif yaw < -180.0:
        yaw += 360.0
    return yaw, q_tilt


def axis_angle(q) -> tuple[tuple, float]:
    """四元数 -> (单位转轴, 角度°)。角度为 0 时转轴退化成 +up，随便给个 Z。"""
    q = tf.normalize(q)
    w = max(-1.0, min(1.0, q[3]))
    ang = 2.0 * math.acos(abs(w))
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-9:
        return (0.0, 0.0, 1.0), 0.0
    sign = 1.0 if w >= 0.0 else -1.0
    return tuple(sign * q[i] / s for i in range(3)), math.degrees(ang)


def check_static(samples: Sequence[Sample], *,
                 max_drift_mm: float = MAX_DRIFT_MM,
                 max_jitter_mm: float = MAX_JITTER_MM,
                 max_span_mm: float = MAX_SPAN_MM,
                 max_tilt_jitter_deg: float = MAX_TILT_JITTER_DEG,
                 min_rate_hz: float = MIN_RATE_HZ,
                 min_samples: int = MIN_SAMPLES) -> StaticReport:
    """这段采样够不够格当基准。

    光看标准差不够 —— SLAM 慢漂的时候每一帧都很稳，但整段在缓慢移动。所以还要
    比**前半段均值和后半段均值**，那个差值才抓得住漂移。
    """
    problems: list = []
    warnings: list = []
    n = len(samples)
    if n < 2:
        return StaticReport(n, 0.0, 0.0, (0.0,) * 3, (0.0,) * 3, 0.0, 0.0, 0.0, {},
                            [f"只有 {n} 帧，采不到东西"], [])

    seconds = (samples[-1].t_ns - samples[0].t_ns) / 1e9
    rate = (n - 1) / seconds if seconds > 0 else 0.0

    jitter = tuple(_std([s.pos[i] for s in samples]) * 1000.0 for i in range(3))
    span = tuple((max(s.pos[i] for s in samples) - min(s.pos[i] for s in samples)) * 1000.0
                 for i in range(3))

    half = n // 2
    a, b = mean_position(samples[:half]), mean_position(samples[half:])
    drift = math.dist(a, b) * 1000.0

    q_mean = mean_quaternion(samples)
    tilt_jitter = math.sqrt(_mean([quat_angle_deg(s.rot, q_mean) ** 2 for s in samples]))

    bad: dict = {}
    for s in samples:
        if s.status != p.POSE_OK:
            name = p.POSE_STATUS_NAMES.get(s.status, f"0x{s.status:02x}")
            bad[name] = bad.get(name, 0) + 1
    ok_pct = 100.0 * (n - sum(bad.values())) / n

    def gate(value, limit, msg, unit=""):
        if value > limit:
            problems.append(f"{msg} {value:.2f}{unit} > 上限 {limit:g}{unit}")
        elif value > limit * WARN_RATIO:
            warnings.append(f"{msg} {value:.2f}{unit} 偏大（上限 {limit:g}{unit}）")

    gate(drift, max_drift_mm, "前后半段漂移", " mm")
    gate(max(jitter), max_jitter_mm, "位置抖动 σ", " mm")
    gate(max(span), max_span_mm, "位置峰峰值", " mm")
    gate(tilt_jitter, max_tilt_jitter_deg, "姿态抖动", "°")

    if n < min_samples:
        problems.append(f"只有 {n} 帧，少于 {min_samples}，统计不可靠")
    if ok_pct < MIN_STATUS_OK_PERCENT:
        problems.append(f"跟踪状态 OK 只占 {ok_pct:.1f}% —— "
                        + "、".join(f"{k}×{v}" for k, v in bad.items()))
    if rate < min_rate_hz:
        warnings.append(f"帧率 {rate:.1f} Hz 偏低（标称 125），丢包可能严重")

    return StaticReport(n, seconds, rate, jitter, span, drift, tilt_jitter, ok_pct, bad,
                        problems, warnings)


# ------------------------------------------------------------------ 求解


def solve(samples: Sequence[Sample], *, mode: str = DEFAULT_MODE,
          z_up: bool = True, body_axes: str = "", mac: bytes = b"\0" * 6,
          report: StaticReport | None = None,
          reuse: "Datum | None" = None, reused_from: str = "",
          lever_body: tuple = (0.0, 0.0, 0.0)) -> Datum:
    """从静置采样解出基准。模式含义见模块开头的表。

    ``reuse`` 给的是台面上标好的 ``level`` 基准。这时倾角修正直接沿用它，只重新
    定位置和偏航 —— 也就是「先标定好再装到机器人上」那条路的第二步。

    顺带能量出台面标定转移得好不好：把沿用的倾角修正贴上去之后，姿态里**本来
    不该再有倾角了**，剩多少就是台面姿态和装机姿态差多少。见
    ``residual_tilt_deg``。
    """
    if mode not in MODES:
        raise ValueError(f"未知模式 {mode!r}，可选 {'/'.join(MODES)}")
    if not samples:
        raise ValueError("没有采样")

    up_axis = 2 if z_up else 1
    if reuse is not None and reuse.lever_body != (0.0, 0.0, 0.0):
        lever_body = reuse.lever_body            # 杆臂也是机体系常量，一起沿用
    if lever_body != (0.0, 0.0, 0.0):
        # 原点定的是杆臂末端，不是设备内部那个参考点
        moved = []
        for smp in samples:
            B = tf.to_matrix(tf.normalize(smp.rot))
            moved.append(smp._replace(pos=tuple(
                smp.pos[i] + sum(B[i][k] * lever_body[k] for k in range(3))
                for i in range(3))))
        samples = moved
        if report is None:
            report = check_static(samples)
    t_w = mean_position(samples)
    q_mean = mean_quaternion(samples)
    yaw, q_tilt = decompose_yaw_tilt(q_mean, up_axis)
    tilt_axis, tilt_deg = axis_angle(q_tilt)

    q_yaw = tf.multiply(q_mean, tf.conjugate(q_tilt))   # q_mean = q_yaw ⊗ q_tilt
    ident = (0.0, 0.0, 0.0, 1.0)
    residual = 0.0

    if reuse is not None:
        if reuse.session_scoped:
            raise ValueError(f"{reused_from or '被沿用的基准'} 是 {reuse.mode} 模式，"
                             f"含地图相关量，换次上电就作废 —— 只有 level 模式能沿用")
        if reuse.z_up != z_up or reuse.body_axes != body_axes:
            raise ValueError("被沿用的基准和当前的 z_up / 机体系约定对不上")
        q_bb = reuse.q_bb
        # 贴上台面标好的倾角修正后，站直时应当只剩偏航；剩下的倾角就是转移误差
        q_eff = tf.multiply(q_mean, q_bb)
        yaw, q_res = decompose_yaw_tilt(q_eff, up_axis)
        tilt_axis, residual = axis_angle(q_res)
        q_ow = tf.conjugate(tf.multiply(q_eff, tf.conjugate(q_res)))
        mode, session = "mount", True
    elif mode == "position":
        q_ow, q_bb, session = ident, ident, True
    elif mode == "yaw":
        q_ow, q_bb, session = tf.conjugate(q_yaw), ident, True
    elif mode == "mount":
        q_ow, q_bb, session = tf.conjugate(q_yaw), tf.conjugate(q_tilt), True
    elif mode == "level":
        # 台面标定：只留由重力定死的那部分，位置和偏航一概不碰
        t_w, q_ow, q_bb, session = (0.0, 0.0, 0.0), ident, tf.conjugate(q_tilt), False
        yaw = 0.0
    else:                                                # full
        q_ow, q_bb, session = tf.conjugate(q_mean), ident, True

    if report is None:
        report = check_static(samples)
    seconds = (samples[-1].t_ns - samples[0].t_ns) / 1e9

    return Datum(
        device_id=p.device_id_str(mac), mac=p.mac_str(mac),
        tracker_index=p.tracker_index(mac), mode=mode, z_up=z_up,
        body_axes=body_axes, up_axis=up_axis,
        t_w=t_w, q_ow=tf.normalize(q_ow), q_bb=tf.normalize(q_bb),
        yaw_deg=yaw, tilt_deg=reuse.tilt_deg if reuse else tilt_deg,
        tilt_axis=tilt_axis,
        lever_body=tuple(float(v) for v in lever_body),
        session_scoped=session, residual_tilt_deg=residual, reused_from=reused_from,
        samples=len(samples), seconds=seconds,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        report=_jsonable(report._asdict()))


# ------------------------------------------------------------------ 存取


def save(path: str | Path, datums: dict) -> None:
    """按设备 id 存一组基准。"""
    payload = {"format": DATUM_FORMAT,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "devices": {k: v.to_dict() for k, v in datums.items()}}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load(path: str | Path, *, z_up: bool | None = None,
         body_axes: str | None = None) -> dict:
    """读回基准，顺便核对约定。

    约定对不上就报错而不是照算 —— 用 +Y 朝上采的基准去套 +Z 朝上的数据，
    算出来的轨迹看着很正常，其实整条都是歪的。这种错误不出声地过去最坑。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fmt = payload.get("format", 0)
    if fmt != DATUM_FORMAT:
        raise ValueError(f"基准文件 {path} 版本 {fmt}，本程序只认 {DATUM_FORMAT}，请重新归零")
    out = {k: Datum.from_dict(v) for k, v in payload["devices"].items()}
    for key, d in out.items():
        if z_up is not None and d.z_up != z_up:
            raise ValueError(
                f"{key} 的基准是按 {'Z' if d.z_up else 'Y'} 朝上采的，"
                f"现在却按 {'Z' if z_up else 'Y'} 朝上跑 —— "
                f"{'加上' if d.z_up else '去掉'} --z-up，或者重新归零")
        if body_axes is not None and d.body_axes != body_axes:
            raise ValueError(
                f"{key} 的基准机体系是 {d.body_axes or '(未换轴)'}，"
                f"现在是 {body_axes or '(未换轴)'} —— 改回去，或者重新归零")
    return out


# ------------------------------------------------------------------ 杆臂
#
# 上报的 xyz 是设备自己那个内部参考点，不是你关心的那个点（G1 胯部中心、
# base_link、随便什么）。两者差一个机体系下的常向量。**机器人原地转身时，
# 这个差直接变成里程计误差** —— 转 180° 会凭空多出 2·|r| 的位移。
# 杆臂 150 mm 就是 300 mm，比回环误差指标大一个量级。
#
# 标法：把设备绕一个按死不动的物理支点各个方向转一圈。body 系下位于 r 的点，
# 世界坐标是 p + R·r；支点不动，所以每一帧都有 p_i + R_i·r = c。对 (r, c) 线性，
# 最小二乘就完了。细节见 frames.estimate_lever_arm。

#: 条件数上限。超了说明姿态转得不够散，解不可靠 —— 实测 cond≤10 精度 <0.1 mm，
#: cond≈30 约 0.5 mm，只绕单轴转会直接爆到 1e15。
LEVER_MAX_CONDITION = 30.0
LEVER_WARN_CONDITION = 10.0
#: 残差上限。支点没按牢、或者中途滑了，就体现在这里。
LEVER_MAX_RMS_M = 0.005


class Lever(NamedTuple):
    """一台设备的杆臂标定结果。跨上电有效 —— 它是机体系里的常量，和地图无关。"""

    device_id: str
    mac: str
    r_body: tuple        #: 机体系，米。目标点位置 = p + R·r_body
    rms_m: float
    condition: float
    spread_deg: float
    samples: int
    z_up: bool
    body_axes: str
    created: str

    @property
    def ok(self) -> bool:
        return self.condition <= LEVER_MAX_CONDITION and self.rms_m <= LEVER_MAX_RMS_M

    @property
    def problems(self) -> list:
        out = []
        if self.condition > LEVER_MAX_CONDITION:
            out.append(f"条件数 {self.condition:.1f} > {LEVER_MAX_CONDITION:g}"
                       f"（姿态才转开 {self.spread_deg:.0f}°）—— 各个方向都要转，"
                       f"别只绕一个轴")
        if self.rms_m > LEVER_MAX_RMS_M:
            out.append(f"残差 {self.rms_m * 1000:.1f} mm > {LEVER_MAX_RMS_M * 1000:g} mm"
                       f" —— 支点没按牢，中途滑了")
        return out

    @property
    def warnings(self) -> list:
        if self.ok and self.condition > LEVER_WARN_CONDITION:
            return [f"条件数 {self.condition:.1f} 偏大（姿态转开 {self.spread_deg:.0f}°），"
                    f"再转开一点能更准"]
        return []

    def summary(self) -> str:
        r = ", ".join(f"{v * 1000:+.1f}" for v in self.r_body)
        return (f"{self.device_id}  杆臂 ({r}) mm  长 {norm3(self.r_body) * 1000:.1f} mm  "
                f"残差 {self.rms_m * 1000:.2f} mm  条件数 {self.condition:.1f}  "
                f"转开 {self.spread_deg:.0f}°")

    def to_dict(self) -> dict:
        d = _jsonable(self._asdict())
        d["format"] = DATUM_FORMAT
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Lever":
        if d.get("format", 0) != DATUM_FORMAT:
            raise ValueError(f"杆臂文件版本 {d.get('format', 0)}，本程序只认 "
                             f"{DATUM_FORMAT}，请重标")
        f = {k: d[k] for k in cls._fields}
        f["r_body"] = tuple(f["r_body"])
        return cls(**f)


def norm3(v) -> float:
    return math.sqrt(sum(c * c for c in v))


def save_levers(path: str | Path, levers: dict) -> None:
    payload = {"format": DATUM_FORMAT,
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "levers": {k: v.to_dict() for k, v in levers.items()}}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_levers(path: str | Path, *, z_up: bool | None = None,
                body_axes: str | None = None) -> dict:
    """读回杆臂，顺便核对约定。

    杆臂是**机体系**里的向量，机体系换了轴它就得跟着换 —— 拿 Y,-X,Z 下标的杆臂
    去套别的约定，方向是错的，而且错得看不出来。所以直接拒绝。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format", 0) != DATUM_FORMAT:
        raise ValueError(f"杆臂文件 {path} 版本 {payload.get('format', 0)}，"
                         f"本程序只认 {DATUM_FORMAT}，请重标")
    out = {k: Lever.from_dict(v) for k, v in payload["levers"].items()}
    for key, lv in out.items():
        if z_up is not None and lv.z_up != z_up:
            raise ValueError(f"{key} 的杆臂是按 {'Z' if lv.z_up else 'Y'} 朝上标的，"
                             f"现在却按 {'Z' if z_up else 'Y'} 朝上跑")
        if body_axes is not None and lv.body_axes != body_axes:
            raise ValueError(f"{key} 的杆臂机体系是 {lv.body_axes or '(未换轴)'}，"
                             f"现在是 {body_axes or '(未换轴)'} —— 杆臂是机体系向量，"
                             f"换了轴方向就错了")
    return out
