#!/usr/bin/env python3
"""核对 tracker 实际输出的机体系是否与 URDF 里的 ``vive_tracker_link`` 对齐。

只读：``Dongle()`` 默认只读打开，不发任何命令。tracker 必须已经在 tracking
（之前用 ``vvu-monitor --enable-tracking`` 打开过）。

机体系约定与 URDF 一致：Z-up 世界 + ``BODY_AXES_REMAP="Y,-X,Z"``，
即 X 沿厚度指向安装面、Y 沿 79 mm 长边、Z 沿 59 mm 短边。
URDF 假设横装、摄像头朝后，rpy="0 0 0"，于是机器人站直时应当：

    body Z ≈ 世界 +Z（竖直）      body X ≈ 机器人前方      body Y ≈ 机器人左方

两种模式::

    static  机器人站直不动，采几秒，看哪根机体轴竖直。
            只能判 roll/pitch：横装 / 竖装 / 上下翻；判不了绕竖直轴的朝向。
    push    先静置 settle 秒，再把机器人**沿它自己的前方**推 ≥0.3 m，
            看位移落在初始机体系的哪根轴上。期望 +X。这一步验证厚度方向的正负。

两种检查都只验证**旋转**。tracker 内部 SLAM 原点相对安装面中心的平移
无法用这两步得到，要靠手眼标定（V_T_T @ T_T_B @ B_T_K = V_T_K）。
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_config as cfg

from viva_vive_ultimate import Dongle, decode_pose, device_id_str
from viva_vive_ultimate.frames import (
    average_quaternion, parse_axis_remap, quat_to_matrix,
    remap_body_quat, to_z_up_position, to_z_up_quat,
)
from viva_vive_ultimate.meshio import BODY_AXES_REMAP

DEFAULT_DEVICE = cfg.TARGET_TRACKER      # 与 g1_config / hybrid 脚本一致，--device-id 覆盖

#: URDF vive_tracker_joint 的 origin（pelvis -> link）
URDF_XYZ = np.asarray(cfg.TRACKER_MOUNT_XYZ, dtype=float)

#: 各种「哪根轴朝上」对应的 URDF rpy 建议
RPY_SUGGEST = {
    ("Z", +1): ("0 0 0", "横装、Z 朝上 —— 与当前 URDF 一致"),
    ("Z", -1): ("3.1416 0 0", "横装但上下翻了"),
    ("Y", +1): ("1.5708 0 0", "竖装、79 mm 长边竖直、Y 朝上"),
    ("Y", -1): ("-1.5708 0 0", "竖装、Y 朝下"),
    ("X", +1): (None, "厚度方向竖直：tracker 不是平贴在竖直背面，或机体系约定有误"),
    ("X", -1): (None, "厚度方向竖直：tracker 不是平贴在竖直背面，或机体系约定有误"),
}


class Stream:
    """只读取流，把目标 tracker 的 OK 帧换算成 Z-up 世界 + 重映射机体系。"""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.R_remap = parse_axis_remap(BODY_AXES_REMAP)
        self.dongle = Dongle()                      # 只读打开，不发命令
        self.seen = collections.Counter()           # (device, kind) -> 计数，用于诊断

    def close(self):
        self.dongle.close()

    def collect(self, seconds: float, label: str):
        """采 seconds 秒（时间驱动，没有报文也会按时返回），返回 (pos (N,3), quat_T (N,4))。"""
        P, Q = [], []
        t0 = time.monotonic()
        last_print = t0
        while True:
            now = time.monotonic()
            if now - t0 >= seconds:
                break
            for report in self.dongle.read_reports(0.2):
                dev = device_id_str(report.mac)
                if not report.is_pose:
                    self.seen[(dev, "heartbeat" if report.is_heartbeat else "other")] += 1
                    continue
                pose = decode_pose(report.payload)
                if pose is None:
                    continue
                self.seen[(dev, pose.status_name)] += 1
                if dev != self.device_id or pose.status_name != "OK":
                    continue
                q = np.asarray(to_z_up_quat(np.asarray(pose.rot, dtype=float)))
                Q.append(remap_body_quat(q, self.R_remap))
                P.append(to_z_up_position(pose.pos))
            if now - last_print >= 1.0:
                last_print = now
                print(f"  [{label}] 剩余 {seconds - (now - t0):4.1f} s，OK 帧 {len(P)}", flush=True)
        if not P:
            lines = "\n".join(f"    {dev}: {kind} x{n}" for (dev, kind), n in sorted(self.seen.items()))
            sys.exit(f"没有收到 {self.device_id} 的 OK 帧。dongle 这段时间看到的：\n"
                     f"{lines or '    （什么都没有）'}\n"
                     "  只有 heartbeat = 连上了但没在 tracking；完全没有 = 没开机/没配对到这只 dongle。")
        return np.asarray(P), np.asarray(Q)


def body_axes_report(Q):
    """返回 (R_V_T 均值, 竖直轴名, 符号, 与竖直的夹角)。"""
    R = quat_to_matrix(average_quaternion(Q))
    print("  机体轴在 Z-up 世界里的方向（列向量）：")
    for k, name in enumerate("XYZ"):
        ax = R[:, k]
        ang = np.degrees(np.arccos(np.clip(abs(ax[2]), -1.0, 1.0)))
        print(f"    body {name} = [{ax[0]:+.3f} {ax[1]:+.3f} {ax[2]:+.3f}]   与竖直夹角 {ang:5.1f}°")
    k = int(np.argmax(np.abs(R[2, :])))
    sign = int(np.sign(R[2, k]))
    tilt = np.degrees(np.arccos(np.clip(abs(R[2, k]), -1.0, 1.0)))
    return R, "XYZ"[k], sign, tilt


def mode_static(st: Stream, seconds: float):
    print(f"\n[static] 机器人站直、保持不动，采 {seconds:.0f} s ...")
    P, Q = st.collect(seconds, "static")
    print(f"  OK 帧 {len(P)}，位置抖动 std [mm] = {(P.std(0) * 1000).round(2)}")
    R, axis, sign, tilt = body_axes_report(Q)
    rpy, why = RPY_SUGGEST[(axis, sign)]
    print(f"\n  竖直的是 body {'+' if sign > 0 else '-'}{axis}，偏离竖直 {tilt:.1f}°  ->  {why}")
    if rpy is None:
        print("  结论：与 URDF 不一致，先检查安装方式 / 机体系约定。")
    elif rpy == "0 0 0":
        print(f"  结论：roll/pitch 与 URDF 一致（残余倾角 {tilt:.1f}° 含骨盆站姿倾角与安装误差）。")
        print("  绕竖直轴的朝向（X 朝前还是朝后）静态判不了，跑 --mode push。")
    else:
        print(f"  结论：把 vive_tracker_joint 的 rpy 改成 \"{rpy}\"，然后重跑本检查。")
    return R


def mode_push(st: Stream, settle: float, move: float, tail: float):
    print(f"\n[push] 第一步：机器人站直不动 {settle:.0f} s（取初始姿态）...")
    P0, Q0 = st.collect(settle, "settle")
    R0 = quat_to_matrix(average_quaternion(Q0))
    p0 = P0.mean(0)
    print(f"  初始位置 {p0.round(4)}，抖动 std [mm] = {(P0.std(0) * 1000).round(2)}")
    _, axis, sign, tilt = body_axes_report(Q0)
    if not (axis == "Z" and sign > 0):
        print(f"  注意：静态判断竖直轴是 {'+' if sign > 0 else '-'}{axis}，不是 +Z，先修 URDF rpy 再看 push 结果。")

    print(f"\n[push] 第二步：现在把机器人沿它自己的前方直推 ≥0.3 m，不要转向，{move:.0f} s 内完成 ...")
    P1, _ = st.collect(move, "push")
    # 终点取最后 tail 秒的均值；同时报最大位移
    n_tail = max(5, int(len(P1) * tail / move))
    p_end = P1[-n_tail:].mean(0)
    d_V = p_end - p0
    d_T0 = R0.T @ d_V                      # 位移换到初始机体系
    dist = np.linalg.norm(d_V)
    print(f"\n  世界系位移 d_V  = {d_V.round(4)}   |d| = {dist * 1000:.0f} mm")
    print(f"  初始机体系位移 d_T0 = {d_T0.round(4)}")
    if dist < 0.15:
        print("  位移太小（<150 mm），判不出方向。再推远一点。")
        return
    u = d_T0 / dist
    k = int(np.argmax(np.abs(u)))
    sign = int(np.sign(u[k]))
    off = np.degrees(np.arccos(min(1.0, abs(u[k]))))
    name = "XYZ"[k]
    print(f"  位移主轴：body {'+' if sign > 0 else '-'}{name}，偏离该轴 {off:.1f}°")
    if name == "X" and sign > 0:
        print("  结论：前推 = body +X，厚度方向正负与 URDF 一致（X 指向机器人前方）。")
    elif name == "X":
        print("  结论：前推 = body -X。要么 tracker 是反着贴的（摄像头面贴机器人，不太可能），"
              "要么 meshio 的 CAD->机体系厚度方向符号反了；URDF 里等价于 rpy=\"0 0 3.1416\"。")
    elif name == "Y":
        print("  结论：前推落在 body Y 上 —— 安装绕竖直轴转了 90°，或推的方向不是机器人前方。")
    else:
        print("  结论：前推落在 body Z 上 —— 与静态判断矛盾，检查是否推的时候把机器人抬起来了。")
    if off > 20:
        print(f"  注意：偏离主轴 {off:.1f}° 偏大，可能推歪了或 SLAM 漂了，建议重推一次。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["static", "push"], default="static")
    ap.add_argument("--device-id", default=DEFAULT_DEVICE, help=f"tracker 设备 id，默认 {DEFAULT_DEVICE}")
    ap.add_argument("--seconds", type=float, default=5.0, help="static：采样秒数")
    ap.add_argument("--settle", type=float, default=3.0, help="push：初始静置秒数")
    ap.add_argument("--move", type=float, default=10.0, help="push：推动窗口秒数")
    ap.add_argument("--tail", type=float, default=1.5, help="push：终点取最后几秒均值")
    a = ap.parse_args(argv)

    print("URDF vive_tracker_joint: xyz", URDF_XYZ, "rpy 0 0 0  （名义 T_T_B 平移 =", (-URDF_XYZ).round(3), "）")
    print(f"机体系：Z-up + BODY_AXES_REMAP={BODY_AXES_REMAP}，设备 {a.device_id}，只读")
    st = Stream(a.device_id)
    try:
        if a.mode == "static":
            mode_static(st, a.seconds)
        else:
            mode_push(st, a.settle, a.move, a.tail)
    finally:
        st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
