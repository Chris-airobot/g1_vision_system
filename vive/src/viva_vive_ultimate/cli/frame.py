"""测量安装偏置：把 tracker 报的姿态对齐到你想要的坐标系。

    vvu-frame --seconds 10

把 tracker **按你希望被视为「零姿态」的朝向**摆好、保持不动，跑这条命令。
工具会给出一个修正四元数，之后用它换算，这个朝向就读成单位姿态。

⚠️ 一个必须知道的事实
---------------------
设备的世界系是**重力对齐**的（+Y 朝上），但**偏航每次开机都不同** ——
地图是每次上电现场重建的，原点朝向取决于建图那一刻设备的水平朝向。

所以修正量分两部分：

- **倾斜部分**（由重力定死）→ **跨会话可复现**，这才是真正的安装偏置
- **偏航部分** → **只在本次会话有效**，每次上电要重新归零

工具会把两部分分开报给你。
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from .. import protocol as p
from ._common import CYAN, DIM, RST, BOLD, TrackingEnabler, add_device_args, open_dongle

#: 世界系「上」的轴。本设备实测是 +Y。
UP_AXIS = 1
AXIS_NAMES = "XYZ"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-frame", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_args(parser)
    parser.add_argument("--seconds", type=float, default=10.0, help="静止采样时长")
    parser.add_argument("--tracker", help="只测这一台（device id，如 95:65:2e:67）")
    parser.add_argument("--z-up", action="store_true",
                        help="额外给出转成 Z 朝上（常见机器人约定）的变换")
    args = parser.parse_args(argv)

    try:
        import numpy as np
        from ..frames import (average_quaternion, decompose_yaw_tilt, quat_conjugate,
                              quat_multiply, quat_to_matrix)
    except ImportError as exc:
        print(f'需要 numpy：pip install -e ".[analysis]"  ({exc})', file=sys.stderr)
        return 2

    try:
        dongle = open_dongle(args)
    except (OSError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        return 2

    enabler = TrackingEnabler(dongle, args)
    samples: dict[str, list] = {}
    moved: dict[str, float] = {}
    prev: dict[str, tuple] = {}

    print(f"{BOLD}保持 tracker 静止，采样 {args.seconds:.0f} 秒 …{RST}", flush=True)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            for report in dongle.read_reports(timeout=0.3):
                enabler(report.mac)
                if not report.is_pose:
                    continue
                pose = p.decode_pose(report.payload)
                if pose is None or pose.status != p.POSE_OK:
                    continue
                did = p.device_id_str(report.mac)
                if args.tracker and did != args.tracker:
                    continue
                samples.setdefault(did, []).append(pose.rot)
                if did in prev:
                    moved[did] = moved.get(did, 0.0) + math.dist(pose.pos, prev[did])
                prev[did] = pose.pos
    except KeyboardInterrupt:
        pass
    finally:
        dongle.close()

    if not samples:
        print("没采到有效姿态。tracker 开机了吗？status 是 OK 吗？", file=sys.stderr)
        return 1

    for did, quats in sorted(samples.items()):
        n = len(quats)
        travel_mm = moved.get(did, 0.0) * 1000
        print(f"\n{'=' * 72}")
        print(f"{BOLD} {did}{RST}   {n} 帧   采样期间移动 {travel_mm:.1f} mm"
              + (f"   {DIM}(> 5 mm，可能没拿稳){RST}" if travel_mm > 5 else ""))
        print("=" * 72)

        q = average_quaternion(quats)                 # (x, y, z, w)
        spread = np.degrees(2 * np.arccos(np.clip(
            np.abs(np.asarray(quats) @ q) / np.linalg.norm(quats, axis=1), -1, 1))).max()
        print(f"\n{CYAN}实测姿态{RST}  (x,y,z,w) = "
              f"({q[0]:+.5f}, {q[1]:+.5f}, {q[2]:+.5f}, {q[3]:+.5f})   "
              f"采样内最大偏差 {spread:.3f}°")

        M = quat_to_matrix(q)
        print(f"\n{CYAN}本体轴在世界系下的指向{RST}")
        for j in range(3):
            col = M[:, j]
            near = int(np.argmax(np.abs(col)))
            sign = "+" if col[near] > 0 else "-"
            print(f"   本体 {AXIS_NAMES[j]} 轴  ({col[0]:+.4f}, {col[1]:+.4f}, {col[2]:+.4f})"
                  f"   ≈ {sign}{AXIS_NAMES[near]}"
                  + ("   ← 这是「上」" if near == UP_AXIS and col[near] > 0 else ""))

        yaw, q_tilt = decompose_yaw_tilt(q, UP_AXIS)
        tilt_ang = math.degrees(2 * math.acos(min(1.0, abs(float(q_tilt[3])))))
        print(f"\n{CYAN}拆解{RST}")
        print(f"   偏航（绕 {AXIS_NAMES[UP_AXIS]}）  {yaw:+8.3f}°   "
              f"{DIM}← 本次会话有效，每次上电要重测{RST}")
        print(f"   倾斜               {tilt_ang:8.3f}°   "
              f"{DIM}← 由重力定死，跨会话可复现，这才是安装偏置{RST}")

        q_inv = quat_conjugate(q)
        print(f"\n{CYAN}修正四元数{RST}（把当前朝向变成单位姿态）")
        print(f"   Q_REF_INV = ({q_inv[0]:+.6f}, {q_inv[1]:+.6f}, "
              f"{q_inv[2]:+.6f}, {q_inv[3]:+.6f})     # (x,y,z,w)")

        print(f"\n{CYAN}用法{RST}")
        print("   from viva_vive_ultimate.frames import quat_multiply")
        print(f"   Q_REF_INV = np.array([{q_inv[0]:+.6f}, {q_inv[1]:+.6f}, "
              f"{q_inv[2]:+.6f}, {q_inv[3]:+.6f}])")
        print("   q_out = quat_multiply(Q_REF_INV, pose.rot)   # 校正后的姿态")
        print("   p_out = R_REF_INV @ (pose.pos - P_REF)       # 位置同理，见下")

        if args.z_up:
            # 把 +Y 朝上转成 +Z 朝上：绕 X 转 +90°
            s = math.sin(math.radians(45)); c = math.cos(math.radians(45))
            q_zup = np.array([s, 0.0, 0.0, c])
            q_total = quat_multiply(q_zup, q_inv)
            R = quat_to_matrix(q_total)
            print(f"\n{CYAN}Z 朝上约定{RST}（在上面基础上再绕 X 转 +90°）")
            print(f"   Q_TOTAL = ({q_total[0]:+.6f}, {q_total[1]:+.6f}, "
                  f"{q_total[2]:+.6f}, {q_total[3]:+.6f})")
            print("   R_TOTAL =")
            for row in R:
                print(f"      [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")

    print(f"\n{DIM}提示：换一个已知朝向再测一次，两次的「倾斜」应当一致 —— "
          f"不一致说明摆放不准或设备没拿稳。{RST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
