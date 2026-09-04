"""终端实时检查位置与姿态。拿着 tracker 移动/旋转，对着读数核。

    vvu-monitor                     看
    vvu-monitor --enable-tracking   顺便启流

按键: z 归零 · r 重置统计 · q 退出
"""

from __future__ import annotations

import argparse
import sys
import termios
import time
import tty

from .. import protocol as p
from .. import transforms as tf
from ..state import TrackerState
from ._common import (BOLD, CLEAR, CLR_EOL, CYAN, DIM, ESC, GREEN, HIDE, HOME,
                      RED, RST, SHOW, YELLOW, TrackingEnabler, add_device_args,
                      open_dongle)

WIDTH = 78


def bar(value: float, lo: float, hi: float, width: int = 34) -> str:
    frac = 0.5 if hi - lo < 1e-9 else (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    cells = ["─"] * width
    cells[width // 2] = "┼"
    cells[int(round(frac * (width - 1)))] = "●"
    return "".join(cells)


def render(st: TrackerState, out: list[str]) -> None:
    w = out.append
    pose = st.pose
    if pose is None:
        w(f" tracker {st.index}  {st.mac_str}   {DIM}等待姿态包…{RST}")
        return

    color = GREEN if pose.status == p.POSE_OK else (
        YELLOW if pose.status == p.POSE_RECENTLY_LOST else RED)
    up = int(st.uptime)
    w(f"{BOLD} tracker {st.index}{RST}  {st.mac_str}   {color}{pose.status_name:<13s}{RST} "
      f"{st.hz:6.1f} Hz   丢包 {st.lost} ({st.loss_percent:.2f}%)   "
      f"{up // 60:02d}:{up % 60:02d}   btn=0x{pose.btns:02x}  dt={pose.device_time}")
    w("")

    rel = st.relative_pos
    zeroed = st.zero_pos is not None
    w(f"{CYAN} 位置 (m){RST}      当前        {'相对零点' if zeroed else '(未归零)'}"
      f"        范围              跨度")
    for i, name in enumerate("XYZ"):
        lo, hi = st.lo[i], st.hi[i]
        w(f"   {name}      {pose.pos[i]:+9.4f}     {rel[i]:+9.4f}     "
          f"{lo:+7.3f}..{hi:+7.3f}   {st.span(i) * 1000:7.1f} mm")
    w("")
    for i, name in enumerate("XYZ"):
        lo, hi = st.lo[i], st.hi[i]
        if hi - lo < 0.02:
            lo, hi = pose.pos[i] - 0.5, pose.pos[i] + 0.5
        w(f"   {name} {DIM}{bar(pose.pos[i], lo, hi)}{RST}  {pose.pos[i]:+.3f}")
    w("")

    q = tf.normalize(pose.rot)
    qrel = st.relative_quat
    w(f"{CYAN} 姿态{RST}")
    w(f"   四元数 (x,y,z,w)  {q[0]:+8.4f} {q[1]:+8.4f} {q[2]:+8.4f} {q[3]:+8.4f}"
      f"   |q|={tf.norm(pose.rot):.4f}")
    roll, pitch, yaw = tf.to_euler_deg(qrel)
    tag = "相对零点" if st.zero_quat else "绝对"
    w(f"   欧拉角 ({tag})    roll(X) {roll:+8.2f}°   pitch(Y) {pitch:+8.2f}°   "
      f"yaw(Z) {yaw:+8.2f}°")
    w("")
    w(f"   {DIM}本体轴指向（世界系）        X 分量    Y 分量    Z 分量{RST}")
    for name, axis in zip("XYZ", tf.body_axes(qrel)):
        w(f"     本体 {name} 轴            {axis[0]:+7.3f}   {axis[1]:+7.3f}   {axis[2]:+7.3f}")
    w("")
    w(f"{CYAN} 运动{RST}   |acc| = {tf.magnitude(pose.acc):6.3f}    "
      f"|角速度| = {tf.magnitude(pose.rot_vel):6.3f} rad/s    "
      f"累计行程 = {st.path_length:6.3f} m")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-monitor", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_args(parser)
    parser.add_argument("--fps", type=float, default=20.0, help="终端刷新率")
    args = parser.parse_args(argv)

    try:
        dongle = open_dongle(args)
    except (OSError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        return 2

    enabler = TrackingEnabler(dongle, args, verbose=False)
    states: dict[int, TrackerState] = {}
    interactive = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin.fileno()) if interactive else None
    if interactive:
        tty.setcbreak(sys.stdin.fileno())

    sys.stdout.write(HIDE + CLEAR)
    next_draw, started = 0.0, time.monotonic()

    try:
        while True:
            if interactive:
                import select as _sel
                if _sel.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == "q":
                        break
                    if key == "z":
                        for s in states.values():
                            s.set_zero()
                    if key == "r":
                        for s in states.values():
                            s.reset_stats()

            for report in dongle.read_reports(timeout=0.02):
                enabler(report.mac)
                if not report.is_pose:
                    continue
                pose = p.decode_pose(report.payload)
                if pose is None:
                    continue
                tid = report.tracker_index
                if tid not in states:
                    states[tid] = TrackerState(tid, report.mac)
                states[tid].update(pose)

            now = time.monotonic()
            if now < next_draw:
                continue
            next_draw = now + 1.0 / args.fps

            out = [f"{BOLD}  VIVE Ultimate Tracker — 位置/姿态实时检查{RST}"
                   f"   {DIM}{dongle.device}{RST}", "─" * WIDTH]
            if not states:
                out += ["", f"  {YELLOW}还没收到姿态包。{RST}",
                        "  · tracker 开机了吗？灯是双闪绿吗？",
                        "  · 只收到 2 字节心跳的话，加 --enable-tracking 重跑",
                        f"  {DIM}已等待 {int(now - started)} 秒{RST}"]
            for st in sorted(states.values(), key=lambda s: s.index):
                render(st, out)
                out.append("─" * WIDTH)
            out.append(f" {DIM}[z] 归零   [r] 重置统计   [q] 退出{RST}")

            sys.stdout.write(HOME + "\n".join(l + CLR_EOL for l in out) + ESC + "J")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        dongle.close()
        if old_term is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term)
        sys.stdout.write(SHOW + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
