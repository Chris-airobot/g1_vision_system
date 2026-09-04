"""实测姿态链路的速率、丢包与抖动。

    vvu-bench --seconds 60
    vvu-bench --seconds 60 --json run1.json      # 存下来好对比
    vvu-bench --seconds 60 --ack ACF30           # 先改相机 FPS 再测

只读；只有 --ack / --enable-tracking 会写，且都走 DCMD_TX 白名单。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import Counter

from .. import protocol as p
from ._common import BOLD, CYAN, DIM, RST, TrackingEnabler, add_device_args, open_dongle

#: USB EP1 IN 是 interrupt、bInterval=1，即 1 ms 轮询上限
USB_REPORT_CEILING_HZ = 1000.0


class Meter:
    """单台 tracker 的原始采样，测完再统一算。"""

    def __init__(self, index: int, mac: bytes) -> None:
        self.index = index
        self.mac = mac
        self.recv_ns: list[int] = []
        self.idx: list[int] = []
        self.device_time: list[int] = []
        self.status = Counter()

    def add(self, pose: p.Pose, recv_ns: int) -> None:
        self.recv_ns.append(recv_ns)
        self.idx.append(pose.idx)
        self.device_time.append(pose.device_time)
        self.status[pose.status_name] += 1

    def report(self) -> dict:
        n = len(self.recv_ns)
        if n < 3:
            return {"tracker": self.index, "mac": p.mac_str(self.mac), "frames": n,
                    "error": "样本太少"}

        wall = (self.recv_ns[-1] - self.recv_ns[0]) / 1e9
        gaps = [p.idx_gap(a, b) for a, b in zip(self.idx, self.idx[1:])]
        lost = sum(g for g in gaps if g > 0)
        # dongle 偶发重复投递同一帧（整包除主机时间戳外完全相同）
        duplicates = sum(1 for g in gaps if g == -1)
        source = n - duplicates + lost

        # device_time 累计 tick 对墙钟 -> 反推 tick 长度
        ticks = [(b - a) & 0xFFFF for a, b in zip(self.device_time, self.device_time[1:])]
        total_ticks = sum(ticks)
        tick_us = wall / total_ticks * 1e6 if total_ticks else float("nan")

        # 设备侧停顿：idx 连续但 device_time 跳了好几个帧周期。
        # tracker 自己没生成位姿，idx 不计数，所以丢包统计看不见。
        stall_ticks = [t for t, g in zip(ticks, gaps)
                       if g == 0 and t > p.DEVICE_STALL_TICKS]
        # 只统计连续帧（gap==0）且非停顿的单帧周期
        single = [t for t, g in zip(ticks, gaps)
                  if g == 0 and t <= p.DEVICE_STALL_TICKS]
        period_hist = Counter(single).most_common(6)

        dt_ms = [(b - a) / 1e6 for a, b in zip(self.recv_ns, self.recv_ns[1:])]
        ordered = sorted(dt_ms)

        def pct(q: float) -> float:
            return ordered[min(len(ordered) - 1, int(len(ordered) * q))]

        return {
            "tracker": self.index,
            "mac": p.mac_str(self.mac),
            "duration_s": round(wall, 4),
            "frames": n,
            "lost": lost,
            "duplicates": duplicates,
            "device_stalls": len(stall_ticks),
            "stalled_s": round(sum(stall_ticks) * p.DEVICE_TIME_TICK_SECONDS, 4),
            "source_hz": round(source / wall, 3),
            "delivered_hz": round((n - duplicates) / wall, 3),
            "loss_pct": round(100.0 * lost / source, 3) if source else 0.0,
            "max_consecutive_lost": max(gaps) if gaps else 0,
            "tick_us": round(tick_us, 4),
            "mean_period_ticks": round(total_ticks / (source - 1), 2) if source > 1 else 0,
            "period_hist": [[int(t), c, round(100.0 * c / len(single), 2)]
                            for t, c in period_hist] if single else [],
            "arrival_ms": {"p50": round(pct(0.50), 3), "p95": round(pct(0.95), 3),
                           "p99": round(pct(0.99), 3), "max": round(ordered[-1], 3)},
            "bursts_lt_1ms": sum(1 for x in dt_ms if x < 1.0),
            "stalls_gt_20ms": sum(1 for x in dt_ms if x > 20.0),
            "status": dict(self.status),
        }


def render(reports: list[dict], device: str, seconds: float) -> None:
    print()
    print("=" * 80)
    print(f"{BOLD} VVU 速率实测{RST}   {DIM}{device}   {seconds:.1f}s{RST}")
    print("=" * 80)

    total_delivered = 0.0
    for r in reports:
        print()
        if "error" in r:
            print(f"{BOLD} tracker {r['tracker']}{RST}  {r['mac']}   {r['error']}")
            continue
        total_delivered += r["delivered_hz"]
        print(f"{BOLD} tracker {r['tracker']}{RST}  {r['mac']}")
        print(f"   {CYAN}源速率{RST}        {r['source_hz']:8.2f} Hz    "
              f"(收 {r['frames']} + 丢 {r['lost']})")
        print(f"   {CYAN}实收速率{RST}      {r['delivered_hz']:8.2f} Hz")
        print(f"   {CYAN}RF 丢包{RST}       {r['loss_pct']:8.2f} %     "
              f"最大连续丢 {r['max_consecutive_lost']} 帧"
              + (f"   重复帧 {r['duplicates']}" if r.get("duplicates") else ""))
        if r.get("device_stalls"):
            print(f"   {CYAN}设备侧停顿{RST}   {r['device_stalls']:5d} 次   累计 "
                  f"{r['stalled_s']*1000:.0f} ms   "
                  f"{DIM}（idx 连续，丢包统计看不见；实测只在 SLAM 丢追时出现）{RST}")
        print()
        print(f"   device_time    tick {r['tick_us']:.3f} µs    "
              f"平均周期 {r['mean_period_ticks']:.1f} ticks "
              f"= {r['mean_period_ticks'] * r['tick_us'] / 1000:.3f} ms")
        if r["period_hist"]:
            hist = "  ".join(f"{t}:{c}({q:.1f}%)" for t, c, q in r["period_hist"])
            print(f"   单帧周期分布   {hist}")
        a = r["arrival_ms"]
        print()
        print(f"   主机到达间隔   p50 {a['p50']:.2f}   p95 {a['p95']:.2f}   "
              f"p99 {a['p99']:.2f}   max {a['max']:.1f} ms")
        print(f"   成簇 <1ms {r['bursts_lt_1ms']} 次    卡顿 >20ms {r['stalls_gt_20ms']} 次")
        print(f"   状态分布       {r['status']}")

    print()
    print("-" * 80)
    usb = 100.0 * total_delivered / USB_REPORT_CEILING_HZ
    print(f" 合计 {len(reports)} 台    报告率 {total_delivered:.1f} /s    "
          f"USB 占用 {usb:.1f}%  (上限 {USB_REPORT_CEILING_HZ:.0f}/s)")
    if len(reports) > 1:
        hzs = [r["delivered_hz"] for r in reports if "delivered_hz" in r]
        if hzs:
            print(f" 每台实收 {min(hzs):.1f} .. {max(hzs):.1f} Hz    "
                  f"总源速率 {sum(r.get('source_hz', 0) for r in reports):.1f} Hz")
    print("-" * 80)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-bench", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_args(parser)
    parser.add_argument("--seconds", type=float, default=30.0, help="采集时长，建议 >= 30")
    parser.add_argument("--json", help="把结果写成 JSON，便于多次对比")
    parser.add_argument("--ack", action="append", default=[],
                        help="测量前给每台 tracker 发一条 ACK（如 ACF30）；可重复。"
                             "每条都先过安全闸：SAFE 直接发，GUARDED 要 --unlock，"
                             "FORBIDDEN/未知直接拒绝")
    parser.add_argument("--label", default="", help="给这次测量起个名字，写进 JSON")
    args = parser.parse_args(argv)

    # --ack 里的每一条都先过安全闸（FORBIDDEN/UNKNOWN 直接退出，GUARDED 要 --unlock），
    # 再打开设备 —— 一条不放行就一个字节都不发。
    args.ack = [a for item in args.ack for a in item.split(",") if a]
    needs_write = args.enable_tracking or bool(args.ack)
    args.enable_tracking = args.enable_tracking or bool(args.ack)
    try:
        dongle = open_dongle(args, extra_acks=args.ack)
    except (OSError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        return 2
    if needs_write and not dongle.writable:
        print("需要写权限才能发 ACK", file=sys.stderr)
        return 2

    enabler = TrackingEnabler(dongle, args)
    acked: set[bytes] = set()
    meters: dict[int, Meter] = {}
    started = time.monotonic()
    last_log = started

    print(f"采集 {args.seconds:.0f}s ... Ctrl-C 提前结束", file=sys.stderr, flush=True)
    try:
        while time.monotonic() - started < args.seconds:
            for report in dongle.read_reports(timeout=0.5):
                enabler(report.mac)
                if args.ack and report.mac not in acked:
                    acked.add(report.mac)
                    for ack in args.ack:
                        print(f"[ACK] {p.mac_str(report.mac)} <- {ack}", file=sys.stderr)
                    dongle.send_acks(report.mac, args.ack)
                    # 发完命令给设备一点时间生效，之前的样本作废
                    time.sleep(1.0)
                    meters.clear()
                    started = time.monotonic()
                    continue
                if not report.is_pose:
                    continue
                pose = p.decode_pose(report.payload)
                if pose is None:
                    continue
                tid = report.tracker_index
                if tid not in meters:
                    meters[tid] = Meter(tid, report.mac)
                meters[tid].add(pose, time.monotonic_ns())

            now = time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                frames = sum(len(m.recv_ns) for m in meters.values())
                print(f"\r[{now - started:5.0f}/{args.seconds:.0f}s] "
                      f"{len(meters)} 台  {frames} 帧", end="", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        dongle.close()
    print(file=sys.stderr)

    reports = [meters[k].report() for k in sorted(meters)]
    if not reports:
        print("没收到姿态包。tracker 开机了吗？只有心跳的话加 --enable-tracking", file=sys.stderr)
        return 1

    render(reports, dongle.device, time.monotonic() - started)

    if args.json:
        blob = {"label": args.label, "device": dongle.device,
                "seconds": args.seconds, "acks": args.ack, "trackers": reports}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
