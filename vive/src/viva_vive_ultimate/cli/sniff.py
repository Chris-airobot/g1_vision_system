"""原始 HID 报告嗅探，排错用。严格只读，不做任何过滤。

    vvu-sniff                # 按类别统计
    vvu-sniff --hex          # 每包 hexdump
    vvu-sniff --hex --only rf_status

会显示 **所有** 报告，包括 vvu-monitor / vvu-capture 会过滤掉的
配对事件 (0x18) 和 RF 状态 (0x1d/0x1e)。线上一个包都没有时，
说明 tracker 没开机或没连上 dongle。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

from .. import dongle as dg
from .. import protocol as p


def hexdump(data: bytes, prefix: str = "    ", width: int = 16) -> None:
    for off in range(0, len(data), width):
        chunk = data[off : off + width]
        text = "".join(chr(b) if 0x20 <= b <= 0x7E else "." for b in chunk)
        print(f"{prefix}{off:04x}  "
              + " ".join(f"{b:02x}" for b in chunk).ljust(width * 3) + f" {text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-sniff", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", help="显式指定 /dev/hidrawN")
    parser.add_argument("--hex", action="store_true", help="每包 hexdump")
    parser.add_argument("--only", help="只显示这一类（pose/ack/heartbeat/pair_event/rf_status/...）")
    parser.add_argument("--seconds", type=float, default=0, help="时长；0 = 直到 Ctrl-C")
    parser.add_argument("--quiet-pose", action="store_true",
                        help="姿态包只计数不打印（姿态流很吵）")
    args = parser.parse_args(argv)

    try:
        dongle = dg.Dongle(args.device, writable=False)
    except (OSError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        return 2

    kinds: Counter[str] = Counter()
    cmds: Counter[int] = Counter()
    first_of_kind: dict[str, bytes] = {}
    total = 0
    started = time.monotonic()
    last_idle = started
    print(f"reading {dongle.device} (read-only, 全部报告) ... Ctrl-C 停止", flush=True)

    try:
        while not (args.seconds and time.monotonic() - started >= args.seconds):
            got = False
            for raw in dongle.read_raw(timeout=0.5):
                got = True
                total += 1
                cmds[raw[0]] += 1
                kind = p.classify(raw)
                kinds[kind] += 1
                first_of_kind.setdefault(kind, raw)

                if args.only and kind != args.only:
                    continue
                if kind == "pose" and args.quiet_pose:
                    continue

                if kind == "ack":
                    report = p.parse_report(raw)
                    print(f"[ACK ] t{report.tracker_index} {p.mac_str(report.mac)} "
                          f"{report.payload.decode('utf-8', 'replace')!r}", flush=True)
                elif kind == "pair_event":
                    ev = p.parse_pair_event(raw)
                    print(f"[{'解除配对' if ev.unpaired else '配对'}] "
                          f"t{ev.tracker_index} {p.mac_str(ev.mac)}", flush=True)
                elif args.hex:
                    print(f"--- {kind}  cmd=0x{raw[0]:02x}  {len(raw)}B")
                    hexdump(raw)

            now = time.monotonic()
            if not got and now - last_idle >= 3.0:
                last_idle = now
                if total == 0:
                    print(f"[{now - started:5.1f}s] 线上一个包都没有 —— "
                          f"tracker 开机了吗？灯是双闪绿吗？", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        dongle.close()

    elapsed = time.monotonic() - started
    print(f"\n共 {total} 个 report / {elapsed:.1f}s ({total / max(elapsed, 1e-9):.1f} Hz)")
    if not total:
        print("\n线上完全没有流量。检查顺序：")
        print("  1. tracker 开机了吗（灯：蓝闪=未连接，呼吸绿=丢追，双闪绿=正常）")
        print("  2. 和这个 dongle 配对过吗")
        print("  3. dongle 换过 USB 口的话，重新插一下 tracker 也试试")
        return 1
    print("类别   : " + "  ".join(f"{k}:{v}" for k, v in kinds.most_common()))
    print("cmd_id : " + "  ".join(f"0x{k:02x}:{v}" for k, v in cmds.most_common()))
    if not args.hex:
        print("\n各类别首包（--hex 看全部）：")
        for kind, raw in first_of_kind.items():
            print(f"  {kind:22s} {raw[:24].hex(' ')}")
    if kinds.get("heartbeat", 0) > kinds.get("pose", 0):
        print("\n载荷大多是 2 字节心跳 —— tracker 连上了但 SLAM 没启动，"
              "用 vvu-monitor --enable-tracking 拉起来。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
