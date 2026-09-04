"""CLI 公用：参数、启流预检、终端着色。

所有会写 dongle 的 CLI 都从这里拿参数和打开设备，所以安全闸的 CLI 面只有这一处：

- ``--enable-tracking``  才以读写方式打开（否则只读，物理上发不出命令）
- ``--unlock FAMILY --unlock-reason "..."``  显式放行 GUARDED 家族（如 ANI、ATH）
- ``--dry-run``  只校验、只记审计日志，不碰设备
- 打开设备**之前**先把整段启流序列（以及 bench 的 ``--ack``）过一遍闸，
  有一条不放行就直接退出，一个字节都不会发。
"""

from __future__ import annotations

import argparse
import sys

from .. import dongle as dg
from .. import protocol as p
from .. import safety

ESC = "\x1b["
CLR_EOL, HOME, CLEAR = ESC + "K", ESC + "H", ESC + "2J"
HIDE, SHOW = ESC + "?25l", ESC + "?25h"
DIM, BOLD = ESC + "2m", ESC + "1m"
RED, GREEN, YELLOW, CYAN, RST = (ESC + s for s in ("31m", "32m", "33m", "36m", "0m"))


def add_device_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", help="显式指定 /dev/hidrawN")
    parser.add_argument("--enable-tracking", action="store_true",
                        help="发送启流 ACK 序列（tracker 只推 2 字节心跳时需要）。会写 Tracker")
    parser.add_argument("--mode", type=int, default=dg.TRACKING_MODE_SLAM_HOST,
                        help=f"tracking mode，默认 {dg.TRACKING_MODE_SLAM_HOST}=SLAM_HOST，"
                             f"{dg.TRACKING_MODE_SLAM_CLIENT}=SLAM_CLIENT")
    parser.add_argument("--country", default="CN", help="WiFi 国家码（只在 --wifi-host 时发）")
    parser.add_argument("--wifi-host", action="store_true",
                        help="启流时发 Wc/ATH/AWH 让 tracker 起 WiFi Direct 热点（GUARDED，"
                             "需要 --unlock Wc,ATH,AWH）")
    parser.add_argument("--set-device-id", type=int, default=None,
                        help="⚠️ 发 ANI 设置 device id（GUARDED，需要 --unlock ANI）。"
                             "改 id 可能让设备找不到已有地图")
    parser.add_argument("--tracker-mac", help="显式指定 tracker MAC，如 23:31:aa:bb:cc:dd")
    grp = parser.add_argument_group("安全闸")
    grp.add_argument("--unlock", action="append", default=[], metavar="FAMILY",
                     help="解锁一类 GUARDED ACK（逗号分隔或重复给出），如 ANI、ATH。"
                          f"可选：{', '.join(sorted(safety.GUARDED_FAMILIES))}")
    grp.add_argument("--unlock-reason", default="", metavar="TEXT",
                     help="解锁理由，必填，写进审计日志")
    grp.add_argument("--dry-run", action="store_true",
                     help="只校验并记录会发出的命令，不真的写 dongle（设备只读打开）")
    grp.add_argument("--write-log", default=None, metavar="PATH",
                     help=f"审计日志路径，默认 $VVU_WRITE_LOG 或 {safety.DEFAULT_LOG}")


def build_unlock(args) -> safety.Unlock | None:
    """把 --unlock/--unlock-reason 变成 Unlock；参数不合法直接 SystemExit(2)。"""
    items = getattr(args, "unlock", None) or []
    fams = [f for item in items for f in item.split(",") if f.strip()]
    if not fams:
        return None
    try:
        return safety.Unlock(fams, getattr(args, "unlock_reason", "") or "")
    except ValueError as exc:
        raise SystemExit(f"--unlock 参数不合法：{exc}")


def planned_acks(args) -> list[str]:
    """按当前参数，启流会发出的 ACK 序列（不含 bench 的 --ack）。"""
    if not getattr(args, "enable_tracking", False):
        return []
    return dg.tracking_sequence(
        mode=getattr(args, "mode", dg.TRACKING_MODE_SLAM_HOST),
        country=getattr(args, "country", "CN"),
        new_id=getattr(args, "set_device_id", None),
        wifi_host=getattr(args, "wifi_host", False),
    )


def preflight(args, extra_acks: list[str] | None = None) -> safety.Unlock | None:
    """打开设备前把所有计划发出的 ACK 过一遍闸。不过就 SystemExit(2)，不碰设备。"""
    unlock = build_unlock(args)
    planned = planned_acks(args) + list(extra_acks or [])
    try:
        verdicts = safety.require_sequence(planned, unlock)
    except safety.UnsafeCommand as exc:
        raise SystemExit(f"{RED}[安全闸] {exc}{RST}")
    guarded = [v for v in verdicts if v.risk == safety.GUARDED]
    if guarded:
        print(f"{YELLOW}[安全闸] 已解锁 GUARDED：" +
              ", ".join(f"{v.ack}({v.family})" for v in guarded) +
              f"  理由：{unlock.reason if unlock else ''}{RST}", file=sys.stderr, flush=True)
    return unlock


def open_dongle(args, extra_acks: list[str] | None = None) -> dg.Dongle:
    unlock = preflight(args, extra_acks)
    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        print(f"{YELLOW}[安全闸] dry-run：不会真的写 dongle{RST}", file=sys.stderr, flush=True)
    audit = safety.WriteAudit(getattr(args, "write_log", None))
    return dg.Dongle(getattr(args, "device", None),
                     writable=getattr(args, "enable_tracking", False),
                     unlock=unlock, dry_run=dry_run, audit=audit)


class TrackingEnabler:
    """看到 tracker MAC 时自动发一次启流序列。"""

    def __init__(self, dongle: dg.Dongle, args, verbose: bool = True) -> None:
        self.dongle = dongle
        self.args = args
        self.verbose = verbose
        self.done: set[bytes] = set()
        if args.enable_tracking and args.tracker_mac:
            self(p.parse_mac(args.tracker_mac))

    def __call__(self, mac: bytes) -> None:
        if not self.args.enable_tracking or mac in self.done:
            return
        self.done.add(mac)
        if self.verbose:
            print(f"[启流] {p.mac_str(mac)} mode={self.args.mode}"
                  f"{'  (dry-run)' if self.dongle.dry_run else ''}",
                  file=sys.stderr, flush=True)
        try:
            results = self.dongle.enable_tracking(
                mac, mode=self.args.mode, country=self.args.country,
                wifi_host=getattr(self.args, "wifi_host", False),
                new_id=getattr(self.args, "set_device_id", None))
        except safety.UnsafeCommand as exc:
            print(f"{RED}[启流拒绝] {exc}{RST}", file=sys.stderr, flush=True)
            return
        except OSError as exc:
            print(f"[启流失败] {exc}", file=sys.stderr, flush=True)
            return
        if self.verbose:
            for ack, ret in results:
                print(f"  -> {ack:<10s} ret={ret.hex(' ') if ret else '(none)'}",
                      file=sys.stderr, flush=True)
