"""静置 30 秒定原点。

    vvu-datum -o datum.json                  # 静置 30 s，写基准
    vvu-datum -o datum.json --seconds 60     # 想更稳就采久一点
    vvu-datum --show datum.json              # 看看之前存的是什么

设备上电即建图，世界系的原点和偏航每次开机都不一样。每次上电跑一遍这个，
后面 vvu-capture / vvu-viz 加 --datum 就能把轨迹归到同一个原点上。

**想先在台面上标好、再装到机器人上**，那就分两步 —— 只有由重力定死的倾角能
留到下次，位置和偏航必须装好之后现场定：

    vvu-datum --datum-mode level -o body.json     # 台面上做一次，永久有效
    vvu-datum --reuse body.json -o origin.json    # 装好后每次上电，10 秒够

第二步会顺带把「台面姿态和装机姿态差多少」量出来（残余倾角）。

**归零期间别碰设备。** 程序会核对这段到底静没静 —— 抖动、峰峰值、前后半段
漂移、跟踪状态全都要过线，不过线不写文件。
"""

from __future__ import annotations

import argparse
import sys
import time

from .. import datum as dt
from .. import protocol as p
from ._common import (BOLD, CYAN, DIM, GREEN, RED, RST, YELLOW,
                      TrackingEnabler, add_device_args, open_dongle)


def _show(path: str) -> int:
    try:
        datums = dt.load(path)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"{BOLD}{path}{RST}")
    for d in datums.values():
        print(f"  {d.summary()}")
        print(f"    {DIM}{d.created}  {d.samples} 帧 / {d.seconds:.1f}s{RST}")
        r = d.report
        print(f"    {DIM}抖动 σ={'/'.join(f'{v:.2f}' for v in r['jitter_mm'])} mm  "
              f"半段漂移 {r['drift_mm']:.2f} mm  姿态抖动 {r['tilt_jitter_deg']:.3f}°{RST}")
        for w in r.get("warnings", []):
            print(f"    {YELLOW}警告 {w}{RST}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-datum", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_args(parser)
    parser.add_argument("-o", "--out", help="输出基准 JSON")
    parser.add_argument("--show", metavar="FILE", help="打印已有基准后退出")
    parser.add_argument("--seconds", type=float, default=30.0, help="静置采样时长，默认 30")
    parser.add_argument("--settle", type=float, default=3.0,
                        help="正式采样前先丢掉几秒，等手放开的余震过去，默认 3")
    parser.add_argument("--wait", type=float, default=20.0, help="等首帧的超时，默认 20 s")
    parser.add_argument("--datum-mode", dest="datum_mode", default=dt.DEFAULT_MODE,
                        choices=dt.MODES,
                        help="归零方式，默认 mount（静置时输出单位阵，且保持重力对齐）；"
                             "level = 台面标定，只留跨上电有效的倾角修正")
    parser.add_argument("--reuse", metavar="FILE",
                        help="沿用台面上标好的 level 基准，本次只重新定位置和偏航")
    parser.add_argument("--device-id", action="append", default=None,
                        help="只给指定设备归零，如 3b:db:ef:e3。可重复")
    parser.add_argument("--no-z-up", dest="z_up", action="store_false", default=True,
                        help="不换成 +Z 朝上，保留设备原生的 +Y 朝上")
    parser.add_argument("--body-axes", default="Y,-X,Z",
                        help="机体系轴重映射，要和后面采集时用的一致")
    parser.add_argument("--force", action="store_true",
                        help="即使没通过静置检查也写文件（不建议）")
    args = parser.parse_args(argv)

    if args.show:
        return _show(args.show)
    if not args.out:
        parser.error("要么 -o 写基准，要么 --show 看基准")

    body_remap = None
    if args.body_axes:
        try:
            from ..frames import parse_axis_remap
            body_remap = parse_axis_remap(args.body_axes)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        except ImportError as exc:
            print(f"{exc}\n（不想装 numpy 就用 --body-axes ''）", file=sys.stderr)
            return 2

    reuse: dict = {}
    if args.reuse:
        try:
            reuse = dt.load(args.reuse, z_up=args.z_up, body_axes=args.body_axes)
        except (OSError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"沿用台面标定 {args.reuse}", file=sys.stderr)
        for _d in reuse.values():
            print(f"  {_d.summary()}", file=sys.stderr)

    wanted = set(args.device_id) if args.device_id else None

    try:
        dongle = open_dongle(args)
    except (OSError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        return 2

    enabler = TrackingEnabler(dongle, args)
    buckets: dict[str, list] = {}
    macs: dict[str, bytes] = {}

    def convert(pose):
        pos, rot = pose.pos, pose.rot
        if args.z_up:
            from ..frames import to_z_up_position, to_z_up_quat
            pos, rot = tuple(to_z_up_position(pos)), tuple(to_z_up_quat(rot))
        if body_remap is not None:
            from ..frames import remap_body_quat
            rot = tuple(remap_body_quat(rot, body_remap))
        return pos, rot

    def pump(deadline: float, collect: bool) -> None:
        """读到 deadline 为止。collect=False 时只保活，不留样本。"""
        while time.monotonic() < deadline:
            for report in dongle.read_reports(timeout=0.2):
                enabler(report.mac)
                if not report.is_pose:
                    continue
                pose = p.decode_pose(report.payload)
                if pose is None:
                    continue
                key = p.device_id_str(report.mac)
                if wanted is not None and key not in wanted:
                    continue
                macs[key] = report.mac
                if collect:
                    pos, rot = convert(pose)
                    buckets.setdefault(key, []).append(
                        dt.Sample(time.monotonic_ns(), pos, rot, pose.status))

    up = "+Z" if args.z_up else "+Y"
    what = "台面标定倾角" if args.datum_mode == "level" else "静置定原点"
    print(f"{BOLD}{what}{RST}  模式 {'mount(沿用)' if reuse else args.datum_mode}  "
          f"世界系 {up} 朝上"
          + (f"  机体系 {args.body_axes}" if args.body_axes else ""), file=sys.stderr)

    try:
        # ---- 等首帧
        print(f"等 tracker 上线（最多 {args.wait:.0f}s）…", end="", file=sys.stderr, flush=True)
        t0 = time.monotonic()
        while not macs and time.monotonic() - t0 < args.wait:
            pump(time.monotonic() + 0.5, collect=False)
        if not macs:
            print(f"\r{RED}超时：没收到位姿。检查配对，或加 --enable-tracking{RST}",
                  file=sys.stderr)
            return 3
        print(f"\r收到 {len(macs)} 台：" + "  ".join(sorted(macs)) + " " * 20, file=sys.stderr)

        # ---- 静置
        hint = ("把设备摆成**装到机器人上站直时的那个姿态**，然后别碰它。"
                if args.datum_mode == "level" else "把设备摆到参考位置，然后别碰它。")
        print(f"\n{BOLD}{YELLOW}{hint}{RST}", file=sys.stderr)
        end = time.monotonic() + args.settle
        while (left := end - time.monotonic()) > 0:
            print(f"\r  稳定中 {left:4.1f}s …", end="", file=sys.stderr, flush=True)
            pump(min(end, time.monotonic() + 0.2), collect=False)
        print(f"\r  {CYAN}开始采样{RST}" + " " * 20, file=sys.stderr)

        start = time.monotonic()
        end = start + args.seconds
        last = 0.0
        while time.monotonic() < end:
            pump(min(end, time.monotonic() + 0.2), collect=True)
            now = time.monotonic()
            if now - last >= 0.5:
                last = now
                live = "  ".join(f"{k[-5:]} {len(v)}" for k, v in sorted(buckets.items()))
                print(f"\r  [{now - start:5.1f}/{args.seconds:.0f}s] {live}",
                      end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}中断{RST}", file=sys.stderr)
        return 130
    finally:
        dongle.close()

    # ---- 检查 + 求解
    if not buckets:
        print(f"{RED}一帧都没采到{RST}", file=sys.stderr)
        return 3

    print(file=sys.stderr)
    results: dict = {}
    failed = False
    for key in sorted(buckets):
        samples = buckets[key]
        rep = dt.check_static(samples)
        mark = f"{GREEN}通过{RST}" if rep.ok else f"{RED}不合格{RST}"
        print(f"{BOLD}{key}{RST}  {mark}", file=sys.stderr)
        print(f"  {rep.describe()}", file=sys.stderr)
        for w in rep.warnings:
            print(f"  {YELLOW}警告 {w}{RST}", file=sys.stderr)
        for pr in rep.problems:
            print(f"  {RED}失败 {pr}{RST}", file=sys.stderr)
        if not rep.ok:
            failed = True
            if not args.force:
                continue
        try:
            d = dt.solve(samples, mode=args.datum_mode, z_up=args.z_up,
                         body_axes=args.body_axes, mac=macs[key], report=rep,
                         reuse=reuse.get(key), reused_from=args.reuse or "")
        except ValueError as exc:
            print(f"  {RED}{exc}{RST}", file=sys.stderr)
            failed = True
            continue
        if args.reuse and key not in reuse:
            print(f"  {YELLOW}{key} 不在台面标定文件里，这台按 {args.datum_mode} 现场算"
                  f"{RST}", file=sys.stderr)
        results[key] = d
        print(f"  {d.summary()}", file=sys.stderr)
        if d.reused_from:
            # 贴上台面标好的倾角修正后本来不该再有倾角，剩多少就是台面姿态和装机
            # 姿态差多少。绕重力那部分不算 —— 偏航反正每次现场重定。
            col = GREEN if d.residual_tilt_deg < 1.0 else YELLOW
            print(f"  {col}残余倾角 {d.residual_tilt_deg:.2f}° "
                  f"= 台面姿态与装机姿态的差（不含绕重力那部分）{RST}", file=sys.stderr)
            if d.residual_tilt_deg > 3.0:
                print(f"  {YELLOW}偏大。机器人没站直，或者装上去的角度和台面上摆的"
                      f"不一样 —— 输出静置时会歪这么多{RST}", file=sys.stderr)
        # 倾角本身不是毛病 —— 机体系怎么定的，倾角就多大。比如 Y,-X,Z 把 X 定在
        # 厚度方向，设备平躺时倾角天然就是 90° 上下。要紧的是这个倾角**去哪了**：
        # mount/full 会吸收掉，yaw/position 会原样留在输出里。
        elif args.datum_mode == "level":
            print(f"  {DIM}倾角修正 {d.tilt_deg:.2f}°"
                  f"（绕 {'/'.join(f'{v:+.2f}' for v in d.tilt_axis)}）"
                  f"—— 不含位置和偏航，跨上电有效{RST}", file=sys.stderr)
        elif args.datum_mode in ("mount", "full"):
            print(f"  {DIM}已吸收安装倾角 {d.tilt_deg:.1f}°"
                  f"（绕 {'/'.join(f'{v:+.2f}' for v in d.tilt_axis)}）{RST}", file=sys.stderr)
        elif d.tilt_deg > 30.0:
            print(f"  {YELLOW}安装倾角 {d.tilt_deg:.1f}° 会原样留在输出里 —— "
                  f"想让静置时读数是单位阵就用 --datum-mode mount{RST}", file=sys.stderr)
        if args.datum_mode == "full":
            print(f"  {YELLOW}full 模式把整个姿态塞进世界系那侧，输出系不再重力对齐。"
                  f"想要「静置即单位阵」又保住重力对齐，用 mount{RST}", file=sys.stderr)

    if failed and not args.force:
        print(f"\n{RED}有设备没通过静置检查，没有写文件。{RST}\n"
              f"设备真的没动过的话，多半是 SLAM 在漂 —— 等跟踪稳下来再来一次；"
              f"确实要存就加 --force。", file=sys.stderr)
        return 1
    if not results:
        return 1

    dt.save(args.out, results)
    print(f"\n{GREEN}写入 {len(results)} 台的基准 -> {args.out}{RST}", file=sys.stderr)
    if args.datum_mode == "level" and not reuse:
        print(f"{DIM}这是台面标定，只含倾角修正，跨上电有效。装到机器人上之后：\n"
              f"  vvu-datum --reuse {args.out} -o origin.json --seconds 10{RST}",
              file=sys.stderr)
    else:
        print(f"{DIM}后面这样用：\n"
              f"  vvu-capture -o run.csv --datum {args.out}\n"
              f"  vvu-viz --datum {args.out}{RST}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
