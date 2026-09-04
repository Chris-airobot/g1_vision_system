"""采集姿态到 CSV。

    vvu-capture --out run.csv --seconds 60
    vvu-capture --out run.csv --enable-tracking
    vvu-capture --out run.csv --datum datum.json     # 归到 vvu-datum 定的原点
"""

from __future__ import annotations

import argparse
import sys
import time

from .. import protocol as p
from ..recorder import CsvRecorder
from ..state import TrackerState
from ._common import TrackingEnabler, add_device_args, open_dongle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-capture", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_args(parser)
    parser.add_argument("-o", "--out", required=True, help="输出 CSV 路径")
    parser.add_argument("--seconds", type=float, default=0, help="采集时长；0 = 直到 Ctrl-C")
    parser.add_argument("--quiet", action="store_true", help="不打印进度")
    parser.add_argument("--z-up", action="store_true", default=None,
                        help="写入 CSV 前把世界系换成 +Z 朝上（IMU/机器人常规约定）。"
                             "配 --datum 时不写就跟随基准文件")
    parser.add_argument("--datum", help="vvu-datum 定的基准 JSON，把位姿归到那个原点")
    parser.add_argument("--body-axes", default="Y,-X,Z",
                        help="机体系轴重映射，如 'Z,-X,-Y'（新X=旧Z, 新Y=-旧X, "
                             "新Z=-旧Y）。只影响姿态，位置表达在世界系不受影响")
    args = parser.parse_args(argv)

    datums: dict = {}
    if args.datum:
        from ..datum import load as load_datum
        try:
            probe = load_datum(args.datum)
        except (OSError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        if args.z_up is None:            # 没显式指定就跟随基准，省得约定对不上
            args.z_up = next(iter(probe.values())).z_up
        try:
            datums = load_datum(args.datum, z_up=args.z_up, body_axes=args.body_axes)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"基准 {args.datum}", file=sys.stderr)
        for d in datums.values():
            print(f"  {d.summary()}", file=sys.stderr)
    args.z_up = bool(args.z_up)

    body_remap = None
    if args.body_axes:
        from ..frames import parse_axis_remap
        try:
            body_remap = parse_axis_remap(args.body_axes)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"机体系重映射 {args.body_axes}", file=sys.stderr)

    try:
        dongle = open_dongle(args)
    except (OSError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        return 2

    enabler = TrackingEnabler(dongle, args)
    states: dict[int, TrackerState] = {}
    no_datum: set = set()
    started = time.monotonic()
    last_log = started

    with CsvRecorder(args.out) as rec:
        if not args.quiet:
            print(f"采集中 -> {args.out}   Ctrl-C 停止", file=sys.stderr, flush=True)
        try:
            while not (args.seconds and time.monotonic() - started >= args.seconds):
                for report in dongle.read_reports(timeout=0.5):
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
                    if args.z_up:
                        from ..frames import to_z_up_position, to_z_up_quat
                        pose = pose._replace(pos=tuple(to_z_up_position(pose.pos)),
                                             rot=tuple(to_z_up_quat(pose.rot)))
                    if body_remap is not None:
                        from ..frames import remap_body_quat
                        pose = pose._replace(
                            rot=tuple(remap_body_quat(pose.rot, body_remap)))
                    if datums:
                        d = datums.get(p.device_id_str(report.mac))
                        if d is None:
                            if report.mac not in no_datum:
                                no_datum.add(report.mac)
                                print(f"\n[警告] {p.device_id_str(report.mac)} 不在基准文件里，"
                                      f"这台按原始世界系写出", file=sys.stderr)
                        else:
                            pos, rot = d.apply(pose.pos, pose.rot)
                            pose = pose._replace(pos=pos, rot=rot)
                    rec.write(tid, report.mac, pose)

                now = time.monotonic()
                if not args.quiet and now - last_log >= 1.0:
                    last_log = now
                    parts = [f"t{s.index} {s.hz:5.1f}Hz {s.pose.status_name if s.pose else '-'}"
                             f" 丢{s.loss_percent:.1f}%" for s in states.values()]
                    print(f"\r[{now - started:6.1f}s] {rec.rows:7d} 行   " + "  ".join(parts),
                          end="", file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            dongle.close()

        elapsed = time.monotonic() - started
        print(file=sys.stderr)
        print(f"写入 {rec.rows} 行 / {elapsed:.1f}s -> {args.out}", file=sys.stderr)
        if datums:
            from ..datum import save as save_datum
            side = str(args.out) + ".datum.json"
            save_datum(side, datums)
            print(f"  基准副本 -> {side}（这条轨迹是按它归零的）", file=sys.stderr)
        for st in sorted(states.values(), key=lambda s: s.index):
            print(f"  tracker {st.index} {st.mac_str}: {st.count} 帧 "
                  f"({st.count / elapsed:.1f} Hz)  丢包 {st.lost} ({st.loss_percent:.2f}%)  "
                  f"行程 {st.path_length:.3f} m", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
