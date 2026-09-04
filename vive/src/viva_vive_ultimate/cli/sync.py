"""同时录 tracker 与 Intel RealSense D435i，落到同一个主机时基上。

    vvu-sync --seconds 60 --out-dir run1
    vvu-sync --seconds 60 --out-dir run1 --align        # 录完直接算偏移

产出：
    run1/utk.csv      tracker：device_time、拟合出的主机时刻、姿态、角速度
    run1/d435i_imu.csv     D435i 陀螺/加速度：主机时刻、三轴
    run1/d435i_frames.csv  深度/RGB 每帧的主机时刻与帧号（640x480 @30fps）
    run1/meta.json    时钟拟合参数、两边的时基说明

D435i 用回调 API 采集，深度/RGB 默认 640x480 @30fps，IMU 取设备支持的最高速率。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from .. import protocol as p
from ..clocksync import ClockSync
from ._common import TrackingEnabler, add_device_args, open_dongle


def clock_pair() -> tuple[int, int]:
    """同时取 CLOCK_MONOTONIC 与 CLOCK_REALTIME，用于两个时基互转。

    librealsense 的 global time 用 ``system_clock``（CLOCK_REALTIME，毫秒），
    而本项目内部用 CLOCK_MONOTONIC（纳秒）。两者要靠这一对采样对齐。
    """
    mono = time.monotonic_ns()
    real = time.clock_gettime_ns(time.CLOCK_REALTIME)
    mono2 = time.monotonic_ns()
    return (mono + mono2) // 2, real


class RealsenseRecorder:
    """录 D435i：640x480 深度 + RGB + 最高速率 IMU。

    用**回调 API** 而不是 ``wait_for_frames``：单一 pipeline 里同时开视频和 IMU 时，
    frameset 聚合会把 400 Hz 的 IMU 压到视频的 30 fps。回调让每一帧到达即交付。

    回调跑在 librealsense 自己的线程上，所以只往内存里塞，收工再统一落盘，
    避免 IO 阻塞采集。
    """

    def __init__(self, out_dir: Path, width: int = 640, height: int = 480,
                 video_fps: int = 30) -> None:
        self.out_dir = out_dir
        self.width = width
        self.height = height
        self.video_fps = video_fps
        self.error: str | None = None
        self.domains: set[str] = set()
        self.rates: dict[str, int] = {}
        self.imu: list[tuple] = []
        self.frames: list[tuple] = []
        self._pipeline = None
        self._lock = threading.Lock()
        self._mono0 = 0
        self._real0 = 0

    # -------------------------------------------------------------- 采集

    def _handle(self, frame) -> None:
        import pyrealsense2 as rs
        ts_ms = frame.get_timestamp()
        domain = str(frame.get_frame_timestamp_domain())
        # global time 是 CLOCK_REALTIME 毫秒 -> 换成本项目统一用的 monotonic 纳秒
        mono_ns = int(ts_ms * 1e6) - self._real0 + self._mono0

        motion = frame.as_motion_frame()
        if motion:
            d = motion.get_motion_data()
            with self._lock:
                self.domains.add(domain)
                self.imu.append((mono_ns, ts_ms, domain,
                                 motion.get_profile().stream_name(), d.x, d.y, d.z))
            return

        video = frame.as_video_frame()
        if video:
            with self._lock:
                self.domains.add(domain)
                self.frames.append((mono_ns, ts_ms, domain,
                                    video.get_profile().stream_name(),
                                    frame.get_frame_number(),
                                    video.get_width(), video.get_height()))

    def _callback(self, frame) -> None:
        try:
            fs = frame.as_frameset()
            if fs:
                for f in fs:
                    self._handle(f)
            else:
                self._handle(frame)
        except Exception as exc:                       # noqa: BLE001
            if self.error is None:
                self.error = f"回调异常: {exc}"

    def start(self) -> bool:
        try:
            import pyrealsense2 as rs
        except ImportError:
            self.error = "pyrealsense2 未安装：pip install pyrealsense2"
            return False
        try:
            # 采样率因固件而异（这台 accel/gyro 都到 400，老固件是 accel 63/250），
            # 查设备实际支持的档位取最高，别硬编码。
            devices = rs.context().query_devices()
            if len(devices) == 0:
                self.error = "没检测到 RealSense 设备"
                return False
            rates: dict = {}
            for sensor in devices[0].query_sensors():
                for prof in sensor.get_stream_profiles():
                    if prof.stream_type() in (rs.stream.gyro, rs.stream.accel):
                        st = prof.stream_type()
                        rates[st] = max(rates.get(st, 0), prof.fps())
            self.rates = {str(k).replace("stream.", ""): v for k, v in rates.items()}
            self.rates["depth"] = self.video_fps
            self.rates["color"] = self.video_fps

            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, self.width, self.height,
                              rs.format.z16, self.video_fps)
            cfg.enable_stream(rs.stream.color, self.width, self.height,
                              rs.format.rgb8, self.video_fps)
            for stream in (rs.stream.gyro, rs.stream.accel):
                if stream in rates:
                    cfg.enable_stream(stream, rs.format.motion_xyz32f, rates[stream])

            self._mono0, self._real0 = clock_pair()
            self._pipeline = rs.pipeline()
            profile = self._pipeline.start(cfg, self._callback)

            # 打开 global time：帧时间戳直接落在主机时钟域
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
        except Exception as exc:                       # noqa: BLE001
            self.error = f"D435i 启动失败: {exc}"
            return False
        return True

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:                          # noqa: BLE001
                pass
            self._pipeline = None

    # -------------------------------------------------------------- 落盘

    def flush(self) -> None:
        with self._lock:
            imu, frames = list(self.imu), list(self.frames)
        with open(self.out_dir / "d435i_imu.csv", "w", encoding="utf-8") as fh:
            fh.write("host_mono_ns,rs_timestamp_ms,domain,stream,x,y,z\n")
            for r in imu:
                fh.write(f"{r[0]},{r[1]:.6f},{r[2]},{r[3]},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f}\n")
        with open(self.out_dir / "d435i_frames.csv", "w", encoding="utf-8") as fh:
            fh.write("host_mono_ns,rs_timestamp_ms,domain,stream,frame_number,width,height\n")
            for r in frames:
                fh.write(f"{r[0]},{r[1]:.6f},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n")

    @property
    def imu_rows(self) -> int:
        return len(self.imu)

    @property
    def frame_rows(self) -> int:
        return len(self.frames)


def record_tracker(dongle, enabler, sync: ClockSync, path: Path,
                   stop: threading.Event) -> int:
    rows = 0
    with open(path, "w", buffering=1, encoding="utf-8") as fh:
        fh.write("host_mono_ns,fit_host_ns,device_time,ticks,tracker,idx,status,"
                 "x,y,z,qx,qy,qz,qw,wx,wy,wz\n")
        while not stop.is_set():
            for report in dongle.read_reports(timeout=0.2):
                enabler(report.mac)
                if not report.is_pose:
                    continue
                pose = p.decode_pose(report.payload)
                if pose is None:
                    continue
                now = time.monotonic_ns()
                ticks = sync.add(pose.device_time, now)
                fit_ns = sync.host_ns(ticks=ticks)
                fh.write(
                    f"{now},{'' if fit_ns is None else int(fit_ns)},"
                    f"{pose.device_time},{ticks},{report.tracker_index},{pose.idx},"
                    f"{pose.status_name},"
                    + ",".join(f"{v:.6f}" for v in pose.pos) + ","
                    + ",".join(f"{v:.6f}" for v in pose.rot) + ","
                    + ",".join(f"{v:.6f}" for v in pose.rot_vel) + "\n")
                rows += 1
    return rows


def run_alignment(out_dir: Path) -> None:
    """录完直接跑一次互相关，给出常量偏移。"""
    try:
        import csv

        import numpy as np

        from ..timealign import magnitude, split_half_check
    except ImportError as exc:
        print(f"对齐需要 numpy: {exc}", file=sys.stderr)
        return

    utk = list(csv.DictReader(open(out_dir / "utk.csv")))
    cam = [r for r in csv.DictReader(open(out_dir / "d435i_imu.csv"))
           if r["stream"] == "Gyro"]
    if len(utk) < 100 or len(cam) < 100:
        print("样本太少，跳过对齐", file=sys.stderr)
        return

    ut = np.array([int(r["fit_host_ns"]) for r in utk if r["fit_host_ns"]]) / 1e9
    uv = magnitude(*[[float(r[k]) for r in utk if r["fit_host_ns"]] for k in ("wx", "wy", "wz")])
    ct = np.array([int(r["host_mono_ns"]) for r in cam]) / 1e9
    cv = magnitude(*[[float(r[k]) for r in cam] for k in ("x", "y", "z")])

    print("\n=== 互相关标定 ===")
    try:
        sh = split_half_check(ut, uv, ct, cv)
    except ValueError as exc:
        print(f"  失败: {exc}")
        return
    print("  " + sh.summary())
    print(f"  相关 {sh.full.correlation:.3f}   重叠 {sh.full.overlap_s:.1f}s")
    if sh.consistent:
        print(f"\n  ✓ 常量偏移 = {sh.full.offset_ms:+.2f} ms")
        print(f"    把 utk.csv 的 fit_host_ns 加上这个值，就与 D435i 对齐了")
    else:
        print("\n  ✗ 前后半段不一致 —— 结果不可信。检查：")
        print("    · 标定时两个设备刚性绑在一起了吗")
        print("    · 晃动够不够（要随机急抖，别匀速画圈）")
        print("    · 重叠时长够不够（建议 >= 40 s）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-sync", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_args(parser)
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--no-realsense", action="store_true", help="只录 tracker")
    parser.add_argument("--align", action="store_true", help="录完直接算常量偏移")
    parser.add_argument("--window-s", type=float, default=60.0, help="时钟拟合窗口")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        dongle = open_dongle(args)
    except (OSError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        return 2

    stop = threading.Event()
    sync = ClockSync(window_s=args.window_s)
    rs_rec = None
    if not args.no_realsense:
        rs_rec = RealsenseRecorder(out_dir, args.width, args.height, args.video_fps)
        if not rs_rec.start():
            print(f"  {rs_rec.error}", file=sys.stderr)

    mono0, real0 = clock_pair()
    print(f"录制 {args.seconds:.0f}s -> {out_dir}/", file=sys.stderr)
    timer = threading.Timer(args.seconds, stop.set)
    timer.start()
    try:
        rows = record_tracker(dongle, TrackingEnabler(dongle, args), sync,
                              out_dir / "utk.csv", stop)
    except KeyboardInterrupt:
        stop.set()
        rows = -1
    finally:
        timer.cancel()
        stop.set()
        if rs_rec:
            rs_rec.stop()
            rs_rec.flush()
        dongle.close()

    fit = sync.fit
    meta = {
        "seconds": args.seconds,
        "utk_rows": rows,
        "utk_clock_fit": None if fit is None else {
            "slope_ns_per_tick": fit.slope_ns_per_tick,
            "offset_ns": fit.offset_ns,
            "ppm": fit.ppm,
            "samples": fit.samples,
            "span_s": fit.span_s,
            "residual_p50_ns": fit.residual_p50_ns,
            "residual_p95_ns": fit.residual_p95_ns,
        },
        "clock_pair": {"monotonic_ns": mono0, "realtime_ns": real0},
        "timebase": "两个 CSV 的 host_mono_ns / fit_host_ns 都是 CLOCK_MONOTONIC 纳秒",
        "d435i_imu_rows": None if rs_rec is None else rs_rec.imu_rows,
        "d435i_frame_rows": None if rs_rec is None else rs_rec.frame_rows,
        "d435i_resolution": [args.width, args.height],
        "d435i_domains": None if rs_rec is None else sorted(rs_rec.domains),
        "d435i_rates": None if rs_rec is None else rs_rec.rates,
        "d435i_error": None if rs_rec is None else rs_rec.error,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\ntracker {rows} 行", file=sys.stderr)
    if fit:
        print(f"  时钟拟合 {fit.slope_ns_per_tick:.4f} ns/tick ({fit.ppm:+.1f} ppm)  "
              f"残差 p50 {fit.residual_p50_ns/1000:.0f} µs", file=sys.stderr)
    if rs_rec:
        if rs_rec.error:
            print(f"  D435i: {rs_rec.error}", file=sys.stderr)
        else:
            print(f"  D435i  IMU {rs_rec.imu_rows} 行   视频帧 {rs_rec.frame_rows} 行"
                  f"   {args.width}x{args.height}", file=sys.stderr)
            print(f"         采样率 {rs_rec.rates}   时间戳域 {sorted(rs_rec.domains)}",
                  file=sys.stderr)
            if not any("global" in d.lower() for d in rs_rec.domains):
                print("  ⚠ 时间戳域不是 global_time，主机时刻不可靠 —— "
                      "检查固件是否支持 global_time_enabled", file=sys.stderr)

    if args.align and rs_rec and not rs_rec.error:
        run_alignment(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
