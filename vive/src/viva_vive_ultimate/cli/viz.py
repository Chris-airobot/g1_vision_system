"""浏览器里只读观察多台 tracker 的完整回传数据与 3D 位姿。

    vvu-viz                     只读，看当前在流的 tracker
    vvu-viz --device-info docs/tracker_devices.example.json

启动后打开终端里给出的网址。每台 tracker 显示成坐标系三轴、轨迹尾迹和独立数据面板。
面板显示原始 pose、IMU 字段、时间、status/btns 每一位、协议 position_valid，以及更
保守的 odometry_valid。默认不发送任何 Tracker 命令。

⚠️ 各台 tracker 的坐标系是**各自独立**的（各自建图各自定原点），
所以画在同一个场景里只是并排看，**相对位置没有物理意义**，除非先做共位标定
（见 viva_vive_ultimate.frames）。界面上有开关可以叠加标定后的结果。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .. import protocol as p
from .. import transforms as tf
from ..meshio import MESH_AXES_CAD_TO_RAW_BODY
from ..state import PoseValidity, TrackerState, evaluate_pose_validity
from ._common import TrackingEnabler, add_device_args, open_dongle

#: ``--mesh-preset`` 的默认值。注意是 CAD -> **原始**机体系：本工具在挂模型时会把
#: ``--body-axes`` 的换轴补偿回去（切换机体轴时模型保持不动），所以下拉菜单里的值
#: 始终相对原始机体系，而不是 ``meshio.MESH_AXES_CAD_TO_BODY`` 那个已合成的常量。
DEFAULT_MESH_PRESET = MESH_AXES_CAD_TO_RAW_BODY

#: 每台一个颜色（RGB 0-255）
COLORS = [(230, 60, 60), (60, 170, 230), (90, 210, 110),
          (240, 180, 50), (190, 110, 230), (240, 130, 190)]


@dataclass
class DeviceTraffic:
    """一台设备的外层 Dongle 报告统计；不保存 ACK 内容，避免泄露敏感信息。"""

    reports: int = 0
    pose_reports: int = 0
    heartbeats: int = 0
    acks: int = 0
    last_kind: str = "等待"
    last_pkt_idx: int | None = None
    heartbeat_idx: int | None = None
    heartbeat_btns: int | None = None
    last_seen: float = 0.0

    def note(self, report: p.Report, now: float) -> None:
        self.reports += 1
        self.last_pkt_idx = report.pkt_idx
        self.last_seen = now
        if report.is_pose:
            self.pose_reports += 1
            self.last_kind = "pose"
        elif report.is_heartbeat:
            self.heartbeats += 1
            self.last_kind = "heartbeat"
            if len(report.payload) >= 2:
                self.heartbeat_idx, self.heartbeat_btns = report.payload[:2]
        elif report.is_ack:
            self.acks += 1
            self.last_kind = "ack（内容隐藏）"
        else:
            self.last_kind = f"type=0x{report.type:03x}"


def _canonical_device_id(text: str) -> str:
    """把四字节 device ID 或完整六字节 MAC 统一成稳定 device ID。"""
    try:
        raw = bytes(int(part, 16) for part in text.strip().split(":"))
    except ValueError as exc:
        raise ValueError(f"设备键 {text!r} 不是十六进制 ID/MAC") from exc
    if len(raw) == 6:
        raw = raw[2:]
    if len(raw) != 4:
        raise ValueError(f"设备键 {text!r} 应为 4 字节 device ID 或 6 字节完整 MAC")
    return ":".join(f"{v:02x}" for v in raw)


def load_device_info(path: str | None) -> dict[str, dict[str, str]]:
    """载入人工维护的设备名称/厂商 SN；不会向 Tracker 查询。"""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table = payload.get("devices", payload) if isinstance(payload, dict) else None
    if not isinstance(table, dict):
        raise ValueError("device-info 顶层必须是对象，或包含 devices 对象")
    out: dict[str, dict[str, str]] = {}
    for key, value in table.items():
        did = _canonical_device_id(str(key))
        if isinstance(value, str):
            value = {"name": value}
        if not isinstance(value, dict):
            raise ValueError(f"设备 {key!r} 的信息必须是字符串或对象")
        out[did] = {
            "name": str(value.get("name", "")).strip(),
            "serial": str(value.get("serial", "")).strip(),
            "note": str(value.get("note", "")).strip(),
        }
    return out


def _bits8(value: int) -> str:
    return f"{value & 0xFF:08b}"


def _device_title(device_id: str, index: int, info: dict[str, str]) -> str:
    return f"{info.get('name') or 'Tracker'} · t{index} · {device_id}"


def render_device_markdown(st: TrackerState, traffic: DeviceTraffic,
                           validity: PoseValidity | None,
                           display_pos, display_rot,
                           info: dict[str, str]) -> str:
    """生成单台 Tracker 的完整数据面板，独立于 viser，便于测试。"""
    did = p.device_id_str(st.mac)
    serial = info.get("serial") or "只读 pose 流不提供厂商 SN"
    note = info.get("note") or "—"
    identity = (
        f"| 标识 | 值 |\n|---|---|\n"
        f"| 名称 | `{info.get('name') or f'Tracker t{st.index}'}` |\n"
        f"| 稳定 device ID | `{did}`（当前可用的类 SN 标识） |\n"
        f"| 厂商 SN | `{serial}` |\n"
        f"| 完整 MAC | `{p.mac_str(st.mac)}` |\n"
        f"| 当前槽位 | `t{st.index}`（跨配对可能变化） |\n"
        f"| 备注 | {note} |"
    )
    pose = st.pose
    if pose is None or validity is None:
        hb = "—" if traffic.heartbeat_idx is None else (
            f"idx={traffic.heartbeat_idx}, btns=0x{traffic.heartbeat_btns:02x} / "
            f"{_bits8(traffic.heartbeat_btns or 0)}")
        return (
            "### 🔴 ODOMETRY INVALID\n\n"
            "尚未收到完整 37 字节 pose；2 字节 heartbeat 不能用于里程计。\n\n"
            f"{identity}\n\n"
            "| 接收状态 | 值 |\n|---|---|\n"
            f"| 最近报告 | `{traffic.last_kind}` |\n"
            f"| heartbeat | `{hb}` |\n"
            f"| 报告/heartbeat/ACK | `{traffic.reports}` / `{traffic.heartbeats}` / "
            f"`{traffic.acks}` |\n"
            f"| Dongle pkt_idx | `{traffic.last_pkt_idx}` |"
        )

    valid_icon = "🟢" if validity.odometry_valid else "🔴"
    valid_text = "VALID" if validity.odometry_valid else "INVALID"
    reasons = "—" if not validity.reasons else "；".join(validity.reasons)
    position_valid = "TRUE" if validity.position_valid else "FALSE"
    odom_valid = "TRUE" if validity.odometry_valid else "FALSE"
    fresh = "TRUE" if validity.fresh else "FALSE"
    pose_finite = "TRUE" if validity.pose_finite else "FALSE"
    motion_finite = "TRUE" if validity.motion_finite else "FALSE"
    quat_valid = "TRUE" if validity.quaternion_valid else "FALSE"
    page = "高字节页" if pose.btns & 0x80 else "低字节页"
    if validity.quaternion_valid:
        roll, pitch, yaw = tf.to_euler_deg(tf.normalize(display_rot))
    else:
        roll = pitch = yaw = float("nan")
    acc_norm = tf.magnitude(pose.acc)
    gyro_norm = tf.magnitude(pose.rot_vel)
    pkt = "—" if traffic.last_pkt_idx is None else str(traffic.last_pkt_idx)

    return f"""### {valid_icon} ODOMETRY {valid_text}

{identity}

| 有效性 | 值 |
|---|---|
| `position_valid`（协议） | **{position_valid}** |
| `odometry_valid`（保守） | **{odom_valid}** |
| 数据新鲜 `< stale threshold` | `{fresh}`，age `{validity.age_ms:.1f} ms` |
| position/quaternion 有限 | `{pose_finite}` |
| accel/rot_vel 有限 | `{motion_finite}` |
| quaternion valid | `{quat_valid}`，norm `{validity.quaternion_norm:.6f}` |
| INVALID 原因 | {reasons} |

| 状态与标志位 | 原始值 | 解释 |
|---|---|---|
| `status_raw` | `0x{pose.status_raw:02x}` / `{_bits8(pose.status_raw)}` | 完整状态字节 |
| `status_hi` | `0x{pose.status_hi:x}` | 高半字节，语义未闭合 |
| `status` | `0x{pose.status:x}` | **{pose.status_name}**（低半字节） |
| `btns` | `0x{pose.btns:02x}` / `{_bits8(pose.btns)}` | {page}；bit7=`{(pose.btns >> 7) & 1}` |
| `btns b7…b0` | `{_bits8(pose.btns)}` | 每一位按原样显示，具体按键语义尚未闭合 |

| 原始 pose 回传 | X | Y | Z | W/模长 |
|---|---:|---:|---:|---:|
| `pos` m | {pose.pos[0]:+.6f} | {pose.pos[1]:+.6f} | {pose.pos[2]:+.6f} | — |
| `rot` xyzw | {pose.rot[0]:+.6f} | {pose.rot[1]:+.6f} | {pose.rot[2]:+.6f} | {pose.rot[3]:+.6f} |
| `acc` 原始值 | {pose.acc[0]:+.6f} | {pose.acc[1]:+.6f} | {pose.acc[2]:+.6f} | `norm={acc_norm:.6f}` |
| `rot_vel` 原始值 | {pose.rot_vel[0]:+.6f} | {pose.rot_vel[1]:+.6f} | {pose.rot_vel[2]:+.6f} | `norm={gyro_norm:.6f}` |

| 当前显示坐标 | X / roll | Y / pitch | Z / yaw |
|---|---:|---:|---:|
| position m | {display_pos[0]:+.6f} | {display_pos[1]:+.6f} | {display_pos[2]:+.6f} |
| Euler deg | {roll:+.3f} | {pitch:+.3f} | {yaw:+.3f} |

| 时序与链路 | 值 |
|---|---|
| Tracker `idx` | `{pose.idx}`（u8 回绕） |
| Dongle `pkt_idx` | `{pkt}` |
| `device_time` | `{pose.device_time}` ticks ≈ `{pose.device_time * 0.01:.2f} ms`（u16 回绕） |
| 收到 pose / RF 丢帧 / 重复帧 | `{st.count}` / `{st.lost}` / `{st.duplicates}` |
| 接收率 / RF 丢包率 | `{st.hz:.2f} Hz` / `{st.loss_percent:.3f}%` |
| 设备侧停顿 | `{st.stalls}` 次，累计 `{st.stalled_seconds * 1000:.2f} ms` |
| 外层报告 pose/heartbeat/ACK | `{traffic.pose_reports}` / `{traffic.heartbeats}` / `{traffic.acks}` |
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-viz", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_args(parser)
    parser.add_argument("--host", default="127.0.0.1",
                        help="网页监听地址，默认 127.0.0.1，仅本机可访问")
    parser.add_argument("--port", type=int, default=8080, help="网页端口")
    parser.add_argument("--trail", type=int, default=1500, help="轨迹尾迹保留点数")
    parser.add_argument("--axis-len", type=float, default=0.05,
                        help="tracker 坐标轴长度（米）。设备本体约 0.08 m，"
                             "默认 0.05 与之相称")
    parser.add_argument("--axis-ratio", type=float, default=60.0,
                        help="轴长 / 轴半径。数越大越细，默认 60")
    parser.add_argument("--world-axis-len", type=float, default=0.20,
                        help="世界坐标系轴长度（米）")
    parser.add_argument("--fps", type=float, default=30.0, help="界面刷新率")
    parser.add_argument("--data-fps", type=float, default=10.0,
                        help="数值面板刷新率，默认 10 Hz")
    parser.add_argument("--stale-ms", type=float, default=200.0,
                        help="超过此时间没收到 pose，odometry_valid 变 FALSE，默认 200 ms")
    parser.add_argument("--device-info",
                        help="可选 JSON：按稳定 device ID 配置名称、厂商 SN 和备注")
    parser.add_argument("--record", help="同时把所有 tracker 的位姿写入这个 CSV")
    parser.add_argument("--mesh", help="OBJ 模型路径，挂在每台 tracker 的坐标系下")
    parser.add_argument("--mesh-scale", type=float, default=0.001,
                        help="模型缩放。CAD 常见单位是 mm，默认 0.001 转米")
    parser.add_argument("--mesh-opacity", type=float, default=0.75)
    parser.add_argument("--mesh-preset", default=DEFAULT_MESH_PRESET,
                        help="CAD 轴 -> 设备**原始**机体轴，如 'X,-Y,-Z'（--body-axes 的换轴"
                             f"会自动补偿，不要用已合成的常量）。默认 {DEFAULT_MESH_PRESET}。"
                             "给了就用它，忽略 --mesh-rotate")
    parser.add_argument("--mesh-rotate", default="0,0,0",
                        help="模型相对 tracker 本体系的初始旋转，'rx,ry,rz' 度。"
                             "自由角度旋转；轴置换请优先用 --mesh-preset")
    parser.add_argument("--body-axes", default="Y,-X,Z",
                        help="机体系轴重映射，如 'Z,-X,-Y' 表示 新X=旧Z、新Y=-旧X、"
                             "新Z=-旧Y。只改坐标轴，模型会自动补偿保持不动")
    parser.add_argument("--z-up", action="store_true", default=None,
                        help="把世界系从设备的 +Y 朝上换成 +Z 朝上（IMU/机器人常规约定）。"
                             "配 --datum 时不写就跟随基准文件")
    parser.add_argument("--datum", help="vvu-datum 定的基准 JSON，把位姿归到那个原点")
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.data_fps <= 0 or args.stale_ms <= 0:
        parser.error("--fps、--data-fps 和 --stale-ms 必须大于 0")

    try:
        device_info = load_device_info(args.device_info)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"读取 --device-info 失败：{exc}", file=sys.stderr)
        return 2

    try:
        import viser
        import numpy as np
    except ImportError:
        print("需要 viser：pip install viser", file=sys.stderr)
        return 2

    try:
        dongle = open_dongle(args)
    except (OSError, PermissionError) as exc:
        print(exc, file=sys.stderr)
        return 2

    datums: dict = {}
    if args.datum:
        from ..datum import load as load_datum
        try:
            datums = load_datum(args.datum)
        except (OSError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        if args.z_up is None:            # 没显式指定就跟随基准
            args.z_up = next(iter(datums.values())).z_up
        # 基准是在某个机体系约定下算出来的，换了约定就不成立 —— 锁死下拉框
        args.body_axes = next(iter(datums.values())).body_axes
        print(f"基准 {args.datum}", file=sys.stderr)
        for _d in datums.values():
            print(f"  {_d.summary()}", file=sys.stderr)
    args.z_up = bool(args.z_up)

    conv = None
    if args.z_up:
        from ..frames import to_z_up_position, to_z_up_quat
        conv = (to_z_up_position, to_z_up_quat)

    def world_pos(mac, pos):
        """设备世界系 -> 显示用坐标（z_up + 基准归零）。"""
        pos = conv[0](pos) if conv else pos
        d = datums.get(p.device_id_str(mac)) if datums else None
        return d.apply(pos, (0.0, 0.0, 0.0, 1.0))[0] if d else pos

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True
    server.scene.world_axes.axes_length = args.world_axis_len
    server.scene.world_axes.axes_radius = args.world_axis_len / args.axis_ratio
    grid = server.scene.add_grid("/grid", width=6.0, height=6.0,
                                 cell_size=0.25, section_size=1.0)

    gui_info = server.gui.add_markdown("等待姿态数据…")
    gui_trail = server.gui.add_checkbox("显示轨迹", True)
    gui_clear = server.gui.add_button("清空轨迹")
    with server.gui.add_folder("VALID 判定说明"):
        server.gui.add_markdown(
            "- `position_valid`：设备协议值，`OK` 和 `RECENTLY_LOST` 都为 TRUE；"
            "后者位置可能冻结。\n"
            "- `odometry_valid`：仅当 status=`OK`、pose 未超过 `--stale-ms`、"
            "位置/四元数有限且四元数范数正常时为 TRUE。\n"
            "- INVALID pose 仍显示数值用于诊断，但不会追加到轨迹。\n"
            "- pose 流不含厂商 SN；默认显示稳定 device ID。可通过 `--device-info`"
            "人工补充铭牌 SN，全程不会查询或写入 Tracker。")

    from ..frames import all_axis_remaps as _all_remaps
    _body_opts = ["(不变)"] + _all_remaps()
    with server.gui.add_folder("机体系轴重映射"):
        server.gui.add_markdown(
            "三项依次是**新的 X / Y / Z 轴指向旧机体系的哪个轴**。\n\n"
            "只改坐标轴朝向，模型会自动补偿保持不动。")
        _binit = args.body_axes if args.body_axes in _body_opts else _body_opts[0]
        gui_body = server.gui.add_dropdown(
            "新轴 = 旧轴", _body_opts, initial_value=_binit)
        if datums:
            gui_body.disabled = True
            server.gui.add_markdown(
                "⚠️ 已锁定 —— 基准是在这个机体系下算出来的，换了就不成立。"
                "想换轴请用新的机体系重新跑 `vvu-datum`。")
        gui_body_dump = server.gui.add_button("打印机体系设置")

        @gui_body_dump.on_click
        def _(_e) -> None:
            print(f"\n机体系重映射: --body-axes '{gui_body.value}'"
                  if gui_body.value != _body_opts[0] else
                  "\n机体系重映射: 不变", flush=True)

    with server.gui.add_folder("坐标轴外观"):
        gui_axis_len = server.gui.add_slider("tracker 轴长 cm", 1.0, 30.0, 0.5,
                                             args.axis_len * 100)
        gui_world_len = server.gui.add_slider("世界轴长 cm", 2.0, 100.0, 1.0,
                                              args.world_axis_len * 100)
        gui_ratio = server.gui.add_slider("细度（长/半径）", 10.0, 200.0, 5.0,
                                          args.axis_ratio)
        gui_world_show = server.gui.add_checkbox("显示世界坐标系", True)
        gui_grid_show = server.gui.add_checkbox("显示网格", True)

    mesh_data = None
    gui_mesh = gui_mx = gui_my = gui_mz = gui_rx = gui_ry = gui_rz = None
    if args.mesh:
        from ..meshio import bounds, load_obj
        print(f"加载模型 {args.mesh} …", flush=True)
        mv, mf = load_obj(args.mesh, scale=args.mesh_scale)
        lo, hi, sz = bounds(mv)
        print(f"  顶点 {len(mv)}  三角面 {len(mf)}  尺寸 {(sz*1000).round(1)} mm", flush=True)
        mesh_data = (mv, mf)
        with server.gui.add_folder("模型对齐"):
            server.gui.add_markdown(
                "CAD 原点/朝向与 tracker 的位姿原点未必一致。\n\n"
                "拿着 tracker 转动，用下面的滑块把模型转到与实物一致。")
            from ..frames import all_axis_remaps
            gui_mesh = server.gui.add_checkbox("显示模型", True)
            _remaps = ["(用下面的滑块)"] + all_axis_remaps()
            _init = args.mesh_preset if args.mesh_preset in _remaps else _remaps[0]
            gui_preset = server.gui.add_dropdown(
                "CAD 轴 -> 本体轴（24 种全枚举）", _remaps, initial_value=_init)
            try:
                r0 = [float(x) for x in args.mesh_rotate.split(",")]
                assert len(r0) == 3
            except (ValueError, AssertionError):
                print("--mesh-rotate 格式应为 'rx,ry,rz'，已忽略", flush=True)
                r0 = [0.0, 0.0, 0.0]
            gui_rx = server.gui.add_slider("绕X °", -180.0, 180.0, 1.0, r0[0])
            gui_ry = server.gui.add_slider("绕Y °", -180.0, 180.0, 1.0, r0[1])
            gui_rz = server.gui.add_slider("绕Z °", -180.0, 180.0, 1.0, r0[2])
            gui_mx = server.gui.add_slider("移X mm", -100.0, 100.0, 1.0, 0.0)
            gui_my = server.gui.add_slider("移Y mm", -100.0, 100.0, 1.0, 0.0)
            gui_mz = server.gui.add_slider("移Z mm", -100.0, 100.0, 1.0, 0.0)
            gui_dump = server.gui.add_button("打印当前变换")

            @gui_dump.on_click
            def _(_e) -> None:
                if gui_preset.value != _remaps[0]:
                    print(f"\n模型对齐: CAD轴->本体轴 = {gui_preset.value!r}"
                          f"   平移 = ({gui_mx.value:.0f}, {gui_my.value:.0f}, "
                          f"{gui_mz.value:.0f}) mm", flush=True)
                else:
                    print(f"\n模型对齐: 绕XYZ = ({gui_rx.value:.0f}, {gui_ry.value:.0f}, "
                          f"{gui_rz.value:.0f})°   平移 = ({gui_mx.value:.0f}, "
                          f"{gui_my.value:.0f}, {gui_mz.value:.0f}) mm", flush=True)

    recorder = None
    if args.record:
        from ..recorder import CsvRecorder
        recorder = CsvRecorder(args.record)
        print(f"录制中 -> {args.record}", flush=True)

    # 全部按稳定 device ID 建索引，不能用可能跨配对变化的 tracker slot。
    states: dict[str, TrackerState] = {}
    traffic: dict[str, DeviceTraffic] = {}
    trails: dict[str, deque] = {}
    frames: dict[str, object] = {}
    labels: dict[str, object] = {}
    lines: dict[str, object] = {}
    meshes: dict[str, object] = {}
    mesh_frames: dict[str, object] = {}
    gui_device_folders: dict[str, object] = {}
    gui_device_markdown: dict[str, object] = {}
    lock = threading.Lock()
    stop = threading.Event()

    @gui_clear.on_click
    def _(_evt) -> None:
        with lock:
            for t in trails.values():
                t.clear()

    def reader() -> None:
        enabler = TrackingEnabler(dongle, args)
        while not stop.is_set():
            for report in dongle.read_reports(timeout=0.2):
                enabler(report.mac)
                now = time.monotonic()
                did = p.device_id_str(report.mac)
                new_device = False
                with lock:
                    if did not in states:
                        new_device = True
                        states[did] = TrackerState(report.tracker_index, report.mac)
                        traffic[did] = DeviceTraffic()
                        trails[did] = deque(maxlen=args.trail)
                    traffic[did].note(report, now)
                if new_device:
                    info = device_info.get(did, {})
                    print(f"[发现 Tracker] {info.get('name') or f't{report.tracker_index}'}  "
                          f"device_id={did}  MAC={p.mac_str(report.mac)}", flush=True)
                if not report.is_pose:
                    continue
                pose = p.decode_pose(report.payload)
                if pose is None:
                    continue
                if recorder is not None:
                    recorder.write(report.tracker_index, report.mac, pose)
                with lock:
                    states[did].update(pose, pkt_idx=report.pkt_idx,
                                       recv_monotonic=now)
                    # RECENTLY_LOST 虽然 protocol position_valid=True，但位置会冻结；
                    # 轨迹只接受保守 odometry_valid。
                    validity = evaluate_pose_validity(
                        pose, age_s=0.0, stale_after_s=args.stale_ms / 1000.0)
                    if validity.odometry_valid:
                        trails[did].append(world_pos(report.mac, pose.pos))

    threading.Thread(target=reader, daemon=True).start()
    display_host = "localhost" if args.host in ("127.0.0.1", "localhost") else args.host
    print(f"\n打开浏览器： http://{display_host}:{args.port}\n"
          f"监听 {args.host}；默认只允许本机访问\nCtrl-C 结束\n", flush=True)

    next_data_draw = 0.0
    try:
        while True:
            time.sleep(1.0 / args.fps)
            now = time.monotonic()
            redraw_data = now >= next_data_draw
            if redraw_data:
                next_data_draw = now + 1.0 / args.data_fps
            with lock:
                snapshot = [(did, st, traffic[did], list(trails[did]))
                            for did, st in states.items()]
            snapshot.sort(key=lambda item: (item[1].index, item[0]))
            rows = []
            for did, st, link, trail in snapshot:
                pose = st.pose
                info = device_info.get(did, {})
                if did not in gui_device_markdown:
                    folder = server.gui.add_folder(_device_title(did, st.index, info))
                    gui_device_folders[did] = folder
                    with folder:
                        gui_device_markdown[did] = server.gui.add_markdown(
                            "等待完整 pose 数据…")
                if pose is None:
                    if redraw_data:
                        gui_device_markdown[did].content = render_device_markdown(
                            st, link, None, (0.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0, 1.0), info)
                        rows.append(
                            f"🔴 **{info.get('name') or f't{st.index}'}** `{did}` — "
                            f"ODOMETRY INVALID · {link.last_kind} · 尚无完整 pose")
                    continue
                validity = st.validity(stale_after_s=args.stale_ms / 1000.0,
                                       now=now)
                assert validity is not None
                color = COLORS[st.index % len(COLORS)]
                safe_did = did.replace(":", "_")
                name = f"/tracker_{safe_did}"
                # 数值坏掉时仍把原始值放进面板，但不再送入 3D 变换链。
                can_transform = validity.pose_finite and validity.quaternion_valid
                rot = (conv[1](pose.rot) if conv else np.asarray(pose.rot, dtype=float)) \
                    if can_transform else np.asarray(pose.rot, dtype=float)
                remap_inv_q = None
                if can_transform and gui_body.value != _body_opts[0]:
                    from ..frames import (matrix_to_quat as _m2q2,
                                          parse_axis_remap as _par2, remap_body_quat)
                    _R = _par2(gui_body.value)
                    rot = remap_body_quat(rot, _R)          # 右乘 = 机体系换轴
                    remap_inv_q = _m2q2(_R.T)
                pos = world_pos(st.mac, pose.pos) if can_transform else pose.pos
                _d = datums.get(p.device_id_str(st.mac)) if datums else None
                if can_transform and _d is not None:
                    rot = _d.apply((0.0, 0.0, 0.0), rot)[1]
                # viser 的四元数是 (w, x, y, z)；设备给的是 (x, y, z, w)
                wxyz = (rot[3], rot[0], rot[1], rot[2])
                tag = "VALID" if validity.odometry_valid else "INVALID"
                label_text = f"{info.get('name') or f't{st.index}'}  {did}  {tag}"
                if can_transform and did not in frames:
                    frames[did] = server.scene.add_frame(
                        name, wxyz=wxyz, position=pos,
                        axes_length=gui_axis_len.value / 100,
                        axes_radius=gui_axis_len.value / 100 / gui_ratio.value)
                    labels[did] = server.scene.add_label(f"{name}/label", label_text)
                elif can_transform:
                    frames[did].wxyz = wxyz
                    frames[did].position = pos
                    frames[did].axes_length = gui_axis_len.value / 100
                    frames[did].axes_radius = (gui_axis_len.value / 100
                                               / gui_ratio.value)
                if did in labels:
                    labels[did].text = label_text

                if mesh_data is not None and can_transform:
                    import numpy as np
                    # 模型挂在 tracker 坐标系下的一个可调子坐标系里
                    from ..frames import (matrix_to_quat as _m2q, parse_axis_remap as _par,
                                          quat_multiply as _qm)
                    if gui_preset.value != _remaps[0]:
                        _q = _m2q(_par(gui_preset.value))
                        qm = (_q[3], _q[0], _q[1], _q[2])          # (w,x,y,z)
                        rx = ry = rz = 0.0
                    else:
                        rx, ry, rz = (math.radians(g.value)
                                      for g in (gui_rx, gui_ry, gui_rz))
                    if gui_preset.value == _remaps[0]:
                        cx, sx = math.cos(rx/2), math.sin(rx/2)
                        cy, sy = math.cos(ry/2), math.sin(ry/2)
                        cz, sz_ = math.cos(rz/2), math.sin(rz/2)
                        qm = (cx*cy*cz + sx*sy*sz_, sx*cy*cz - cx*sy*sz_,
                              cx*sy*cz + sx*cy*sz_, cx*cy*sz_ - sx*sy*cz)  # (w,x,y,z)
                    if remap_inv_q is not None:
                        # 坐标轴换了轴，模型要补偿回去才能保持不动
                        qq = _qm(np.array([remap_inv_q[0], remap_inv_q[1],
                                           remap_inv_q[2], remap_inv_q[3]]),
                                 np.array([qm[1], qm[2], qm[3], qm[0]]))
                        qm = (qq[3], qq[0], qq[1], qq[2])
                    off = (gui_mx.value/1000, gui_my.value/1000, gui_mz.value/1000)
                    if did not in mesh_frames:
                        mesh_frames[did] = server.scene.add_frame(
                            f"{name}/mesh", wxyz=qm, position=off,
                            show_axes=False)
                        meshes[did] = server.scene.add_mesh_simple(
                            f"{name}/mesh/obj", vertices=mesh_data[0],
                            faces=mesh_data[1], color=color,
                            opacity=args.mesh_opacity)
                    else:
                        mesh_frames[did].wxyz = qm
                        mesh_frames[did].position = off
                    meshes[did].visible = gui_mesh.value

                if gui_trail.value and len(trail) >= 2:
                    import numpy as np
                    pts = np.asarray(trail, dtype=float)
                    seg = np.stack([pts[:-1], pts[1:]], axis=1)
                    if did in lines:
                        lines[did].remove()
                    lines[did] = server.scene.add_line_segments(
                        f"/trail_{safe_did}", points=seg,
                        colors=np.tile(np.array(color, dtype=np.uint8),
                                       (len(seg), 2, 1)),   # -> (N, 2, 3)
                        line_width=2.0)
                elif not gui_trail.value and did in lines:
                    lines[did].remove()
                    del lines[did]

                if redraw_data:
                    gui_device_markdown[did].content = render_device_markdown(
                        st, link, validity, pos, rot, info)
                    icon = "🟢" if validity.odometry_valid else "🔴"
                    rows.append(
                        f"{icon} **{info.get('name') or f't{st.index}'}** `{did}` — "
                        f"ODOMETRY {'VALID' if validity.odometry_valid else 'INVALID'} · "
                        f"status `{pose.status_name}` (`0x{pose.status_raw:02x}`) · "
                        f"age {validity.age_ms:.1f} ms · {st.hz:.1f} Hz · "
                        f"丢包 {st.loss_percent:.2f}%")
            if redraw_data:
                gui_info.content = ("### 在线 Tracker\n\n" + "\n\n".join(rows)) if rows else (
                    "等待 Tracker 报告…\n\n程序以只读方式运行；请先在 VIVE Hub 中确认设备"
                    "已经开始输出 pose。")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        time.sleep(0.3)
        dongle.close()
        if recorder is not None:
            recorder.close()
            print(f"\n已录制 {recorder.rows} 行 -> {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
