"""VIVE Ultimate Tracker 线上包格式。

两条链路的姿态载荷是**逐字节相同**的（已实测对拍验证）：

    dongle HID   report[0x0C:0x0C+0x25]
    WiFi `02 25` frame[2:0x27]

本模块只做纯解码，不碰 IO，不依赖任何第三方包。

包格式来源：
  - vive_ultimate_tracker_re/tracker_toybox  (dongle HID 头部与 ACK 语义)
  - Vive_Ultimate_tracker_Wifi_Solution      (status 低半字节、device_time)
  - UltimateTracker_FirmWare                 (HID 描述符、dongle 命令表交叉验证)
"""

from __future__ import annotations

import struct
from typing import NamedTuple

# ---------------------------------------------------------------- USB 身份

VID_VIVE = 0x0BB4
PID_DONGLE = 0x0350        # VIVE Wireless Dongle（app 模式）
PID_DONGLE_BOOTLOADER = 0x09E9
PID_TRACKER_USB = 0x06A3   # tracker 直连 USB

REPORT_SIZE = 64           # 报告描述符: 06 00 FF ... 95 40 81 00，无 report ID

# ---------------------------------------------------- dongle -> 主机 报告

CMD_TRACKER_DATA = 0x28    # tracker 转发数据
CMD_PAIR_EVENT = 0x18      # 配对 / 解除配对
CMD_RF_STATUS = 0x1E
CMD_RF_STATUS_NEW = 0x1D

TYPE_ACK = 0x101           # 载荷是 ASCII ACK 字符串
TYPE_POSE = 0x110          # 载荷是姿态（或 2 字节心跳）

#: dongle 报告头 —— cmd_id, pkt_idx, tracker_mac, type, data_len
HEADER = struct.Struct("<BH6sHB")

#: 姿态载荷 —— idx, btns, pos(f32x3), rot(f16x4), acc(f16x3), 角速度+device_time, status
POSE = struct.Struct("<BB12s8s6s8sB")
POSE_SIZE = POSE.size      # 0x25 = 37

#: WiFi 链路的 128 字节 `02 25` 帧里，姿态载荷从 offset 2 开始
WIFI_FRAME_MAGIC = b"\x02\x25"
WIFI_FRAME_SIZE = 128
WIFI_POSE_OFFSET = 2

# ------------------------------------------------------------ 状态语义

POSE_SYSTEM_NOT_READY = -1
POSE_NO_IMAGES_YET = 0
POSE_NOT_INITIALIZED = 1
POSE_OK = 2
POSE_LOST = 3
POSE_RECENTLY_LOST = 4

POSE_STATUS_NAMES = {
    POSE_SYSTEM_NOT_READY: "SYSTEM_NOT_READY",
    POSE_NO_IMAGES_YET: "NO_IMAGES_YET",
    POSE_NOT_INITIALIZED: "NOT_INITIALIZED",
    POSE_OK: "OK",
    POSE_LOST: "LOST",
    POSE_RECENTLY_LOST: "RECENTLY_LOST",
}

#: 位置有效的状态集合（与 WiFi 方案 `status_nibble in (2, 4)` 一致）
POSITION_VALID = frozenset({POSE_OK, POSE_RECENTLY_LOST})

#: device_time 的 tick 长度，由 789 ticks / 8 ms 反推，约 10 µs
DEVICE_TIME_TICK_SECONDS = 10e-6

#: 正常帧周期只有 787-790 与 886-888 两簇（分数分频抖动，平均 800）。
#: idx 连续却超过这个阈值，说明是**设备侧停顿** —— tracker 自己没生成位姿，
#: 而 idx 不为这些帧计数，所以基于 idx 的丢包统计看不见。
#: 实测只在 SLAM 丢追时出现，约每 3 秒一次、每次约 24 ms。
DEVICE_STALL_TICKS = 1000


class Pose(NamedTuple):
    """一帧姿态。位置单位米；四元数按 (x, y, z, w) 顺序读出。

    注意四元数**不是** SteamVR 约定——静置平放时 roll ≈ -90°。
    要送进 OpenVR 需要额外的手性变换，自己的程序建议用 tools 实测出映射。
    """

    idx: int              # 帧序号，逐帧 +1，按字节回绕，可用来数丢包
    btns: int             # 按键位；bit7=1 表示这是高字节页
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    acc: tuple[float, float, float]
    rot_vel: tuple[float, float, float]
    device_time: int      # u16，约 10 µs/tick，会回绕
    status: int           # 低半字节
    status_hi: int        # 高半字节，语义未闭合
    status_raw: int

    @property
    def status_name(self) -> str:
        return POSE_STATUS_NAMES.get(self.status, f"0x{self.status:02x}")

    @property
    def position_valid(self) -> bool:
        return self.status in POSITION_VALID


class Report(NamedTuple):
    """一条已解析的 dongle HID 报告。"""

    cmd_id: int
    pkt_idx: int
    mac: bytes
    type: int
    payload: bytes

    @property
    def tracker_index(self) -> int:
        """槽位号。跨会话可能变 —— 稳定标识用 :attr:`device_id`。"""
        return tracker_index(self.mac)

    @property
    def device_id(self) -> bytes:
        """跨会话稳定的设备标识（MAC 后 4 字节）。"""
        return device_id(self.mac)

    @property
    def is_pose(self) -> bool:
        return self.type == TYPE_POSE and len(self.payload) >= POSE_SIZE

    @property
    def is_heartbeat(self) -> bool:
        """只有 <idx><btns> 的 2 字节包 —— 连上了但 SLAM 没启动。"""
        return self.type == TYPE_POSE and len(self.payload) == 2

    @property
    def is_ack(self) -> bool:
        return self.type == TYPE_ACK


def tracker_index(mac: bytes) -> int:
    """tracker 槽位序号 = MAC 第 2 字节的低 4 位。

    ⚠️ **这是槽位号，不是设备身份。** 实测同一台 tracker 在不同配对情况下
    会拿到不同的槽位（``23:31:95:65:2e:67`` -> ``23:32:95:65:2e:67``，
    index 从 1 变成 2）。跨会话追踪同一台设备请用 :func:`device_id`。
    """
    return mac[1] & 0x0F


def device_id(mac: bytes) -> bytes:
    """设备身份 = MAC 后 4 字节，跨会话稳定。

    MAC 的前两字节是厂商前缀 + 槽位号，会变；后四字节才是这台设备的标识。
    """
    return bytes(mac[2:6])


def device_id_str(mac: bytes) -> str:
    return ":".join(f"{b:02x}" for b in device_id(mac))


def mac_str(mac: bytes) -> str:
    return ":".join(f"{b:02x}" for b in mac)


def parse_mac(text: str) -> bytes:
    return bytes(int(b, 16) for b in text.split(":"))


class PairEvent(NamedTuple):
    """cmd_id 0x18：配对 / 解除配对。"""

    mac: bytes
    unpaired: bool

    @property
    def tracker_index(self) -> int:
        return tracker_index(self.mac)


def parse_pair_event(data: bytes) -> PairEvent | None:
    """解 cmd_id 0x18 的配对事件。"""
    if len(data) < 12 or data[0] != CMD_PAIR_EVENT:
        return None
    # <B B H> cmd, data_len, unk 之后是 unk / is_unpair / mac
    return PairEvent(mac=data[6:12], unpaired=bool(data[5]))


def classify(data: bytes) -> str:
    """给一条原始报告一个人类可读的类别名，排查用。"""
    if not data:
        return "empty"
    cmd = data[0]
    if cmd == CMD_TRACKER_DATA:
        report = parse_report(data)
        if report is None:
            return "tracker_data(bad)"
        if report.is_ack:
            return "ack"
        if report.is_heartbeat:
            return "heartbeat"
        if report.is_pose:
            return "pose"
        return f"tracker_data(type=0x{report.type:04x})"
    if cmd == CMD_PAIR_EVENT:
        return "pair_event"
    if cmd == CMD_RF_STATUS:
        return "rf_status"
    if cmd == CMD_RF_STATUS_NEW:
        return "rf_status_new"
    return f"unknown(0x{cmd:02x})"


def parse_report(data: bytes) -> Report | None:
    """把一条 64 字节 HID Input report 拆成头 + 载荷。"""
    if len(data) < HEADER.size or data[0] != CMD_TRACKER_DATA:
        return None
    cmd_id, pkt_idx, mac, ptype, data_len = HEADER.unpack(data[: HEADER.size])
    return Report(cmd_id, pkt_idx, mac, ptype,
                  data[HEADER.size : HEADER.size + data_len])


def decode_pose(payload: bytes) -> Pose | None:
    """解 0x25 字节姿态载荷。dongle 和 WiFi 两条链路通用。"""
    if len(payload) < POSE_SIZE:
        return None
    idx, btns, pos_raw, rot_raw, acc_raw, tail, status_raw = POSE.unpack(payload[:POSE_SIZE])
    # tail 是 3 个半精度角速度 + u16 device_time，不是 4 个半精度
    wx, wy, wz = struct.unpack("<eee", tail[:6])
    device_time = struct.unpack("<H", tail[6:8])[0]
    return Pose(
        idx=idx,
        btns=btns,
        pos=struct.unpack("<fff", pos_raw),
        rot=struct.unpack("<eeee", rot_raw),
        acc=struct.unpack("<eee", acc_raw),
        rot_vel=(wx, wy, wz),
        device_time=device_time,
        status=status_raw & 0x0F,       # 低半字节才是 tracking status
        status_hi=(status_raw >> 4) & 0x0F,
        status_raw=status_raw,
    )


def decode_wifi_frame(frame: bytes) -> Pose | None:
    """解 WiFi 链路的 128 字节 `02 25` 帧（与 dongle 载荷同源）。"""
    if len(frame) < WIFI_POSE_OFFSET + POSE_SIZE or frame[:2] != WIFI_FRAME_MAGIC:
        return None
    return decode_pose(frame[WIFI_POSE_OFFSET : WIFI_POSE_OFFSET + POSE_SIZE])


def idx_gap(prev_idx: int, idx: int) -> int:
    """两帧之间丢了多少帧（idx 按字节回绕）。"""
    return ((idx - prev_idx) & 0xFF) - 1
