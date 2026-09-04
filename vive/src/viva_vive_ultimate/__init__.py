"""viva_vive_ultimate —— 在 Linux 上直接从 VIVE Wireless Dongle 取 Ultimate Tracker 6DoF。

不需要 SteamVR，不需要 Windows，不需要给 tracker 刷 WiFi-only 配置。

    from viva_vive_ultimate import Dongle

    with Dongle() as d:
        for report, pose in d.stream():
            print(pose.pos, pose.rot, pose.status_name)

所有写操作都经过 :mod:`viva_vive_ultimate.safety`（dongle 命令码白名单、0x18 载荷校验、
ACK 三档白名单、审计日志）。见 docs/safety-harness.md。

实测：125 Hz 标称、约 2% 单帧丢包、静置位置噪声 sigma ~= 0.12 mm。
"""

from .dongle import Dongle, find_devices, tracking_sequence
from .safety import (ForbiddenCommand, GuardedCommand, MalformedFrame, UnknownCommand,
                     Unlock, UnsafeCommand, WriteAudit, check_ack, classify_ack)
from .protocol import (PairEvent, Pose, Report, classify, decode_pose,
                       decode_wifi_frame, device_id, device_id_str, mac_str,
                       parse_mac, parse_pair_event, tracker_index)
from .state import PoseValidity, TrackerState, evaluate_pose_validity
from .recorder import CsvRecorder

__version__ = "0.2.0"

__all__ = [
    "Dongle", "find_devices", "tracking_sequence",
    "Unlock", "UnsafeCommand", "ForbiddenCommand", "GuardedCommand", "UnknownCommand",
    "MalformedFrame", "WriteAudit", "check_ack", "classify_ack",
    "Pose", "Report", "PairEvent", "classify", "decode_pose", "decode_wifi_frame",
    "device_id", "device_id_str", "mac_str", "parse_mac", "parse_pair_event",
    "tracker_index",
    "PoseValidity", "TrackerState", "evaluate_pose_validity", "CsvRecorder",
]
