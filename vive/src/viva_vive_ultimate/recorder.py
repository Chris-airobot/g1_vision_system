"""把姿态流写成 CSV。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TextIO

from . import protocol as p

COLUMNS = [
    "recv_time_ns", "tracker", "mac", "idx", "status", "status_raw", "btns",
    "device_time", "x", "y", "z", "qx", "qy", "qz", "qw",
    "ax", "ay", "az", "wx", "wy", "wz",
]


class CsvRecorder:
    """逐帧追加写入。行缓冲，进程被 kill 也不丢已写的数据。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: TextIO = open(self.path, "w", buffering=1, encoding="utf-8")
        self._fh.write(",".join(COLUMNS) + "\n")
        self.rows = 0

    def write(self, tracker_index: int, mac: bytes, pose: p.Pose,
              recv_time_ns: int | None = None) -> None:
        t = recv_time_ns if recv_time_ns is not None else time.monotonic_ns()
        fields = [
            str(t), str(tracker_index), p.mac_str(mac), str(pose.idx),
            pose.status_name, f"0x{pose.status_raw:02x}", str(pose.btns),
            str(pose.device_time),
            *(f"{v:.6f}" for v in pose.pos),
            *(f"{v:.5f}" for v in pose.rot),
            *(f"{v:.5f}" for v in pose.acc),
            *(f"{v:.5f}" for v in pose.rot_vel),
        ]
        self._fh.write(",".join(fields) + "\n")
        self.rows += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "CsvRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
