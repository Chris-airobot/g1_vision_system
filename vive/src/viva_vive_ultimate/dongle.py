"""通过 /dev/hidraw 与 VIVE Wireless Dongle 通信。纯标准库（fcntl + ioctl）。

安全边界
--------
**本文件是整个包里唯一碰 ``ioctl`` / ``O_RDWR`` 的地方**（tests/test_safety.py 里有
结构性测试盯着这一点），而所有写操作都先经过 :mod:`viva_vive_ultimate.safety`：

- dongle 命令码只放行 ``DCMD_TX`` (0x18)，没有解锁手段。
  ``0x21`` 会砖掉 dongle，``0x1C`` 进 DFU，``0x98/0x99/0x9A`` 写固件。
- 0x18 载荷逐字节校验（子命令、长度、固定尾部、ASCII）。
- ACK 字符串分 SAFE / GUARDED / FORBIDDEN 三档；GUARDED 要显式 :class:`~safety.Unlock`。
- 每次写尝试都进审计日志（``$VVU_WRITE_LOG``，默认 ``~/.local/state/viva_vive_ultimate/writes.log``）。
- ``dry_run=True`` 时只校验、只记录，不碰设备，设备也只以只读方式打开。
"""

from __future__ import annotations

import fcntl
import glob
import os
import select
import struct
import time
from typing import Iterator

from . import protocol as p
from . import safety
from .safety import (DRYRUN, REFUSED, SENT, Unlock, UnsafeCommand, WriteAudit)

# ------------------------------------------------------------- dongle 命令

DCMD_TX = safety.DCMD_TX                    # 转发 ACK 给 tracker
TX_ACK_TO_MAC = safety.TX_ACK_TO_MAC        # 校验完整 MAC
TX_ACK_TO_PARTIAL_MAC = safety.TX_ACK_TO_PARTIAL_MAC   # 只校验 MAC 前 2 字节

#: 白名单。不在这里的命令一律拒绝发送。（= safety.ALLOWED_DONGLE_COMMANDS）
ALLOWED_COMMANDS = safety.ALLOWED_DONGLE_COMMANDS

#: 已知会造成不可逆后果的命令，单独列出用于报错提示。完整表见 safety.DONGLE_COMMANDS。
DANGEROUS_COMMANDS = {
    0x21: "会砖掉 dongle",
    0x1C: "进入 DFU bootloader",
    0x98: "固件写入 START",
    0x99: "固件写入 DATA",
    0x9A: "固件写入 END",
    0xEF: "改写设备 ID",
}

# --------------------------------------------------------------- ACK 命令

ACK_ROLE_ID = "ARI"
ACK_TRACKING_MODE = "ATM"
ACK_TRACKING_HOST = "ATH"
ACK_WIFI_HOST = "AWH"
ACK_WIFI_COUNTRY = "Wc"
#: ⚠️ 设置 device id。**默认不要发** —— 见 enable_tracking 的说明。GUARDED。
ACK_NEW_ID = "ANI"
ACK_END_MAP = "ALE"
ACK_POWER_OFF = "APF"
ACK_STANDBY = "APS"
ACK_WIFI_CONNECT = "WC"
ACK_MAP_STATUS = "MS"

#: lambda（SLAM 引擎）状态查询/设置前缀
ACK_LAMBDA_ASK_STATUS = "P63:"     # 只读查询
ACK_LAMBDA_SET_STATUS = "P61:"
ACK_LAMBDA_COMMAND = "P64:"

#: 可查询的 lambda 状态键
KEY_TRANSMISSION_READY = 0
KEY_RECEIVED_FIRST_FILE = 1
KEY_RECEIVED_HOST_ED = 2
KEY_RECEIVED_HOST_MAP = 3
KEY_CURRENT_MAP_ID = 4
KEY_MAP_STATE = 5
KEY_CURRENT_TRACKING_STATE = 6

LAMBDA_KEY_NAMES = {
    KEY_TRANSMISSION_READY: "TRANSMISSION_READY",
    KEY_RECEIVED_FIRST_FILE: "RECEIVED_FIRST_FILE",
    KEY_RECEIVED_HOST_ED: "RECEIVED_HOST_ED",
    KEY_RECEIVED_HOST_MAP: "RECEIVED_HOST_MAP",
    KEY_CURRENT_MAP_ID: "CURRENT_MAP_ID",
    KEY_MAP_STATE: "MAP_STATE",
    KEY_CURRENT_TRACKING_STATE: "CURRENT_TRACKING_STATE",
}

#: 地图状态码
MAP_STATE_NAMES = {
    0: "NOT_CHECKED", 1: "EXIST", 2: "NOTEXIST", 3: "REBUILT", 4: "SAVE_OK",
    5: "SAVE_FAIL", 6: "REUSE_OK", 7: "REUSE_FAIL_FEATURE_DIFF",
    8: "REUSE_FAIL_FEATURE_LESS", 9: "REBUILD_WAIT_FOR_STATIC",
    10: "REBUILD_CREATE_MAP",
}

TRACKING_MODE_NONE = -1
TRACKING_MODE_SLAM_CLIENT = 11
TRACKING_MODE_SLAM_HOST = 20


def tracking_sequence(
    mode: int = TRACKING_MODE_SLAM_HOST,
    country: str = "CN",
    role_id: int | None = None,
    as_host: bool = True,
    new_id: int | None = None,
    wifi_host: bool = False,
) -> list[str]:
    """:meth:`Dongle.enable_tracking` 会发出的 ACK 序列。纯函数，CLI 用它做预检。

    默认只有 ``ATM-1`` + ``ATM<mode>``（都是 SAFE）。其余每一项都是 GUARDED，
    不解锁就会在发出**任何一条**之前被整体拒绝。
    """
    seq = []
    if role_id is not None:
        seq.append(f"{ACK_ROLE_ID}{role_id}")
    seq.append(f"{ACK_TRACKING_MODE}{TRACKING_MODE_NONE}")
    if wifi_host:
        seq += [
            f"{ACK_WIFI_COUNTRY}{country}",
            f"{ACK_TRACKING_HOST}{1 if as_host else 0}",
            f"{ACK_WIFI_HOST}{1 if as_host else 0}",
        ]
    if new_id is not None:
        seq.append(f"{ACK_NEW_ID}{new_id}")
    seq.append(f"{ACK_TRACKING_MODE}{mode}")
    return seq


# ------------------------------------------------------------ hidraw ioctl


def _ioc(direction: int, type_: int, nr: int, size: int) -> int:
    req = (direction << 30) | (size << 16) | (type_ << 8) | nr
    return req - (1 << 32) if req >= (1 << 31) else req


def _hidiocsfeature(length: int) -> int:
    return _ioc(3, ord("H"), 0x06, length)


def _hidiocgfeature(length: int) -> int:
    return _ioc(3, ord("H"), 0x07, length)


def find_devices(vid: int = p.VID_VIVE, pid: int = p.PID_DONGLE) -> list[str]:
    """列出匹配的 /dev/hidrawN。"""
    found = []
    for path in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        try:
            with open(os.path.join(path, "device/uevent")) as fh:
                uevent = fh.read().upper()
        except OSError:
            continue
        if f"{vid:08X}" in uevent and f"{pid:08X}" in uevent:
            found.append("/dev/" + os.path.basename(path))
    return found


class PermissionHint(PermissionError):
    def __init__(self, device: str) -> None:
        super().__init__(
            f"没有 {device} 的读写权限。先装 udev 规则：\n"
            f"    sudo ./scripts/setup_udev.sh\n"
            f"（或临时用 sudo 运行）"
        )


class Dongle:
    """打开的 dongle。同一时刻只能被一个进程持有。

    ``writable``  才能发命令（O_RDWR）。默认只读。
    ``unlock``    :class:`~safety.Unlock`，放行指定的 GUARDED ACK 家族。
    ``dry_run``   只校验、只记审计，不 ioctl；设备强制只读打开。
    ``audit``     :class:`~safety.WriteAudit`；默认落到 ``$VVU_WRITE_LOG``。
    """

    # 类级默认，保证跳过 __init__ 的测试替身也有这些属性
    unlock: Unlock | None = None
    dry_run: bool = False
    audit: WriteAudit | None = None

    def __init__(self, device: str | None = None, writable: bool = False, *,
                 unlock: Unlock | None = None, dry_run: bool = False,
                 audit: WriteAudit | None = None) -> None:
        if device is None:
            candidates = find_devices()
            if not candidates:
                raise FileNotFoundError(
                    f"没找到 {p.VID_VIVE:04x}:{p.PID_DONGLE:04x} 的 hidraw 节点，dongle 插了吗？"
                )
            device = candidates[0]
        self.device = device
        self.dry_run = bool(dry_run)
        self.writable = bool(writable) and not self.dry_run
        self.unlock = unlock
        self.audit = audit if audit is not None else WriteAudit()
        flags = (os.O_RDWR if self.writable else os.O_RDONLY) | os.O_NONBLOCK
        try:
            self.fd = os.open(device, flags)
        except PermissionError as exc:
            raise PermissionHint(device) from exc

    # ---------------------------------------------------------------- 读

    def read_raw(self, timeout: float = 0.5, limit: int = 512) -> Iterator[bytes]:
        """等到有数据后，把内核缓冲里的**所有**报告一次吃干净，不做任何过滤。

        排查用：配对事件 0x18、RF 状态 0x1d/0x1e 等都只出现在这里，
        :meth:`read_reports` 会把它们过滤掉。
        """
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return
        for _ in range(limit):
            try:
                raw = os.read(self.fd, 1024)
            except BlockingIOError:
                return
            if not raw:
                return
            yield raw

    def read_reports(self, timeout: float = 0.5, limit: int = 512) -> Iterator[p.Report]:
        """一次唤醒里的 tracker 数据报告（cmd_id 0x28）。其余类型被过滤掉。"""
        for raw in self.read_raw(timeout, limit):
            report = p.parse_report(raw)
            if report is not None:
                yield report

    def read_poses(self, timeout: float = 0.5) -> Iterator[tuple[p.Report, p.Pose]]:
        """一次唤醒里的姿态。要持续取流请用 :meth:`stream`。"""
        for report in self.read_reports(timeout):
            if report.is_pose:
                pose = p.decode_pose(report.payload)
                if pose is not None:
                    yield report, pose

    def stream(self, timeout: float = 0.5) -> Iterator[tuple[p.Report, p.Pose]]:
        """持续产出 (report, pose)，直到调用方 break 或 KeyboardInterrupt。"""
        while True:
            yield from self.read_poses(timeout)

    def stream_reports(self, timeout: float = 0.5) -> Iterator[p.Report]:
        """持续产出所有报告，包括 ACK 与心跳。"""
        while True:
            yield from self.read_reports(timeout)

    # ---------------------------------------------------------------- 写

    def _send_feature(self, report: bytes) -> int:
        """最底层写。只有 :meth:`send_command` 调它，且到这里时已经过闸。"""
        if not self.writable:
            raise PermissionError("dongle 以只读方式打开，无法发送命令")
        buf = bytearray(b"\x00" + report.ljust(p.REPORT_SIZE, b"\x00")[: p.REPORT_SIZE])
        return fcntl.ioctl(self.fd, _hidiocsfeature(len(buf)), buf)

    def _get_feature(self) -> bytes:
        buf = bytearray(p.REPORT_SIZE + 1)
        fcntl.ioctl(self.fd, _hidiocgfeature(len(buf)), buf)
        return bytes(buf[1:])       # buf[0] 是 report number

    def _audit(self, decision: str, **kw) -> None:
        if self.audit is not None:
            self.audit.record(decision, device=self.device, unlock=self.unlock, **kw)

    def send_command(self, cmd_id: int, data: bytes = b"", retries: int = 10) -> bytes:
        """发一条 dongle 命令并读回同 cmd_id 的响应。

        **唯一的写入口。** 顺序：命令码白名单 → 0x18 载荷结构 → ACK 白名单/解锁
        → dry-run 短路 → 写权限 → ioctl。任何一步不过都不会碰设备，并记审计。
        """
        try:
            safety.require_dongle_command(cmd_id)
        except UnsafeCommand as exc:
            self._audit(REFUSED, cmd_id=cmd_id, risk=safety.check_dongle_command(cmd_id).risk,
                        reason=str(exc))
            raise
        try:
            frame = safety.parse_tx_payload(data)
            verdict = safety.require_ack(frame.ack, self.unlock)
        except UnsafeCommand as exc:
            self._audit(REFUSED, cmd_id=cmd_id, reason=str(exc))
            raise
        if self.dry_run:
            self._audit(DRYRUN, cmd_id=cmd_id, mac=frame.mac, ack=frame.ack,
                        risk=verdict.risk, reason=verdict.reason)
            return b""
        if not self.writable:
            self._audit(REFUSED, cmd_id=cmd_id, mac=frame.mac, ack=frame.ack,
                        risk=verdict.risk, reason="dongle 以只读方式打开")
            raise PermissionError("dongle 以只读方式打开，无法发送命令")
        self._send_feature(struct.pack("<BB", cmd_id, len(data) + 2) + data)
        self._audit(SENT, cmd_id=cmd_id, mac=frame.mac, ack=frame.ack,
                    risk=verdict.risk, reason=verdict.reason)
        for _ in range(retries):
            resp = self._get_feature()
            if len(resp) >= 3 and resp[0] == cmd_id:
                return resp[2 : 2 + max(0, resp[1] - 4)]
        return b""

    def send_ack(self, mac: bytes, ack: str) -> bytes:
        """把一条 ASCII ACK 命令经 RF 转发给某台 tracker。过 :func:`safety.require_ack`。"""
        safety.validate_ack_text(ack)
        if len(mac) != 6:
            raise ValueError(f"MAC 必须 6 字节，实际 {len(mac)}")
        preamble = struct.pack("<B6sBB", TX_ACK_TO_PARTIAL_MAC, bytes(mac), 0, 1)
        payload = struct.pack("<B", len(ack)) + ack.encode("ascii")
        return self.send_command(DCMD_TX, preamble + payload)

    def send_acks(self, mac: bytes, acks: list[str], delay: float = 0.05) -> list[tuple[str, bytes]]:
        """按顺序发一段 ACK。**整段先过闸再发**，不会发到一半停在中间状态。"""
        safety.require_sequence(acks, self.unlock)
        results = []
        for ack in acks:
            results.append((ack, self.send_ack(mac, ack)))
            if delay > 0 and not self.dry_run:
                time.sleep(delay)
        return results

    def enable_tracking(
        self,
        mac: bytes,
        mode: int = TRACKING_MODE_SLAM_HOST,
        country: str = "CN",
        role_id: int | None = None,
        as_host: bool = True,
        new_id: int | None = None,
        wifi_host: bool = False,
        delay: float = 0.05,
    ) -> list[tuple[str, bytes]]:
        """启流序列。tracker 只推 2 字节心跳时用它把 SLAM 拉起来。

        默认只发最少的两条 ``ATM-1`` + ``ATM<mode>``（SAFE），其余全部可选且都是
        GUARDED —— 需要 :class:`~safety.Unlock` 覆盖对应家族，否则整段拒绝、一条不发。
        实测只发一条 ``ATM20`` 就能把 SLAM 拉起来，多发的每一条都是风险。

        ⚠️ **``new_id`` 默认为 None，即不发 ``ANI``。**

        ``ANI`` 设置 device id，而设备的地图很可能**按 device id 索引**
        （``persist.lambda.multimap.size = 3``，地图按槽位存）。改 id 可能让
        已有地图找不到（docs/protocol.md 里两次实测结论相反，尚未闭合）。

        ``wifi_host`` 默认 False，跳过 Wc/ATH/AWH —— 单台走 dongle 取数不需要
        它去起 WiFi Direct 热点；而且 ATH/AWH **发一次持久化到永远**。
        """
        seq = tracking_sequence(mode, country, role_id, as_host, new_id, wifi_host)
        return self.send_acks(mac, seq, delay)

    def ask_lambda_status(self, mac: bytes, key: int) -> bytes:
        """查询一个 lambda（SLAM）状态键。只读，不改设备状态。SAFE。

        回复以 ``LS`` 开头的 ACK 异步返回，格式 ``LS<key>,<value>,<?>``。
        """
        return self.send_ack(mac, f"{ACK_LAMBDA_ASK_STATUS}{key}")

    def wifi_connect(self, mac: bytes) -> bytes:
        """让 client 去连 host 的 WiFi Direct 热点。GUARDED（WC）。"""
        return self.send_ack(mac, ACK_WIFI_CONNECT)

    def end_map(self, mac: bytes) -> bytes:
        """结束建图。GUARDED（ALE）。client 卡在 MAP_NOT_CHECKED 时 tracker_toybox 会发这个。"""
        return self.send_ack(mac, ACK_END_MAP)

    def power_off(self, mac: bytes) -> bytes:
        """关机。GUARDED（APF）。"""
        return self.send_ack(mac, ACK_POWER_OFF)

    # -------------------------------------------------------------- 生命周期

    def close(self) -> None:
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None  # type: ignore[assignment]
        if self.audit is not None:
            self.audit.close()

    def __enter__(self) -> "Dongle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
