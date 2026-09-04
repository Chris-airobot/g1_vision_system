"""安全闸（harness）—— 所有发往 dongle / tracker 的写操作都必须经过这里。

为什么需要它
------------
- dongle 命令 ``0x21`` 会把 dongle 砖掉（上游作者实测，解砖命令得靠能发 USB 命令，
  砖了就发不出去）；``0x1C`` 进 DFU；``0x98/0x99/0x9A`` 写固件。
- 0x18 载荷里经 RF 转发给 tracker 的 ASCII ACK 同样有能打坏设备的：``AFM`` 触发
  tracker 固件更新、``C*``/``c*`` 写标定、``APR`` 复位、``ATM1`` 会破坏
  ``persist.lambda.trans_setup``……
- 还有一批命令不致命但会持久化改状态：``ATH``/``AWH`` 发一次管到永远、``ANI``
  改 device id、``W*`` 写 WiFi 口令、``APC`` 清配对表（重新配对要 Windows + VIVE Hub）。

设计
----
两层白名单，**缺省拒绝**，每次写尝试都进审计日志：

1. **dongle 命令码**（Feature report 第 1 字节）—— 只放行 ``0x18 DCMD_TX``。
   硬边界，**没有解锁手段**。上游标为只读但本项目没验证过的（0x26/0x27/0xF0/0xFF）
   同样不放行。

2. **0x18 载荷结构** —— 子命令只允许 0x03/0x04（定向到某个 MAC），拒绝 0x05
   （广播到所有 tracker）和其它；长度字段、固定尾部、ASCII 可打印逐项校验。
   这样用 ``send_command(0x18, 手拼载荷)`` 也绕不过 ACK 白名单。

3. **ACK 字符串** —— 三档：

   ========== =============================================================
   SAFE       本项目日常使用、可逆、无已知副作用。直接放行。
   GUARDED    会持久化或改变设备状态，但有恢复路径。必须显式 :class:`Unlock`
              并写明理由；理由进审计日志。
   FORBIDDEN  固件更新 / 标定 / 身份字段 / 已知打坏配置。**没有解锁手段。**
   ========== =============================================================

   不在表里的一律 UNKNOWN，拒绝。**要放行新命令，先在 docs/safety-harness.md
   里补证据，再加规则，再加测试。**

本模块纯策略、零 IO（审计日志除外），可以脱离真机测试。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TextIO

# ------------------------------------------------------------------ 风险档

SAFE = "SAFE"
GUARDED = "GUARDED"
FORBIDDEN = "FORBIDDEN"
UNVERIFIED = "UNVERIFIED"    # 只用于 dongle 命令：上游标只读，本项目未验证 -> 不放行
UNKNOWN = "UNKNOWN"

RISK_ORDER = (SAFE, GUARDED, UNVERIFIED, FORBIDDEN, UNKNOWN)


# ------------------------------------------------------------------ 异常

class UnsafeCommand(ValueError):
    """被安全闸拒绝的写操作。基类，方便调用方一网打尽。"""


class ForbiddenCommand(UnsafeCommand):
    """FORBIDDEN：没有解锁手段。"""


class GuardedCommand(UnsafeCommand):
    """GUARDED：缺少覆盖该家族的 :class:`Unlock`。"""


class UnknownCommand(UnsafeCommand):
    """不在白名单表里。"""


class MalformedFrame(UnsafeCommand):
    """0x18 载荷结构不对。"""


# ------------------------------------------------------------------ dongle 命令码

class DongleCommand(NamedTuple):
    cmd_id: int
    name: str
    risk: str
    reason: str
    source: str


DCMD_TX = 0x18

#: 全部已知 dongle 命令。来源：vive_ultimate_tracker_re/tracker_toybox/enums_horusd_dongle.py，
#: 与 UltimateTracker_FirmWare 逆出的 dispatcher 交叉验证过的标 [对拍]。
DONGLE_COMMANDS: dict[int, DongleCommand] = {c.cmd_id: c for c in (
    DongleCommand(0x18, "DCMD_TX", SAFE,
                  "把 ASCII ACK 经 RF 转发给指定 tracker；载荷另行逐字节校验", "[对拍]"),
    DongleCommand(0x1C, "DCMD_RESET_DFU", FORBIDDEN, "进入 DFU bootloader", "[引用]"),
    DongleCommand(0x1D, "DCMD_REQUEST_RF_CHANGE_BEHAVIOR", FORBIDDEN,
                  "改 RF 行为：配对/省电/重启 RF/出厂复位(0x05)/清配对表(0x06)", "[引用]"),
    DongleCommand(0x1E, "DCMD_1E", UNVERIFIED, "读 RF 行为状态；上游标只读，本项目未验证", "[引用]"),
    DongleCommand(0x21, "DCMD_21", FORBIDDEN, "会砖掉 dongle（上游作者实测：BRICKED MY DONGLE）", "[引用]"),
    DongleCommand(0x26, "DCMD_26", UNVERIFIED, "回显 USB 缓冲；上游标只读，本项目未验证", "[引用]"),
    DongleCommand(0x27, "DCMD_27", UNVERIFIED, "RF_REPORT_RF_IDS；上游标只读，本项目未验证", "[引用]"),
    DongleCommand(0x28, "DCMD_28", FORBIDDEN, "子命令 0x06 可能重启，0x08/0x09 写 tracker 表", "[引用]"),
    DongleCommand(0x98, "DCMD_FLASH_WRITE_1", FORBIDDEN, "固件写入 START（staging flash）", "[对拍]"),
    DongleCommand(0x99, "DCMD_FLASH_WRITE_2", FORBIDDEN, "固件写入 DATA（带 crc32）", "[对拍]"),
    DongleCommand(0x9A, "DCMD_FLASH_WRITE_3", FORBIDDEN, "固件写入 END（boot CRC gate）", "[对拍]"),
    DongleCommand(0x9E, "DCMD_9E", FORBIDDEN, "接受 1 字节，语义未知", "[引用]"),
    DongleCommand(0x9F, "DCMD_9F", FORBIDDEN, "data[0]==2 时复位进刷写流程", "[引用]"),
    DongleCommand(0xEB, "DCMD_EB", FORBIDDEN, "重启 dongle", "[引用]"),
    DongleCommand(0xEF, "DCMD_WRITE_CR_ID", FORBIDDEN, "改写设备 ID（PCB ID / SKU / SN / Ship SN）", "[引用]"),
    DongleCommand(0xF0, "DCMD_GET_CR_ID", UNVERIFIED, "读 PCB ID / SKU / SN；上游标只读，本项目未验证", "[引用]"),
    DongleCommand(0xF3, "DCMD_F3", FORBIDDEN, "0x1D 的包装", "[引用]"),
    DongleCommand(0xF4, "DCMD_F4", FORBIDDEN, "带校验和的 tracker 相关子命令，语义未知", "[引用]"),
    DongleCommand(0xFF, "DCMD_QUERY_ROM_VERSION", UNVERIFIED, "读 ROM 版本；上游标只读，本项目未验证", "[引用]"),
)}

#: 唯一放行的 dongle 命令。
ALLOWED_DONGLE_COMMANDS = frozenset(c.cmd_id for c in DONGLE_COMMANDS.values() if c.risk == SAFE)
assert ALLOWED_DONGLE_COMMANDS == {DCMD_TX}


def check_dongle_command(cmd_id: int) -> DongleCommand:
    """给一个命令码一个裁决。不抛异常。"""
    known = DONGLE_COMMANDS.get(cmd_id)
    if known is not None:
        return known
    return DongleCommand(cmd_id, f"DCMD_{cmd_id:02X}", UNKNOWN, "不在白名单内", "")


def require_dongle_command(cmd_id: int) -> DongleCommand:
    """不是 0x18 就抛 :class:`ForbiddenCommand`。"""
    verdict = check_dongle_command(cmd_id)
    if verdict.risk != SAFE:
        raise ForbiddenCommand(f"拒绝发送命令 0x{cmd_id & 0xFF:02x}：{verdict.reason}")
    return verdict


# ------------------------------------------------------------------ 0x18 载荷结构

TX_ACK_TO_MAC = 0x03            # 校验完整 MAC
TX_ACK_TO_PARTIAL_MAC = 0x04    # 只校验 MAC 前 2 字节
TX_BROADCAST = 0x05             # 给所有 tracker 发 P:%d —— 拒绝
ALLOWED_TX_SUBCMDS = frozenset({TX_ACK_TO_MAC, TX_ACK_TO_PARTIAL_MAC})

#: 上游注明 TX_ACK_TO_MAC 的 data 长度必须 <= 0x2C；减去 9 字节前导和 1 字节长度。
MAX_ACK_LEN = 0x2C - 10
TX_PREAMBLE_LEN = 9             # subcmd + 6 MAC + 0x00 0x01


class TxFrame(NamedTuple):
    subcmd: int
    mac: bytes
    ack: str


def parse_tx_payload(data: bytes) -> TxFrame:
    """把 0x18 的 data 部分拆开并逐项校验；不合规抛 :class:`MalformedFrame`。"""
    data = bytes(data)
    if len(data) < TX_PREAMBLE_LEN + 2:
        raise MalformedFrame(f"0x18 载荷过短：{len(data)} 字节")
    subcmd = data[0]
    if subcmd not in ALLOWED_TX_SUBCMDS:
        if subcmd == TX_BROADCAST:
            raise MalformedFrame("拒绝 0x18 子命令 0x05：会广播给所有 tracker")
        raise MalformedFrame(f"拒绝 0x18 子命令 0x{subcmd:02x}：只允许 0x03/0x04（定向到 MAC）")
    if data[7:9] != b"\x00\x01":
        raise MalformedFrame(f"0x18 固定尾部应为 00 01，实际 {data[7:9].hex(' ')}")
    n = data[9]
    if n == 0:
        raise MalformedFrame("ACK 长度为 0")
    if n > MAX_ACK_LEN:
        raise MalformedFrame(f"ACK 长度 {n} 超过上限 {MAX_ACK_LEN}")
    if len(data) != TX_PREAMBLE_LEN + 1 + n:
        raise MalformedFrame(f"ACK 长度字段 {n} 与载荷长度 {len(data) - 10} 不符")
    raw = data[10:]
    try:
        ack = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise MalformedFrame(f"ACK 不是 ASCII：{raw!r}") from exc
    validate_ack_text(ack)
    return TxFrame(subcmd, data[1:7], ack)


def validate_ack_text(ack: str) -> None:
    if not isinstance(ack, str) or not ack:
        raise MalformedFrame("ACK 为空")
    if len(ack) > MAX_ACK_LEN:
        raise MalformedFrame(f"ACK 长度 {len(ack)} 超过上限 {MAX_ACK_LEN}")
    if any(not (0x20 <= ord(c) <= 0x7E) for c in ack):
        raise MalformedFrame(f"ACK 含不可打印或非 ASCII 字符：{ack!r}")


# ------------------------------------------------------------------ ACK 白名单

class AckRule(NamedTuple):
    pattern: str
    family: str
    risk: str
    reason: str
    source: str


#: 按顺序 fullmatch，第一条命中生效。来源标注同 docs/protocol.md：
#: [实测] 本项目真机；[引用] 上游逆向，未独立验证。
ACK_RULES: tuple[AckRule, ...] = (
    # ---------------------------------------------------------------- SAFE
    AckRule(r"ATM(-1|11|20|21)", "ATM", SAFE,
            "tracking mode：-1 清空 / 11 SLAM client / 20 SLAM host / 21 body。本项目启流用，可逆",
            "[实测]"),
    AckRule(r"P63:\d+", "P63", SAFE, "查询 lambda(SLAM) 状态键，只读", "[引用]"),
    AckRule(r"LP", "LP", SAFE, "查询 lambda 属性 trans_setup/normalmode/3rdhost，只读", "[实测]"),
    AckRule(r"ACF\d{1,3}", "ACF", SAFE, "相机 FPS，运行时设置，默认 50（vvu-bench 实验用）", "[实测]"),
    AckRule(r"ATW", "ATW", SAFE, "启用加速度数据；WiFi 方案默认序列的一部分", "[引用]"),
    # ---------------------------------------------------------------- FORBIDDEN（先于 GUARDED 里的宽匹配）
    AckRule(r"ATM1", "ATM1", FORBIDDEN,
            "tracking mode 1 会破坏 persist.lambda.trans_setup（上游注明），恢复需要 ADB", "[引用]"),
    AckRule(r"AFM.*", "AFM", FORBIDDEN, "START_FOTA：触发 tracker 固件更新，砖机风险", "[引用]"),
    AckRule(r"FD.*", "FD", FORBIDDEN, "文件下载通道（固件/地图传输）", "[引用]"),
    AckRule(r"APR", "APR", FORBIDDEN, "reset，深度未知（可能是出厂复位）", "[引用]"),
    AckRule(r"[Cc].*", "CALIB", FORBIDDEN, "标定类命令：写坏相机/IMU 标定会永久损坏追踪", "[引用]"),
    AckRule(r"(ADS|ASS|ASI|API|AV1|Av1|ANA|AZZ|AGN|NA).*", "IDENTITY", FORBIDDEN,
            "设备身份/信息字段（SN、SKU、PCB ID、版本）：只应由 tracker 上报，不应下发", "[引用]"),
    # ---------------------------------------------------------------- GUARDED
    AckRule(r"ARI\d+", "ARI", GUARDED, "role id；可能持久化", "[引用]"),
    AckRule(r"ATH[01]", "ATH", GUARDED,
            "tracking host；持久化，发一次管到永远，清除要显式发 ATH0", "[实测]"),
    AckRule(r"AWH[01]", "AWH", GUARDED,
            "wifi host；持久化，会起 WiFi Direct 热点并广播明文口令", "[实测]"),
    AckRule(r"Wc[A-Z]{2}", "Wc", GUARDED, "WiFi 国家码；持久化", "[引用]"),
    AckRule(r"WC", "WC", GUARDED, "让 client 去连 host 热点", "[引用]"),
    AckRule(r"ANI\d+", "ANI", GUARDED,
            "设置 device id；地图可能按 id 索引，改了可能找不到已有地图", "[实测]"),
    AckRule(r"APF", "APF", GUARDED, "关机", "[引用]"),
    AckRule(r"APS", "APS", GUARDED, "待机", "[引用]"),
    AckRule(r"APC", "APC", GUARDED, "关机并清配对表；重新配对需要 Windows + VIVE Hub", "[引用]"),
    AckRule(r"ALE", "ALE", GUARDED, "结束建图", "[引用]"),
    AckRule(r"ATS\d+", "ATS", GUARDED, "设置设备时钟（clock_settime）", "[引用]"),
    AckRule(r"ACP\d+", "ACP", GUARDED, "相机策略，语义未闭合", "[引用]"),
    AckRule(r"P61:.+", "P61", GUARDED, "写 lambda 状态", "[引用]"),
    AckRule(r"P64:3", "RESET_MAP", GUARDED, "lambda RESET_MAP：清掉当前地图", "[引用]"),
    AckRule(r"P64:[012]", "P64", GUARDED, "lambda 命令：0 ASK_ED / 1 ASK_MAP / 2 KF_SYNC", "[引用]"),
    AckRule(r"FW\d*", "FW", GUARDED, "语义未知（上游 TODO）；WiFi 方案默认发 FW2", "[引用]"),
    AckRule(r"W[SsptfiI].*", "WIFI_CFG", GUARDED,
            "WiFi SSID/口令/信道/IP 写入；持久化，可用 ADB 恢复", "[引用]"),
)

#: 可以被 Unlock 覆盖的家族名。
GUARDED_FAMILIES = frozenset(r.family for r in ACK_RULES if r.risk == GUARDED)
FORBIDDEN_FAMILIES = frozenset(r.family for r in ACK_RULES if r.risk == FORBIDDEN)

_COMPILED = tuple((re.compile(r.pattern), r) for r in ACK_RULES)


class AckVerdict(NamedTuple):
    ack: str
    risk: str
    family: str
    reason: str
    source: str
    allowed: bool
    unlock_reason: str = ""


@dataclass(frozen=True)
class Unlock:
    """显式解锁若干 GUARDED 家族。理由必填，会写进审计日志。

    只能解锁 GUARDED；试图解锁 FORBIDDEN 或未知家族直接抛 ValueError。
    """

    families: frozenset[str]
    reason: str

    def __init__(self, families: Iterable[str] | str, reason: str) -> None:
        if isinstance(families, str):
            families = [families]
        fams = frozenset(f.strip() for f in families if f and f.strip())
        if not fams:
            raise ValueError("Unlock 至少要指定一个家族")
        bad = sorted(fams - GUARDED_FAMILIES)
        if bad:
            hint = ", ".join(f"{b}（FORBIDDEN，无解锁手段）" if b in FORBIDDEN_FAMILIES
                             else f"{b}（未知家族）" for b in bad)
            raise ValueError(f"不能解锁 {hint}。可解锁的家族：{', '.join(sorted(GUARDED_FAMILIES))}")
        if not reason or not reason.strip():
            raise ValueError("Unlock 必须写理由（会进审计日志）")
        object.__setattr__(self, "families", fams)
        object.__setattr__(self, "reason", reason.strip())

    def covers(self, family: str) -> bool:
        return family in self.families

    def __str__(self) -> str:
        return f"unlock[{','.join(sorted(self.families))}] {self.reason}"


def classify_ack(ack: str) -> AckVerdict:
    """查表。不看解锁、不抛异常。"""
    try:
        validate_ack_text(ack)
    except MalformedFrame as exc:
        return AckVerdict(ack, UNKNOWN, "", str(exc), "", False)
    for regex, rule in _COMPILED:
        if regex.fullmatch(ack):
            return AckVerdict(ack, rule.risk, rule.family, rule.reason, rule.source,
                              rule.risk == SAFE)
    return AckVerdict(ack, UNKNOWN, "", "未知 ACK，不在白名单内", "", False)


def check_ack(ack: str, unlock: Unlock | None = None) -> AckVerdict:
    """查表 + 套解锁。不抛异常；看 ``allowed``。"""
    v = classify_ack(ack)
    if v.risk == GUARDED and unlock is not None and unlock.covers(v.family):
        return v._replace(allowed=True, unlock_reason=unlock.reason)
    return v


def require_ack(ack: str, unlock: Unlock | None = None) -> AckVerdict:
    """不放行就抛对应异常。"""
    v = check_ack(ack, unlock)
    if v.allowed:
        return v
    if v.risk == FORBIDDEN:
        raise ForbiddenCommand(f"拒绝 ACK {ack!r}（{v.family}，FORBIDDEN）：{v.reason}")
    if v.risk == GUARDED:
        raise GuardedCommand(
            f"拒绝 ACK {ack!r}（{v.family}，GUARDED）：{v.reason}。"
            f"要发必须显式解锁：Unlock({v.family!r}, reason=...) / CLI --unlock {v.family} --unlock-reason ...")
    if v.family == "" and v.reason.startswith("ACK"):
        raise MalformedFrame(v.reason)
    raise UnknownCommand(f"拒绝 ACK {ack!r}：{v.reason}")


def require_sequence(acks: Iterable[str], unlock: Unlock | None = None) -> list[AckVerdict]:
    """整段序列先全部过闸再发 —— 不会发到一半停在中间状态。"""
    return [require_ack(a, unlock) for a in acks]


# ------------------------------------------------------------------ 审计日志

DEFAULT_LOG = os.path.join(
    os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"),
    "viva_vive_ultimate", "writes.log")

SENT = "SENT"
REFUSED = "REFUSED"
DRYRUN = "DRYRUN"


class WriteAudit:
    """追加式审计日志。每条写尝试一行，放行/拒绝/dry-run 都记。

    路径优先级：构造参数 > ``$VVU_WRITE_LOG`` > :data:`DEFAULT_LOG`。
    ``path=False`` 关闭文件（测试用）。文件打不开时静默退化为只走 ``stream``。
    """

    def __init__(self, path: str | os.PathLike | None | bool = None,
                 stream: TextIO | None = None, echo: bool = False) -> None:
        if path is False:
            self.path: str | None = None
        else:
            self.path = str(path) if path else (os.environ.get("VVU_WRITE_LOG") or DEFAULT_LOG)
        self.stream = stream if stream is not None else sys.stderr
        self.echo = echo
        self.entries: list[str] = []
        self._fh: TextIO | None = None
        self._failed = False

    def _file(self) -> TextIO | None:
        if self._fh is None and self.path and not self._failed:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8")
            except OSError:
                self._failed = True
        return self._fh

    def record(self, decision: str, *, device: str = "", cmd_id: int | None = None,
               mac: bytes | None = None, ack: str | None = None, risk: str = "",
               reason: str = "", unlock: Unlock | None = None) -> str:
        stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        parts = [stamp, f"pid={os.getpid()}", decision, f"dev={device or '-'}"]
        if cmd_id is not None:
            parts.append(f"cmd=0x{cmd_id & 0xFF:02x}")
        if mac is not None:
            parts.append("mac=" + ":".join(f"{b:02x}" for b in mac))
        if ack is not None:
            parts.append(f"ack={ack!r}")
        if risk:
            parts.append(f"risk={risk}")
        if reason:
            parts.append(f"reason={reason}")
        if unlock is not None:
            parts.append(f"unlock={unlock}")
        line = " ".join(parts)
        self.entries.append(line)
        fh = self._file()
        if fh is not None:
            fh.write(line + "\n")
            fh.flush()
        if self.echo:
            print(f"[写闸] {line}", file=self.stream, flush=True)
        return line

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


#: 不落盘、不回显的审计器，测试用。
def null_audit() -> WriteAudit:
    return WriteAudit(path=False)


# ------------------------------------------------------------------ CLI

def _fmt_table(rows: list[tuple[str, ...]], widths: tuple[int, ...]) -> str:
    out = []
    for row in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vvu-safety",
        description="查看/预检安全闸。不碰任何设备。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="退出码：0 全部放行；2 有被拒绝的。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    chk = sub.add_parser("check", help="预检若干 ACK（逗号或空格分隔）")
    chk.add_argument("acks", nargs="+")
    chk.add_argument("--unlock", action="append", default=[], metavar="FAMILY")
    chk.add_argument("--unlock-reason", default="")
    chk.add_argument("--wifi", action="store_true",
                     help="WiFi 桥接预检：只拦 FORBIDDEN，GUARDED/UNKNOWN 只警告")

    sub.add_parser("table", help="打印 dongle 命令表与 ACK 规则表")

    log = sub.add_parser("log", help="打印审计日志")
    log.add_argument("--tail", type=int, default=50)
    log.add_argument("--path", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "table":
        print("dongle 命令码（只放行 SAFE）")
        rows = [(f"0x{c.cmd_id:02X}", c.name, c.risk, c.reason, c.source)
                for c in sorted(DONGLE_COMMANDS.values())]
        print(_fmt_table(rows, (6, 34, 10, 60, 6)))
        print()
        print("ACK 规则（按顺序 fullmatch；不命中 = UNKNOWN = 拒绝）")
        rows = [(r.pattern, r.family, r.risk, r.reason, r.source) for r in ACK_RULES]
        print(_fmt_table(rows, (40, 10, 10, 60, 6)))
        print()
        print(f"可解锁家族：{', '.join(sorted(GUARDED_FAMILIES))}")
        print(f"ACK 长度上限 {MAX_ACK_LEN}；0x18 子命令只允许 0x03/0x04")
        return 0

    if args.cmd == "log":
        path = args.path or os.environ.get("VVU_WRITE_LOG") or DEFAULT_LOG
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            print(f"没有审计日志：{path}")
            return 0
        print(f"# {path}  共 {len(lines)} 条")
        sys.stdout.writelines(lines[-args.tail:])
        return 0

    # check
    acks = [a for item in args.acks for a in item.split(",") if a]
    unlock = None
    fams = [f for item in args.unlock for f in item.split(",") if f]
    if fams:
        try:
            unlock = Unlock(fams, args.unlock_reason)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    bad = 0
    for ack in acks:
        v = check_ack(ack, unlock)
        if args.wifi:
            ok = v.risk != FORBIDDEN
            tag = "ok " if v.allowed else ("WARN" if ok else "STOP")
        else:
            ok = v.allowed
            tag = "ok " if ok else "STOP"
        bad += not ok
        fam = v.family or "-"
        print(f"{tag}  {ack:<16s} {v.risk:<10s} {fam:<10s} {v.reason}")
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
