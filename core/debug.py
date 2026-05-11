# core/debug.py
"""
调试输出工具
所有调试信息统一走这里，受 DEBUG 开关控制
"""

from core.config import DEBUG


def dlog(lines: list[str], msg: str):
    """
    向 lines 追加调试信息（仅 DEBUG=True 时）
    lines: 日志行列表（传引用，直接 append）
    msg:   调试信息字符串
    """
    if DEBUG:
        lines.append(f"[DEBUG] {msg}\n")


def dprint(msg: str):
    """
    打印到 stdout（仅 DEBUG=True 时）
    用于 fetcher.py 等无法访问 lines 的地方
    """
    if DEBUG:
        print(f"[DEBUG] {msg}")