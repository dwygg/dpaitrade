from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OHLCRow:
    """
    通用 OHLC 数据行。

    必填字段：
    - ts
    - open
    - high
    - low
    - close

    其余 CSV 列会放入 meta 中，便于后续扩展，例如：
    - spread
    - atr
    - volatility_score
    - symbol
    """

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    meta: dict[str, Any] = field(default_factory=dict)


def _parse_timestamp(value: str) -> datetime:
    """
    解析时间字符串。

    支持：
    - 2026-03-06 10:00:00
    - 2026/03/06 10:00:00
    - 2026.03.06 10:00:00
    - 2026-03-06T10:00:00
    - 2026-03-06T10:00:00Z
    """
    text = value.strip()
    if not text:
        raise ValueError("timestamp 为空")

    if text.endswith("Z"):
        text = text[:-1]

    # 先尝试 ISO 格式
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = None

    if dt is None:
        # 兼容 MT5 导出的点分日期格式
        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y.%m.%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M",
            "%Y.%m.%d %H:%M",
        )

        for fmt in formats:
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        raise ValueError(f"无法解析时间字段：{value}")

    # 若带时区，统一去掉 tzinfo
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    return dt

def _auto_parse_value(value: str) -> Any:
    """
    自动解析额外字段。

    规则：
    - 空字符串原样返回
    - 能转 float 就转 float
    - 否则保留原字符串
    """
    text = value.strip()
    if text == "":
        return ""

    try:
        return float(text)
    except ValueError:
        return text


def load_ohlc_csv(path: str | Path) -> list[OHLCRow]:
    """
    从 CSV 加载 OHLC 数据。

    要求 CSV 至少包含以下列：
    - timestamp
    - open
    - high
    - low
    - close

    其余列会自动放入 meta。
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

    rows: list[OHLCRow] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 文件缺少表头：{csv_path}")

        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV 缺少必要列 {missing}，文件：{csv_path}")

        for idx, raw in enumerate(reader, start=2):
            try:
                ts = _parse_timestamp(raw["timestamp"])
                open_ = float(raw["open"])
                high = float(raw["high"])
                low = float(raw["low"])
                close = float(raw["close"])
            except Exception as exc:
                raise ValueError(
                    f"CSV 解析失败：文件={csv_path}，行号={idx}，错误={exc}"
                ) from exc

            meta: dict[str, Any] = {}
            for key, value in raw.items():
                if key in {"timestamp", "open", "high", "low", "close"}:
                    continue
                meta[key] = _auto_parse_value(value)

            rows.append(
                OHLCRow(
                    ts=ts,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    meta=meta,
                )
            )

    rows.sort(key=lambda x: x.ts)
    return rows


def slice_recent_rows(
    rows: list[OHLCRow],
    ts_list: list[datetime],
    current_ts: datetime,
    lookback: int,
) -> list[OHLCRow]:
    """
    从时间序列中截取“截止 current_ts 的最近 lookback 条”。

    参数：
    - rows: 已按时间升序排列的数据
    - ts_list: rows 对应的时间列表
    - current_ts: 当前时刻
    - lookback: 需要向后回看的数量

    返回：
    - 满足 ts <= current_ts 的最后 lookback 条
    """
    if lookback <= 0:
        return []

    right = bisect_right(ts_list, current_ts)
    left = max(0, right - lookback)
    return rows[left:right]