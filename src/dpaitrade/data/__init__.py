"""
data 包。

这里提供 CSV 数据加载与时间切片工具。
"""

from .loaders import OHLCRow, load_ohlc_csv, slice_recent_rows

__all__ = [
    "OHLCRow",
    "load_ohlc_csv",
    "slice_recent_rows",
]