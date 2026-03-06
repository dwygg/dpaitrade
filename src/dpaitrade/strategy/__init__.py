"""
strategy 包。

这里定义多周期策略识别模块，包括：
- D1 市场状态识别
- H4 回调 / 边界识别
- M15 入场确认
- 候选信号组装
"""

from .d1_regime import D1Bar, D1RegimeDetector, D1RegimeResult
from .h4_pullback import H4Bar, H4PullbackDetector, H4PullbackResult
from .m15_entry import M15Bar, M15EntryDetector, M15EntryResult
from .signal_builder import SignalBuildContext, SignalBuilder

__all__ = [
    "D1Bar",
    "D1RegimeDetector",
    "D1RegimeResult",
    "H4Bar",
    "H4PullbackDetector",
    "H4PullbackResult",
    "M15Bar",
    "M15EntryDetector",
    "M15EntryResult",
    "SignalBuildContext",
    "SignalBuilder",
]
