"""
backtest 包。

这里提供回测引擎及相关基础能力。
当前阶段先暴露主回测引擎与默认占位实现。
"""

from .engine import BacktestEngine, DefaultRiskManager, NoopExecutionSimulator

__all__ = [
    "BacktestEngine",
    "DefaultRiskManager",
    "NoopExecutionSimulator",
]
