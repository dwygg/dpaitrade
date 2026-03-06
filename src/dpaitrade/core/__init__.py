"""
core 包。

这里集中定义项目中的核心数据结构与公共类型。
"""

from .types import (
    AgentDecision,
    BacktestResult,
    BacktestStep,
    CandidateSignal,
    Direction,
    MarketState,
    PortfolioState,
    Regime,
    RiskDecision,
    TradeRecord,
)

__all__ = [
    "Direction",
    "Regime",
    "MarketState",
    "CandidateSignal",
    "AgentDecision",
    "RiskDecision",
    "PortfolioState",
    "TradeRecord",
    "BacktestStep",
    "BacktestResult",
]
