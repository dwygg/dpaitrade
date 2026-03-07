"""
strategy 包。

旧的按周期拆分策略模块已停用。
当前统一使用：
- StructureAnalyzer
- TrendContinuationPolicy
"""

from .policy import StrategyContext, TrendContinuationPolicy, TrendContinuationPolicyConfig

__all__ = [
    "StrategyContext",
    "TrendContinuationPolicy",
    "TrendContinuationPolicyConfig",
]