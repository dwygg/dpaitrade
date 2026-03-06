"""
agent 包。

这里定义交易系统中的代理接口与代理实现。
当前阶段主要用于：
- 直通代理
- 简单规则过滤代理
- 后续的 LLM 过滤代理
"""

from .interface import BaseAgent, PassthroughAgent, SimpleRuleFilterAgent

__all__ = [
    "BaseAgent",
    "PassthroughAgent",
    "SimpleRuleFilterAgent",
]
