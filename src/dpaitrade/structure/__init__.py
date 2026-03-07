"""
structure 包。

这里提供统一结构状态对象与统一结构分析器。
"""

from .analyzer import GenericBar, StructureAnalyzer, StructureAnalyzerConfig
from .state import StructureState

__all__ = [
    "GenericBar",
    "StructureAnalyzer",
    "StructureAnalyzerConfig",
    "StructureState",
]