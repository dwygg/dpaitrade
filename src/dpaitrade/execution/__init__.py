"""
execution 包。

这里提供交易执行与成交仿真模块。
当前阶段先暴露简单执行模拟器与其配置。
"""

from .simulator import SimpleExecutionSimulator, SimulationConfig

__all__ = [
    "SimpleExecutionSimulator",
    "SimulationConfig",
]
