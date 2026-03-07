from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

PrimaryBias = Literal["long", "short", "neutral"]
Phase = Literal["impulse", "pullback", "range", "reversal_candidate", "unknown"]


@dataclass(slots=True)
class StructureState:
    """
    统一结构状态对象。

    设计原则：
    - 不按 D1 / H4 / M15 区分状态语言
    - 同一套字段适用于任意周期
    - 周期差异通过配置体现，而不是通过状态定义体现
    """

    timeframe: str
    ts: datetime

    primary_bias: PrimaryBias
    phase: Phase

    trend_score: float
    range_score: float
    confidence: float

    last_swing_high: Optional[float]
    last_swing_low: Optional[float]

    structure_high_broken: bool
    structure_low_broken: bool

    is_range_like: bool
    constraint_upper: Optional[float]
    constraint_lower: Optional[float]
    boundary_tolerance: float
    near_upper: bool
    near_lower: bool

    in_value_zone: bool
    continuation_ready: bool
    reversal_warning: bool

    reason: str