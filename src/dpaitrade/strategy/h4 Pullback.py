from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from dpaitrade.core.types import Direction


@dataclass(slots=True)
class H4Bar:
    """
    H4 周期 K 线数据。
    """

    open: float
    high: float
    low: float
    close: float


@dataclass(slots=True)
class H4PullbackResult:
    """
    H4 回调识别结果。

    字段说明：
    - pullback_active: 是否正处于趋势中的回调阶段
    - boundary_zone: 是否位于震荡边界区域
    - preferred_direction: 当前更偏向的潜在交易方向
    - pullback_depth_ratio: 回调深度比例
    - reason: 结果解释
    """

    pullback_active: bool
    boundary_zone: bool
    preferred_direction: Direction
    pullback_depth_ratio: float
    reason: str


class H4PullbackDetector:
    """
    H4 回调阶段识别器。

    目标：
    - 当 D1 已经判断为趋势时，识别 H4 是否处于“顺势回调”阶段
    - 当 D1 偏向震荡或不明确时，识别 H4 是否处于震荡边界区域

    这里的实现保持朴素：
    - 根据最近 swing_lookback 根 H4 K 线的区间确定 swing high / low
    - 结合 D1 bias 判断当前回调是否位于合理区间
    - 如果不是趋势思路，则判断价格是否接近区间边界
    """

    def __init__(
        self,
        swing_lookback: int = 24,
        pullback_min_ratio: float = 0.25,
        pullback_max_ratio: float = 0.75,
        boundary_near_ratio: float = 0.15,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.swing_lookback = swing_lookback
        self.pullback_min_ratio = pullback_min_ratio
        self.pullback_max_ratio = pullback_max_ratio
        self.boundary_near_ratio = boundary_near_ratio
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.strategy.h4_pullback")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    def detect(
        self,
        bars: list[H4Bar],
        d1_regime: str,
        d1_bias: Direction,
    ) -> H4PullbackResult:
        """
        识别 H4 是否进入回调阶段或震荡边界区域。

        参数：
        - bars: 按时间升序排列的 H4 K 线
        - d1_regime: 上层 D1 模块输出的市场状态
        - d1_bias: 上层 D1 模块输出的方向偏置
        """
        if len(bars) < max(5, self.swing_lookback):
            reason = (
                f"H4 数据不足，当前仅有 {len(bars)} 根，"
                f"少于识别所需的 {self.swing_lookback} 根，返回默认结果"
            )
            self.logger.info(reason)
            return H4PullbackResult(
                pullback_active=False,
                boundary_zone=False,
                preferred_direction="neutral",
                pullback_depth_ratio=0.0,
                reason=reason,
            )

        window = bars[-self.swing_lookback :]
        swing_high = max(bar.high for bar in window)
        swing_low = min(bar.low for bar in window)
        last_close = window[-1].close
        swing_range = max(swing_high - swing_low, 1e-8)

        # 价格在整个区间中的位置，越接近 1 越靠近上沿，越接近 0 越靠近下沿
        position_ratio = (last_close - swing_low) / swing_range

        # 回调深度定义：
        # - 多头趋势里，越从高点回落，深度越大
        # - 空头趋势里，越从低点反弹，深度越大
        if d1_bias == "long":
            pullback_depth_ratio = 1.0 - position_ratio
        elif d1_bias == "short":
            pullback_depth_ratio = position_ratio
        else:
            pullback_depth_ratio = 0.0

        pullback_depth_ratio = max(0.0, min(1.0, round(pullback_depth_ratio, 4)))

        if d1_regime == "trend" and d1_bias in ("long", "short"):
            is_pullback = self.pullback_min_ratio <= pullback_depth_ratio <= self.pullback_max_ratio
            preferred_direction: Direction = d1_bias if is_pullback else "neutral"

            reason = (
                f"H4 趋势回调识别：D1状态={d1_regime}，D1偏置={d1_bias}，"
                f"区间位置={position_ratio:.4f}，回调深度={pullback_depth_ratio:.4f}，"
                f"结果={'处于回调阶段' if is_pullback else '未处于理想回调阶段'}"
            )
            self.logger.info(reason)
            return H4PullbackResult(
                pullback_active=is_pullback,
                boundary_zone=False,
                preferred_direction=preferred_direction,
                pullback_depth_ratio=pullback_depth_ratio,
                reason=reason,
            )

        # 非趋势场景下，判断是否接近震荡边界区域
        near_upper = position_ratio >= 1.0 - self.boundary_near_ratio
        near_lower = position_ratio <= self.boundary_near_ratio
        boundary_zone = near_upper or near_lower

        preferred_direction = "neutral"
        if near_lower:
            preferred_direction = "long"
        elif near_upper:
            preferred_direction = "short"

        reason = (
            f"H4 震荡边界识别：D1状态={d1_regime}，D1偏置={d1_bias}，"
            f"区间位置={position_ratio:.4f}，boundary_zone={boundary_zone}，"
            f"preferred_direction={preferred_direction}"
        )
        self.logger.info(reason)
        return H4PullbackResult(
            pullback_active=False,
            boundary_zone=boundary_zone,
            preferred_direction=preferred_direction,
            pullback_depth_ratio=pullback_depth_ratio,
            reason=reason,
        )
