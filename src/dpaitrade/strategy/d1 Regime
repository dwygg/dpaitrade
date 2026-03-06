from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from dpaitrade.core.types import Direction, Regime


@dataclass(slots=True)
class D1Bar:
    """
    D1 周期 K 线数据。

    当前只保留最常用字段，便于先把策略骨架搭起来。
    后续如果要加入成交量、会话、事件标签等信息，可以继续扩展。
    """

    open: float
    high: float
    low: float
    close: float


@dataclass(slots=True)
class D1RegimeResult:
    """
    D1 市场状态识别结果。

    字段说明：
    - regime: 当前市场属于趋势 / 震荡 / 未知
    - bias: 当前方向偏置，多 / 空 / 中性
    - trend_score: 趋势强度评分，范围建议 0~1
    - range_score: 震荡强度评分，范围建议 0~1
    - reason: 文字解释，便于日志与调试
    """

    regime: Regime
    bias: Direction
    trend_score: float
    range_score: float
    reason: str


class D1RegimeDetector:
    """
    D1 市场状态识别器。

    设计原则：
    1. 先用朴素但稳定的规则，避免一开始过度复杂
    2. 输出结果要结构化，方便后续回测与 Agent 使用
    3. 允许未来替换成更复杂的方法，例如机器学习分类器

    当前逻辑：
    - 用最近 lookback 根 D1 K 线估计方向性与波动范围
    - 当净位移 / 总波动比例较高时，更偏向趋势
    - 当比例较低时，更偏向震荡
    """

    def __init__(
        self,
        lookback: int = 20,
        trend_threshold: float = 0.35,
        range_threshold: float = 0.20,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.lookback = lookback
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.strategy.d1_regime")
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

    def detect(self, bars: list[D1Bar]) -> D1RegimeResult:
        """
        识别 D1 市场状态。

        参数：
        - bars: 按时间升序排列的 D1 K 线列表

        返回：
        - D1RegimeResult
        """
        if len(bars) < max(3, self.lookback):
            reason = (
                f"D1 数据不足，当前仅有 {len(bars)} 根，"
                f"少于识别所需的 {self.lookback} 根，返回 unknown"
            )
            self.logger.info(reason)
            return D1RegimeResult(
                regime="unknown",
                bias="neutral",
                trend_score=0.0,
                range_score=0.0,
                reason=reason,
            )

        window = bars[-self.lookback :]
        first_close = window[0].close
        last_close = window[-1].close

        net_move = last_close - first_close
        total_range = sum(max(bar.high - bar.low, 1e-8) for bar in window)
        directional_ratio = abs(net_move) / max(total_range, 1e-8)

        # 用最近一段收盘价单调性作为趋势辅助判断
        up_count = 0
        down_count = 0
        for prev_bar, curr_bar in zip(window[:-1], window[1:]):
            if curr_bar.close > prev_bar.close:
                up_count += 1
            elif curr_bar.close < prev_bar.close:
                down_count += 1

        total_steps = max(len(window) - 1, 1)
        dominant_step_ratio = max(up_count, down_count) / total_steps

        trend_score = min(1.0, round(directional_ratio * 1.6 + dominant_step_ratio * 0.4, 4))
        range_score = min(1.0, round((1.0 - directional_ratio) * 0.7 + (1.0 - dominant_step_ratio) * 0.3, 4))

        bias: Direction = "neutral"
        if net_move > 0:
            bias = "long"
        elif net_move < 0:
            bias = "short"

        if directional_ratio >= self.trend_threshold:
            regime: Regime = "trend"
            reason = (
                f"D1 识别为趋势：净位移={net_move:.5f}，"
                f"方向比率={directional_ratio:.4f}，主导步进占比={dominant_step_ratio:.4f}"
            )
        elif directional_ratio <= self.range_threshold:
            regime = "range"
            bias = "neutral"
            reason = (
                f"D1 识别为震荡：净位移={net_move:.5f}，"
                f"方向比率={directional_ratio:.4f}，主导步进占比={dominant_step_ratio:.4f}"
            )
        else:
            regime = "unknown"
            if directional_ratio < 0.28:
                bias = "neutral"
            reason = (
                f"D1 状态不够明确：净位移={net_move:.5f}，"
                f"方向比率={directional_ratio:.4f}，主导步进占比={dominant_step_ratio:.4f}"
            )

        self.logger.info(reason)
        return D1RegimeResult(
            regime=regime,
            bias=bias,
            trend_score=trend_score,
            range_score=range_score,
            reason=reason,
        )
