from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from dpaitrade.core.types import Direction


@dataclass(slots=True)
class M15Bar:
    """
    M15 周期 K 线数据。
    """

    open: float
    high: float
    low: float
    close: float


@dataclass(slots=True)
class M15EntryResult:
    """
    M15 入场识别结果。

    字段说明：
    - entry_ready: 是否满足入场条件
    - direction: 当前入场方向
    - entry_price: 建议入场价
    - stop_loss: 建议止损价
    - rr_estimate: 粗略估计的盈亏比
    - reason: 文字解释
    """

    entry_ready: bool
    direction: Direction
    entry_price: float
    stop_loss: float
    rr_estimate: float
    reason: str


class M15EntryDetector:
    """
    M15 入场识别器。

    当前版本先采用一个简化思路：
    - 对多头：寻找“回踩后重新向上收复近端高点”的结构确认
    - 对空头：寻找“反抽后重新向下跌破近端低点”的结构确认

    注意：
    这里还不是最终的市场结构模型，主要作用是提供一个稳定可扩展的入场骨架。
    后续可以替换为更严格的 BOS / CHoCH / 回踩确认逻辑。
    """

    def __init__(
        self,
        structure_lookback: int = 12,
        min_rr: float = 1.5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.structure_lookback = structure_lookback
        self.min_rr = min_rr
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.strategy.m15_entry")
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
        bars: list[M15Bar],
        preferred_direction: Direction,
    ) -> M15EntryResult:
        """
        识别 M15 是否出现入场确认。

        参数：
        - bars: 按时间升序排列的 M15 K 线
        - preferred_direction: 上层模块给出的偏好方向
        """
        if preferred_direction not in ("long", "short"):
            reason = f"M15 入场检测跳过：preferred_direction={preferred_direction}，当前不做入场确认"
            self.logger.info(reason)
            return M15EntryResult(
                entry_ready=False,
                direction="neutral",
                entry_price=0.0,
                stop_loss=0.0,
                rr_estimate=0.0,
                reason=reason,
            )

        if len(bars) < max(6, self.structure_lookback):
            reason = (
                f"M15 数据不足，当前仅有 {len(bars)} 根，"
                f"少于识别所需的 {self.structure_lookback} 根，返回默认结果"
            )
            self.logger.info(reason)
            return M15EntryResult(
                entry_ready=False,
                direction="neutral",
                entry_price=0.0,
                stop_loss=0.0,
                rr_estimate=0.0,
                reason=reason,
            )

        window = bars[-self.structure_lookback :]
        last_bar = window[-1]
        prev_window = window[:-1]

        if preferred_direction == "long":
            reference_high = max(bar.high for bar in prev_window[-5:])
            local_swing_low = min(bar.low for bar in prev_window[-5:])
            entry_ready = last_bar.close > reference_high
            entry_price = last_bar.close if entry_ready else 0.0
            stop_loss = local_swing_low if entry_ready else 0.0
            risk = max(entry_price - stop_loss, 1e-8) if entry_ready else 0.0
            reward = risk * self.min_rr if entry_ready else 0.0
            rr_estimate = round(reward / max(risk, 1e-8), 4) if entry_ready else 0.0

            reason = (
                f"M15 多头入场识别：收盘价={last_bar.close:.5f}，"
                f"参考高点={reference_high:.5f}，局部低点={local_swing_low:.5f}，"
                f"结果={'满足入场' if entry_ready else '未满足入场'}"
            )
            self.logger.info(reason)
            return M15EntryResult(
                entry_ready=entry_ready,
                direction="long" if entry_ready else "neutral",
                entry_price=entry_price,
                stop_loss=stop_loss,
                rr_estimate=rr_estimate,
                reason=reason,
            )

        reference_low = min(bar.low for bar in prev_window[-5:])
        local_swing_high = max(bar.high for bar in prev_window[-5:])
        entry_ready = last_bar.close < reference_low
        entry_price = last_bar.close if entry_ready else 0.0
        stop_loss = local_swing_high if entry_ready else 0.0
        risk = max(stop_loss - entry_price, 1e-8) if entry_ready else 0.0
        reward = risk * self.min_rr if entry_ready else 0.0
        rr_estimate = round(reward / max(risk, 1e-8), 4) if entry_ready else 0.0

        reason = (
            f"M15 空头入场识别：收盘价={last_bar.close:.5f}，"
            f"参考低点={reference_low:.5f}，局部高点={local_swing_high:.5f}，"
            f"结果={'满足入场' if entry_ready else '未满足入场'}"
        )
        self.logger.info(reason)
        return M15EntryResult(
            entry_ready=entry_ready,
            direction="short" if entry_ready else "neutral",
            entry_price=entry_price,
            stop_loss=stop_loss,
            rr_estimate=rr_estimate,
            reason=reason,
        )
