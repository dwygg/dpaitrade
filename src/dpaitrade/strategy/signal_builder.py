from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dpaitrade.core.types import CandidateSignal, MarketState
from dpaitrade.strategy.d1_regime import D1RegimeResult
from dpaitrade.strategy.h4_pullback import H4PullbackResult
from dpaitrade.strategy.m15_entry import M15EntryResult


@dataclass(slots=True)
class SignalBuildContext:
    """
    信号组装上下文。

    用于把 D1 / H4 / M15 各模块的输出汇总到同一个地方，
    方便后续统一构建 MarketState 和 CandidateSignal。
    """

    ts: datetime
    symbol: str
    atr: float = 0.0
    spread: float = 0.0
    volatility_score: float = 0.0


class SignalBuilder:
    """
    信号组装器。

    职责：
    1. 将 D1 / H4 / M15 识别结果拼装成统一的 MarketState
    2. 在条件满足时，构建 CandidateSignal
    3. 尽量把“跨模块判断”放在这里做，避免单个模块职责过重

    当前第一版的组装逻辑：
    - D1 趋势 + H4 回调 + M15 入场确认 -> 生成趋势候选信号
    - D1 非趋势 + H4 边界 + M15 入场确认 -> 生成边界候选信号
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.strategy.signal_builder")
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

    def build_market_state(
        self,
        ctx: SignalBuildContext,
        d1_result: D1RegimeResult,
        h4_result: H4PullbackResult,
        m15_result: M15EntryResult,
    ) -> MarketState:
        """
        构建统一市场状态对象。
        """
        market_state = MarketState(
            ts=ctx.ts,
            symbol=ctx.symbol,
            d1_regime=d1_result.regime,
            d1_bias=d1_result.bias,
            h4_pullback_active=h4_result.pullback_active,
            h4_boundary_zone=h4_result.boundary_zone,
            m15_entry_ready=m15_result.entry_ready,
            m15_entry_direction=m15_result.direction,
            atr=ctx.atr,
            spread=ctx.spread,
            volatility_score=ctx.volatility_score,
            meta={
                "d1_reason": d1_result.reason,
                "h4_reason": h4_result.reason,
                "m15_reason": m15_result.reason,
                "trend_score": d1_result.trend_score,
                "range_score": d1_result.range_score,
                "pullback_depth_ratio": h4_result.pullback_depth_ratio,
            },
        )
        self.logger.info(
            "已构建 MarketState：symbol=%s，D1状态=%s，D1偏置=%s，H4回调=%s，H4边界=%s，M15入场=%s",
            market_state.symbol,
            market_state.d1_regime,
            market_state.d1_bias,
            market_state.h4_pullback_active,
            market_state.h4_boundary_zone,
            market_state.m15_entry_ready,
        )
        return market_state

    def build_candidate_signal(
        self,
        ctx: SignalBuildContext,
        d1_result: D1RegimeResult,
        h4_result: H4PullbackResult,
        m15_result: M15EntryResult,
    ) -> Optional[CandidateSignal]:
        """
        依据 D1 / H4 / M15 结果生成候选信号。

        返回：
        - CandidateSignal: 当前可考虑的候选信号
        - None: 当前没有可交易候选
        """
        if not m15_result.entry_ready:
            self.logger.info("未生成候选信号：M15 尚未给出入场确认")
            return None

        # 场景一：趋势中的回调
        if d1_result.regime == "trend" and h4_result.pullback_active:
            if m15_result.direction != d1_result.bias:
                self.logger.info(
                    "未生成候选信号：趋势场景下 M15 方向=%s 与 D1 偏置=%s 不一致",
                    m15_result.direction,
                    d1_result.bias,
                )
                return None

            signal = CandidateSignal(
                ts=ctx.ts,
                symbol=ctx.symbol,
                direction=m15_result.direction,
                entry_price=m15_result.entry_price,
                stop_loss=m15_result.stop_loss,
                take_profit=self._estimate_take_profit(m15_result),
                rr_estimate=m15_result.rr_estimate,
                reason=(
                    f"趋势候选信号：D1={d1_result.regime}/{d1_result.bias}，"
                    f"H4处于回调阶段，M15完成{m15_result.direction}入场确认"
                ),
                tags=["trend", "pullback", "m15_confirmed"],
                meta={
                    "d1_reason": d1_result.reason,
                    "h4_reason": h4_result.reason,
                    "m15_reason": m15_result.reason,
                },
            )
            self.logger.info(
                "已生成趋势候选信号：方向=%s，入场价=%.5f，止损=%.5f，RR=%.4f",
                signal.direction,
                signal.entry_price,
                signal.stop_loss,
                signal.rr_estimate,
            )
            return signal

        # 场景二：震荡边界区域
        if d1_result.regime != "trend" and h4_result.boundary_zone:
            if h4_result.preferred_direction not in ("long", "short"):
                self.logger.info("未生成候选信号：H4 虽处于边界区域，但未给出明确方向")
                return None

            if m15_result.direction != h4_result.preferred_direction:
                self.logger.info(
                    "未生成候选信号：震荡边界场景下 M15 方向=%s 与 H4 偏好方向=%s 不一致",
                    m15_result.direction,
                    h4_result.preferred_direction,
                )
                return None

            signal = CandidateSignal(
                ts=ctx.ts,
                symbol=ctx.symbol,
                direction=m15_result.direction,
                entry_price=m15_result.entry_price,
                stop_loss=m15_result.stop_loss,
                take_profit=self._estimate_take_profit(m15_result),
                rr_estimate=m15_result.rr_estimate,
                reason=(
                    f"边界候选信号：D1={d1_result.regime}/{d1_result.bias}，"
                    f"H4处于震荡边界，M15完成{m15_result.direction}入场确认"
                ),
                tags=["range", "boundary", "m15_confirmed"],
                meta={
                    "d1_reason": d1_result.reason,
                    "h4_reason": h4_result.reason,
                    "m15_reason": m15_result.reason,
                },
            )
            self.logger.info(
                "已生成边界候选信号：方向=%s，入场价=%.5f，止损=%.5f，RR=%.4f",
                signal.direction,
                signal.entry_price,
                signal.stop_loss,
                signal.rr_estimate,
            )
            return signal

        self.logger.info(
            "未生成候选信号：当前跨周期条件未对齐，D1=%s，H4回调=%s，H4边界=%s，M15方向=%s",
            d1_result.regime,
            h4_result.pullback_active,
            h4_result.boundary_zone,
            m15_result.direction,
        )
        return None

    @staticmethod
    def _estimate_take_profit(m15_result: M15EntryResult) -> float | None:
        """
        根据 RR 粗略估算止盈价。

        当前阶段只做简单推算，后续可以替换成：
        - 前高 / 前低
        - 流动性目标位
        - 多目标位分批止盈
        """
        if not m15_result.entry_ready:
            return None

        if m15_result.direction == "long":
            risk = max(m15_result.entry_price - m15_result.stop_loss, 1e-8)
            return m15_result.entry_price + risk * m15_result.rr_estimate

        if m15_result.direction == "short":
            risk = max(m15_result.stop_loss - m15_result.entry_price, 1e-8)
            return m15_result.entry_price - risk * m15_result.rr_estimate

        return None
