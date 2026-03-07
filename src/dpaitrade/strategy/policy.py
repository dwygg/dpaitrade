from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dpaitrade.core.types import CandidateSignal
from dpaitrade.structure import StructureState
from dpaitrade.structure import GenericBar

@dataclass(slots=True)
class StrategyContext:
    ts: datetime
    symbol: str
    entry_price: float
    atr: float
    spread: float
    volatility_score: float = 0.0
    # 新增：当前位置与低周期原始 bars
    mid_price: float = 0.0
    low_tf_bars: list[GenericBar] | None = None


@dataclass(slots=True)
class TrendContinuationPolicyConfig:
    """
    趋势延续策略配置。

    当前版本：
    - 只做顺势
    - 可分别开关多/空
    - 中周期负责候选区
    - 小周期负责延续确认
    """

    allow_long: bool = False
    allow_short: bool = True

    stop_buffer_atr_ratio: float = 0.10
    stop_buffer_spread_multiple: float = 1.5

    min_trend_score_high_tf: float = 0.40
    # 旧版 low_tf trend_score 约束可以保留，但不再要求 impulse
    min_trend_score_low_tf: float = 0.20

    # 新增：低周期反转触发参数
    low_tf_trigger_lookback: int = 8
    low_tf_breakout_buffer_atr_ratio: float = 0.03
    low_tf_range_entry_guard_atr_ratio: float = 0.80


class TrendContinuationPolicy:
    """
    趋势延续策略组合器。

    输入：
    - 高周期状态
    - 中周期状态
    - 低周期状态
    - 当前上下文

    输出：
    - CandidateSignal 或 None
    """

    def __init__(
        self,
        config: Optional[TrendContinuationPolicyConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or TrendContinuationPolicyConfig()
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.strategy.policy")
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

    def generate_signal(
        self,
        high_tf: StructureState,
        mid_tf: StructureState,
        low_tf: StructureState,
        ctx: StrategyContext,
    ) -> Optional[CandidateSignal]:
        """
        生成候选信号。

        当前策略逻辑：
        - 高周期定主方向
        - 中周期进入 value zone
        - 小周期重新转向原趋势
        """

        if high_tf.primary_bias == "short":
            return self._build_short_signal(high_tf, mid_tf, low_tf, ctx)

        if high_tf.primary_bias == "long":
            return self._build_long_signal(high_tf, mid_tf, low_tf, ctx)

        self.logger.info("未生成候选信号：高周期主方向为 neutral")
        return None

    def _check_short_reversal_trigger(
        self,
        mid_tf: StructureState,
        low_tf: StructureState,
        ctx: StrategyContext,
    ) -> tuple[bool, str]:
        """
        空头入场触发：

        核心思想：
        - 中周期必须在 value zone / 上缘附近
        - 小周期不能已经跌出去太远
        - 小周期需要出现“反弹失败后的第一下向下触发”
        """
        bars = ctx.low_tf_bars or []
        if len(bars) < self.config.low_tf_trigger_lookback:
            return False, "低周期原始 bars 不足，无法做反转触发判断"

        recent = bars[-self.config.low_tf_trigger_lookback :]
        last_bar = recent[-1]

        # 1) 中周期位置必须正确：靠近上缘
        if not mid_tf.in_value_zone or not mid_tf.near_upper:
            return False, "中周期未处于做空 value zone 上缘"

        # 2) 避免价格已经离上缘太远再追空
        if mid_tf.constraint_upper is not None:
            distance_from_upper = max(mid_tf.constraint_upper - ctx.mid_price, 0.0)
            max_allowed_distance = max(ctx.atr * self.config.low_tf_range_entry_guard_atr_ratio, 1e-8)
            if distance_from_upper > max_allowed_distance:
                return False, "当前价格已离中周期上缘过远，放弃追空"

        # 3) 找最近小级别“局部高点后”的触发低点
        # 用最近几根（去掉最后一根）找一个局部高点，再看高点后的最低点是否被最后一根跌破
        core = recent[:-1]
        if len(core) < 4:
            return False, "低周期窗口过短"

        swing_high_bar = max(core, key=lambda b: b.high)
        swing_high_idx = core.index(swing_high_bar)

        # 局部高点必须不是太早形成，否则说明结构已经过老
        if swing_high_idx >= len(core) - 1:
            return False, "低周期尚未形成有效局部高点后的回落结构"

        post_high_bars = core[swing_high_idx + 1 :]
        if len(post_high_bars) < 2:
            return False, "局部高点后样本不足，无法确认反转触发"

        trigger_low = min(b.low for b in post_high_bars)
        breakout_buffer = ctx.atr * self.config.low_tf_breakout_buffer_atr_ratio

        # 4) 最后一根收盘跌破触发低点 => 认为反转开始
        if last_bar.close >= trigger_low - breakout_buffer:
            return False, "低周期尚未跌破反转触发低点"

        # 5) 最后一根最好是阴线，避免假破
        if last_bar.close >= last_bar.open:
            return False, "低周期最后一根不是转弱阴线"

        return True, (
            f"中周期处于做空区，低周期完成空头反转触发："
            f"局部高点={swing_high_bar.high:.5f}，触发低点={trigger_low:.5f}，"
            f"当前收盘={last_bar.close:.5f}"
        )


    def _check_long_reversal_trigger(
        self,
        mid_tf: StructureState,
        low_tf: StructureState,
        ctx: StrategyContext,
    ) -> tuple[bool, str]:
        """
        多头入场触发：

        - 中周期必须在 value zone / 下缘附近
        - 小周期不能已经涨出去太远
        - 小周期需要出现“回调失败后的第一下向上触发”
        """
        bars = ctx.low_tf_bars or []
        if len(bars) < self.config.low_tf_trigger_lookback:
            return False, "低周期原始 bars 不足，无法做反转触发判断"

        recent = bars[-self.config.low_tf_trigger_lookback :]
        last_bar = recent[-1]

        if not mid_tf.in_value_zone or not mid_tf.near_lower:
            return False, "中周期未处于做多 value zone 下缘"

        if mid_tf.constraint_lower is not None:
            distance_from_lower = max(ctx.mid_price - mid_tf.constraint_lower, 0.0)
            max_allowed_distance = max(ctx.atr * self.config.low_tf_range_entry_guard_atr_ratio, 1e-8)
            if distance_from_lower > max_allowed_distance:
                return False, "当前价格已离中周期下缘过远，放弃追多"

        core = recent[:-1]
        if len(core) < 4:
            return False, "低周期窗口过短"

        swing_low_bar = min(core, key=lambda b: b.low)
        swing_low_idx = core.index(swing_low_bar)

        if swing_low_idx >= len(core) - 1:
            return False, "低周期尚未形成有效局部低点后的回升结构"

        post_low_bars = core[swing_low_idx + 1 :]
        if len(post_low_bars) < 2:
            return False, "局部低点后样本不足，无法确认反转触发"

        trigger_high = max(b.high for b in post_low_bars)
        breakout_buffer = ctx.atr * self.config.low_tf_breakout_buffer_atr_ratio

        if last_bar.close <= trigger_high + breakout_buffer:
            return False, "低周期尚未突破反转触发高点"

        if last_bar.close <= last_bar.open:
            return False, "低周期最后一根不是转强阳线"

        return True, (
            f"中周期处于做多区，低周期完成多头反转触发："
            f"局部低点={swing_low_bar.low:.5f}，触发高点={trigger_high:.5f}，"
            f"当前收盘={last_bar.close:.5f}"
        )

    def _build_short_signal(
        self,
        high_tf: StructureState,
        mid_tf: StructureState,
        low_tf: StructureState,
        ctx: StrategyContext,
    ) -> Optional[CandidateSignal]:
        if not self.config.allow_short:
            self.logger.info("未生成候选信号：当前策略已关闭 short")
            return None

        if high_tf.primary_bias != "short":
            self.logger.info("未生成候选信号：高周期主方向不是 short")
            return None

        if high_tf.phase == "reversal_candidate":
            self.logger.info("未生成候选信号：高周期处于空头反转候选阶段")
            return None

        if high_tf.trend_score < self.config.min_trend_score_high_tf:
            self.logger.info(
                "未生成候选信号：高周期趋势强度不足 trend_score=%.4f",
                high_tf.trend_score,
            )
            return None

        # 中周期必须提供“位置”
        if not mid_tf.in_value_zone or not mid_tf.near_upper:
            self.logger.info("未生成候选信号：中周期尚未进入做空 value zone 上缘")
            return None

        if mid_tf.reversal_warning:
            self.logger.info("未生成候选信号：中周期存在反转警告")
            return None

        # 小周期不再要求已经进入 impulse，而是要求“反转起点触发”
        triggered, trigger_reason = self._check_short_reversal_trigger(
            mid_tf=mid_tf,
            low_tf=low_tf,
            ctx=ctx,
        )
        if not triggered:
            self.logger.info("未生成候选信号：%s", trigger_reason)
            return None

        stop_anchor = low_tf.last_swing_high or mid_tf.constraint_upper
        if stop_anchor is None:
            self.logger.info("未生成候选信号：缺少有效止损锚点")
            return None

        stop_buffer = max(
            ctx.atr * self.config.stop_buffer_atr_ratio,
            ctx.spread * self.config.stop_buffer_spread_multiple,
        )
        stop_loss = stop_anchor + stop_buffer
        entry_price = ctx.entry_price
        risk = stop_loss - entry_price

        if risk <= 0:
            self.logger.info(
                "未生成候选信号：空头止损锚点异常，entry=%.5f，sl=%.5f",
                entry_price,
                stop_loss,
            )
            return None

        signal = CandidateSignal(
            ts=ctx.ts,
            symbol=ctx.symbol,
            direction="short",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=None,
            rr_estimate=0.0,
            reason=(
                "趋势延续空头候选信号："
                f"高周期={high_tf.timeframe}/{high_tf.primary_bias}/{high_tf.phase}，"
                f"中周期进入候选做空区，"
                f"低周期完成反转起点触发；"
                f"{trigger_reason}"
            ),
            tags=["trend_continuation", "short", "value_zone", "reversal_trigger"],
            meta={
                "high_tf_phase": high_tf.phase,
                "mid_tf_phase": mid_tf.phase,
                "low_tf_phase": low_tf.phase,
                "regime": "trend",
                "policy": "trend_continuation_v2",
            },
        )
        self.logger.info(
            "已生成空头候选信号：entry=%.5f，stop_loss=%.5f",
            signal.entry_price,
            signal.stop_loss,
        )
        return signal

    def _build_long_signal(
        self,
        high_tf: StructureState,
        mid_tf: StructureState,
        low_tf: StructureState,
        ctx: StrategyContext,
    ) -> Optional[CandidateSignal]:
        if not self.config.allow_long:
            self.logger.info("未生成候选信号：当前策略已关闭 long")
            return None

        if high_tf.primary_bias != "long":
            self.logger.info("未生成候选信号：高周期主方向不是 long")
            return None

        if high_tf.phase == "reversal_candidate":
            self.logger.info("未生成候选信号：高周期处于多头反转候选阶段")
            return None

        if high_tf.trend_score < self.config.min_trend_score_high_tf:
            self.logger.info(
                "未生成候选信号：高周期趋势强度不足 trend_score=%.4f",
                high_tf.trend_score,
            )
            return None

        if not mid_tf.in_value_zone or not mid_tf.near_lower:
            self.logger.info("未生成候选信号：中周期尚未进入做多 value zone 下缘")
            return None

        if mid_tf.reversal_warning:
            self.logger.info("未生成候选信号：中周期存在反转警告")
            return None

        triggered, trigger_reason = self._check_long_reversal_trigger(
            mid_tf=mid_tf,
            low_tf=low_tf,
            ctx=ctx,
        )
        if not triggered:
            self.logger.info("未生成候选信号：%s", trigger_reason)
            return None

        stop_anchor = low_tf.last_swing_low or mid_tf.constraint_lower
        if stop_anchor is None:
            self.logger.info("未生成候选信号：缺少有效止损锚点")
            return None

        stop_buffer = max(
            ctx.atr * self.config.stop_buffer_atr_ratio,
            ctx.spread * self.config.stop_buffer_spread_multiple,
        )
        stop_loss = stop_anchor - stop_buffer
        entry_price = ctx.entry_price
        risk = entry_price - stop_loss

        if risk <= 0:
            self.logger.info(
                "未生成候选信号：多头止损锚点异常，entry=%.5f，sl=%.5f",
                entry_price,
                stop_loss,
            )
            return None

        signal = CandidateSignal(
            ts=ctx.ts,
            symbol=ctx.symbol,
            direction="long",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=None,
            rr_estimate=0.0,
            reason=(
                "趋势延续多头候选信号："
                f"高周期={high_tf.timeframe}/{high_tf.primary_bias}/{high_tf.phase}，"
                f"中周期进入候选做多区，"
                f"低周期完成反转起点触发；"
                f"{trigger_reason}"
            ),
            tags=["trend_continuation", "long", "value_zone", "reversal_trigger"],
            meta={
                "high_tf_phase": high_tf.phase,
                "mid_tf_phase": mid_tf.phase,
                "low_tf_phase": low_tf.phase,
                "regime": "trend",
                "policy": "trend_continuation_v2",
            },
        )
        self.logger.info(
            "已生成多头候选信号：entry=%.5f，stop_loss=%.5f",
            signal.entry_price,
            signal.stop_loss,
        )
        return signal