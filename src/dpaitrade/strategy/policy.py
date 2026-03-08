from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dpaitrade.core.types import CandidateSignal
from dpaitrade.structure import GenericBar
from dpaitrade.structure import StructureState


@dataclass(slots=True)
class StrategyContext:
    ts: datetime
    symbol: str
    entry_price: float
    atr: float
    spread: float
    volatility_score: float = 0.0
    mid_price: float = 0.0
    low_tf_bars: list[GenericBar] | None = None
    dominant_tf_bars: list[GenericBar] | None = None
    higher_tf_atr: float = 0.0


@dataclass(slots=True)
class TrendContinuationPolicyConfig:
    """单周期趋势延续策略配置。"""

    allow_long: bool = True
    allow_short: bool = True

    dominant_timeframe: str = "low"

    stop_buffer_atr_ratio: float = 0.10
    stop_buffer_spread_multiple: float = 1.5

    min_trend_score_trading_tf: float = 0.18

    trigger_lookback: int = 12

    # 顺势确认 buffer。
    breakout_buffer_atr_ratio: float = 0.03

    # 回调至少达到一定深度，避免在趋势中途噪音直接追价。
    min_pullback_atr_ratio: float = 0.35

    # 回调深度落在前一推动腿的合理区间内，才允许做顺势恢复。
    pullback_retrace_min_ratio: float = 0.30
    pullback_retrace_max_ratio: float = 0.75

    # 最后入场价不能离回调低/高点太远，避免恢复后追得过晚。
    max_entry_distance_atr_ratio: float = 1.20

    # 只允许在指定 phase 入场；空 tuple = 不限制。
    # "range" 阶段是震荡区间，不适合趋势延续策略。
    allowed_phases: tuple[str, ...] = field(default_factory=lambda: ("trend", "continuation"))

    # 确认 bar 实体至少占 ATR 的比例，过小视为无力确认。
    min_confirm_bar_body_atr_ratio: float = 0.15

    # 区间模式：在 range phase 时，以 constraint_upper/lower 作为止盈目标。
    # TP 位置 = 区间边界向内缩进 range_tp_buffer_atr_ratio × ATR，避免在边界被拒绝。
    # 设为 0.0 则 TP = 恰好贴着边界。
    range_tp_buffer_atr_ratio: float = 0.05

    # 是否要求高周期方向与入场方向一致。
    # higher_tf_for_align: "mid"=H4, "high"=D1, ""=不做对齐检查。
    require_higher_tf_bias_align: bool = True
    higher_tf_for_align: str = "mid"

    require_reversal_warning_clear: bool = False


class TrendContinuationPolicy:
    """单周期趋势延续策略。"""

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

    def _effective_atr(self, ctx: StrategyContext) -> float:
        return max(ctx.atr, ctx.spread, 1e-8)

    def _select_dominant_state(
        self,
        high_tf: StructureState,
        mid_tf: StructureState,
        low_tf: StructureState,
    ) -> StructureState:
        mapping = {
            "high": high_tf,
            "mid": mid_tf,
            "low": low_tf,
        }
        return mapping.get(self.config.dominant_timeframe, low_tf)

    def _recent_bars(self, ctx: StrategyContext) -> list[GenericBar]:
        bars = ctx.dominant_tf_bars or ctx.low_tf_bars or []
        if len(bars) < self.config.trigger_lookback:
            return []
        return bars[-self.config.trigger_lookback :]

    @staticmethod
    def _find_latest_local_low(bars: list[GenericBar]) -> Optional[int]:
        if len(bars) < 3:
            return None
        for idx in range(len(bars) - 2, 0, -1):
            if bars[idx].low <= bars[idx - 1].low and bars[idx].low <= bars[idx + 1].low:
                return idx
        return None

    @staticmethod
    def _find_latest_local_high(bars: list[GenericBar]) -> Optional[int]:
        if len(bars) < 3:
            return None
        for idx in range(len(bars) - 2, 0, -1):
            if bars[idx].high >= bars[idx - 1].high and bars[idx].high >= bars[idx + 1].high:
                return idx
        return None

    def _check_long_continuation_trigger(
        self,
        state: StructureState,
        ctx: StrategyContext,
    ) -> tuple[bool, str, Optional[float]]:
        recent = self._recent_bars(ctx)
        if len(recent) < 4:
            return False, "主导周期原始 bars 不足，无法做顺势回调入场判断", None

        atr = self._effective_atr(ctx)
        confirm_buffer = atr * self.config.breakout_buffer_atr_ratio
        core = recent[:-1]
        last_bar = recent[-1]
        prev_bar = recent[-2]

        local_low_idx = self._find_latest_local_low(core)
        if local_low_idx is None:
            return False, "最近窗口未找到有效回调低点", None

        pullback_low_bar = core[local_low_idx]
        pre_pullback_bars = core[: local_low_idx + 1]
        if len(pre_pullback_bars) < 2:
            return False, "回调前推动样本不足", None

        pullback_origin_high = max(b.high for b in pre_pullback_bars)
        pullback_origin_low = min(b.low for b in pre_pullback_bars)
        pullback_depth = pullback_origin_high - pullback_low_bar.low
        if pullback_depth < atr * self.config.min_pullback_atr_ratio:
            return False, "最近回调幅度过小，视为噪音", None

        impulse_range = max(pullback_origin_high - pullback_origin_low, atr)
        retrace_ratio = pullback_depth / max(impulse_range, 1e-8)
        if not (
            self.config.pullback_retrace_min_ratio
            <= retrace_ratio
            <= self.config.pullback_retrace_max_ratio
        ):
            return False, (
                f"回调深度不在入场区间内 retrace={retrace_ratio:.2f}"
            ), None

        if last_bar.close <= last_bar.open:
            return False, "最后一根不是顺势确认阳线", None
        if last_bar.close <= prev_bar.high + confirm_buffer:
            return False, "最后一根尚未收上前一根高点确认位", None
        bar_body = last_bar.close - last_bar.open
        if bar_body < atr * self.config.min_confirm_bar_body_atr_ratio:
            return False, f"确认阳线实体过小 body={bar_body:.5f}，视为无力确认", None

        entry_extension = last_bar.close - pullback_low_bar.low
        if entry_extension > atr * self.config.max_entry_distance_atr_ratio:
            return False, "当前价格离回调低点过远，放弃追多", None

        stop_anchor = getattr(state, "last_swing_low", None) or pullback_low_bar.low
        return True, (
            f"单周期多头延续触发："
            f"回调低点={pullback_low_bar.low:.5f}，"
            f"回调深度比={retrace_ratio:.2f}，"
            f"确认高点={prev_bar.high:.5f}，"
            f"当前收盘={last_bar.close:.5f}"
        ), stop_anchor

    def _check_short_continuation_trigger(
        self,
        state: StructureState,
        ctx: StrategyContext,
    ) -> tuple[bool, str, Optional[float]]:
        recent = self._recent_bars(ctx)
        if len(recent) < 4:
            return False, "主导周期原始 bars 不足，无法做顺势反弹入场判断", None

        atr = self._effective_atr(ctx)
        confirm_buffer = atr * self.config.breakout_buffer_atr_ratio
        core = recent[:-1]
        last_bar = recent[-1]
        prev_bar = recent[-2]

        local_high_idx = self._find_latest_local_high(core)
        if local_high_idx is None:
            return False, "最近窗口未找到有效反弹高点", None

        pullback_high_bar = core[local_high_idx]
        pre_pullback_bars = core[: local_high_idx + 1]
        if len(pre_pullback_bars) < 2:
            return False, "反弹前推动样本不足", None

        pullback_origin_low = min(b.low for b in pre_pullback_bars)
        pullback_origin_high = max(b.high for b in pre_pullback_bars)
        pullback_depth = pullback_high_bar.high - pullback_origin_low
        if pullback_depth < atr * self.config.min_pullback_atr_ratio:
            return False, "最近反弹幅度过小，视为噪音", None

        impulse_range = max(pullback_origin_high - pullback_origin_low, atr)
        retrace_ratio = pullback_depth / max(impulse_range, 1e-8)
        if not (
            self.config.pullback_retrace_min_ratio
            <= retrace_ratio
            <= self.config.pullback_retrace_max_ratio
        ):
            return False, (
                f"反弹深度不在入场区间内 retrace={retrace_ratio:.2f}"
            ), None

        if last_bar.close >= last_bar.open:
            return False, "最后一根不是顺势确认阴线", None
        if last_bar.close >= prev_bar.low - confirm_buffer:
            return False, "最后一根尚未收下前一根低点确认位", None
        bar_body = last_bar.open - last_bar.close
        if bar_body < atr * self.config.min_confirm_bar_body_atr_ratio:
            return False, f"确认阴线实体过小 body={bar_body:.5f}，视为无力确认", None

        entry_extension = pullback_high_bar.high - last_bar.close
        if entry_extension > atr * self.config.max_entry_distance_atr_ratio:
            return False, "当前价格离反弹高点过远，放弃追空", None

        stop_anchor = getattr(state, "last_swing_high", None) or pullback_high_bar.high
        return True, (
            f"单周期空头延续触发："
            f"反弹高点={pullback_high_bar.high:.5f}，"
            f"反弹深度比={retrace_ratio:.2f}，"
            f"确认低点={prev_bar.low:.5f}，"
            f"当前收盘={last_bar.close:.5f}"
        ), stop_anchor

    def _state_allows_entry(self, state: StructureState, direction: str) -> tuple[bool, str]:
        if state.primary_bias != direction:
            return False, f"主导周期主方向不是 {direction}"
        if state.phase == "reversal_candidate":
            return False, "主导周期已进入反转候选阶段"
        if self.config.allowed_phases and state.phase not in self.config.allowed_phases:
            return False, f"主导周期阶段={state.phase!r}，不在允许入场阶段 {self.config.allowed_phases}"
        if state.trend_score < self.config.min_trend_score_trading_tf:
            return False, (
                f"主导周期趋势强度不足 trend_score={state.trend_score:.4f}"
            )
        if self.config.require_reversal_warning_clear and getattr(state, "reversal_warning", False):
            return False, "主导周期存在反转警告"
        return True, "ok"

    def _higher_tf_aligns(
        self,
        high_tf: StructureState,
        mid_tf: StructureState,
        direction: str,
    ) -> tuple[bool, str]:
        if not self.config.require_higher_tf_bias_align or not self.config.higher_tf_for_align:
            return True, "ok"
        htf = {"high": high_tf, "mid": mid_tf}.get(self.config.higher_tf_for_align)
        if htf is None:
            return True, "ok"
        if htf.primary_bias != direction:
            return False, (
                f"高周期({self.config.higher_tf_for_align})方向={htf.primary_bias!r}，"
                f"与入场方向{direction!r}不一致"
            )
        return True, "ok"

    def generate_signal(
        self,
        high_tf: StructureState,
        mid_tf: StructureState,
        low_tf: StructureState,
        ctx: StrategyContext,
    ) -> Optional[CandidateSignal]:
        dominant = self._select_dominant_state(high_tf, mid_tf, low_tf)

        if dominant.primary_bias == "long":
            ok, reason = self._higher_tf_aligns(high_tf, mid_tf, "long")
            if not ok:
                self.logger.info("未生成候选信号：%s", reason)
                return None
            return self._build_long_signal(dominant, ctx)
        if dominant.primary_bias == "short":
            ok, reason = self._higher_tf_aligns(high_tf, mid_tf, "short")
            if not ok:
                self.logger.info("未生成候选信号：%s", reason)
                return None
            return self._build_short_signal(dominant, ctx)

        self.logger.info("未生成候选信号：主导周期主方向为 neutral")
        return None

    def _build_long_signal(
        self,
        state: StructureState,
        ctx: StrategyContext,
    ) -> Optional[CandidateSignal]:
        if not self.config.allow_long:
            self.logger.info("未生成候选信号：当前策略已关闭 long")
            return None

        ok, reason = self._state_allows_entry(state, "long")
        if not ok:
            self.logger.info("未生成候选信号：%s", reason)
            return None

        triggered, trigger_reason, stop_anchor = self._check_long_continuation_trigger(
            state=state,
            ctx=ctx,
        )
        if not triggered:
            self.logger.info("未生成候选信号：%s", trigger_reason)
            return None
        if stop_anchor is None:
            self.logger.info("未生成候选信号：缺少有效多头止损锚点")
            return None

        stop_buffer = max(
            self._effective_atr(ctx) * self.config.stop_buffer_atr_ratio,
            ctx.spread * self.config.stop_buffer_spread_multiple,
        )
        entry_price = ctx.entry_price
        stop_loss = stop_anchor - stop_buffer
        risk = entry_price - stop_loss
        if risk <= 0:
            self.logger.info(
                "未生成候选信号：多头止损锚点异常，entry=%.5f，sl=%.5f",
                entry_price,
                stop_loss,
            )
            return None

        take_profit: Optional[float] = None
        regime = "trend"
        if state.phase == "range":
            if not state.near_lower:
                self.logger.info("未生成候选信号：range phase 多头但价格不在区间低部 (near_lower=False)")
                return None
            if state.constraint_upper is not None:
                tp_buffer = self._effective_atr(ctx) * self.config.range_tp_buffer_atr_ratio
                tp_candidate = state.constraint_upper - tp_buffer
                if tp_candidate > entry_price:
                    take_profit = tp_candidate
            regime = "range"

        rr_estimate = (take_profit - entry_price) / risk if take_profit else 0.0

        signal = CandidateSignal(
            ts=ctx.ts,
            symbol=ctx.symbol,
            direction="long",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rr_estimate=rr_estimate,
            reason=(
                f"单周期{'区间' if regime == 'range' else '趋势延续'}多头候选信号："
                f"主导周期={state.timeframe}/{state.primary_bias}/{state.phase}；"
                f"{trigger_reason}"
            ),
            tags=[
                "single_tf_trend_continuation",
                "long",
                "pullback_resume",
                regime,
            ],
            meta={
                "dominant_tf": state.timeframe,
                "dominant_phase": state.phase,
                "regime": regime,
                "policy": "single_tf_trend_continuation_v2",
            },
        )
        self.logger.info(
            "已生成单周期多头候选信号：entry=%.5f，stop_loss=%.5f",
            signal.entry_price,
            signal.stop_loss,
        )
        return signal

    def _build_short_signal(
        self,
        state: StructureState,
        ctx: StrategyContext,
    ) -> Optional[CandidateSignal]:
        if not self.config.allow_short:
            self.logger.info("未生成候选信号：当前策略已关闭 short")
            return None

        ok, reason = self._state_allows_entry(state, "short")
        if not ok:
            self.logger.info("未生成候选信号：%s", reason)
            return None

        triggered, trigger_reason, stop_anchor = self._check_short_continuation_trigger(
            state=state,
            ctx=ctx,
        )
        if not triggered:
            self.logger.info("未生成候选信号：%s", trigger_reason)
            return None
        if stop_anchor is None:
            self.logger.info("未生成候选信号：缺少有效空头止损锚点")
            return None

        stop_buffer = max(
            self._effective_atr(ctx) * self.config.stop_buffer_atr_ratio,
            ctx.spread * self.config.stop_buffer_spread_multiple,
        )
        entry_price = ctx.entry_price
        stop_loss = stop_anchor + stop_buffer
        risk = stop_loss - entry_price
        if risk <= 0:
            self.logger.info(
                "未生成候选信号：空头止损锚点异常，entry=%.5f，sl=%.5f",
                entry_price,
                stop_loss,
            )
            return None

        take_profit: Optional[float] = None
        regime = "trend"
        if state.phase == "range":
            if not state.near_upper:
                self.logger.info("未生成候选信号：range phase 空头但价格不在区间高部 (near_upper=False)")
                return None
            if state.constraint_lower is not None:
                tp_buffer = self._effective_atr(ctx) * self.config.range_tp_buffer_atr_ratio
                tp_candidate = state.constraint_lower + tp_buffer
                if tp_candidate < entry_price:
                    take_profit = tp_candidate
            regime = "range"

        rr_estimate = (entry_price - take_profit) / risk if take_profit else 0.0

        signal = CandidateSignal(
            ts=ctx.ts,
            symbol=ctx.symbol,
            direction="short",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rr_estimate=rr_estimate,
            reason=(
                f"单周期{'区间' if regime == 'range' else '趋势延续'}空头候选信号："
                f"主导周期={state.timeframe}/{state.primary_bias}/{state.phase}；"
                f"{trigger_reason}"
            ),
            tags=[
                "single_tf_trend_continuation",
                "short",
                "pullback_resume",
                regime,
            ],
            meta={
                "dominant_tf": state.timeframe,
                "dominant_phase": state.phase,
                "regime": regime,
                "policy": "single_tf_trend_continuation_v2",
            },
        )
        self.logger.info(
            "已生成单周期空头候选信号：entry=%.5f，stop_loss=%.5f",
            signal.entry_price,
            signal.stop_loss,
        )
        return signal


# ---------------------------------------------------------------------------
# SwingPointPolicy —— 高周期摆点入场策略
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SwingPointPolicyConfig:
    """H4 摆点入场策略配置。

    设计原则：
    - 不预测市场方向，只识别价格是否到达有意义的位置（H4 摆点）
    - 在摆点附近等待 M15 方向性确认后入场
    - 止盈设在对面摆点，R:R 自然由摆点间距决定
    - 在趋势和震荡市场中均成立
    """

    allow_long: bool = True
    allow_short: bool = True

    # 沿用，供执行模拟器识别主导执行周期
    dominant_timeframe: str = "low"

    # 用哪个周期的摆点作为入场区域："mid"=H4，"high"=D1
    swing_tf: str = "mid"

    # 入场区域宽度：价格在摆点 ± (entry_zone_atr_ratio × higher_tf_atr) 以内
    entry_zone_atr_ratio: float = 0.50

    # higher_tf_atr 为 0 时的备用估算：M15 ATR × 此系数
    m15_to_higher_atr_factor: float = 4.0

    # 止损 buffer（摆点外侧）
    stop_buffer_atr_ratio: float = 0.15
    stop_buffer_spread_multiple: float = 1.5

    # 只有 R:R 估算 >= min_rr 才入场；若 TP 无法确定则不过滤
    min_rr: float = 1.5

    # M15 确认 bar 最小实体（× M15 ATR）
    min_confirm_bar_body_atr_ratio: float = 0.15


class SwingPointPolicy:
    """H4 摆点入场策略。

    逻辑：
    1. 从 mid_tf（H4）状态取摆点：last_swing_low（多）/ last_swing_high（空）
    2. 检查当前 M15 价格是否进入摆点区域（± entry_zone_atr_ratio × H4_ATR）
    3. 要求摆点未被突破（structure_low/high_broken = False）
    4. 等待 M15 最后一根 bar 给出方向性确认（实体阳/阴线）
    5. 止损：摆点外侧 + buffer；止盈：对面摆点
    6. 只有 R:R >= min_rr 才生成信号
    """

    def __init__(
        self,
        config: Optional[SwingPointPolicyConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or SwingPointPolicyConfig()
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.strategy.swing_point")
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

    def _swing_state(
        self, high_tf: StructureState, mid_tf: StructureState
    ) -> StructureState:
        return {"high": high_tf, "mid": mid_tf}.get(self.config.swing_tf, mid_tf)

    def _higher_atr(self, ctx: StrategyContext) -> float:
        if ctx.higher_tf_atr > 0:
            return ctx.higher_tf_atr
        return max(ctx.atr * self.config.m15_to_higher_atr_factor, ctx.spread, 1e-8)

    def _in_zone(self, price: float, level: float, zone: float) -> bool:
        return abs(price - level) <= zone

    def _confirm_long(
        self,
        swing: StructureState,
        ctx: StrategyContext,
        zone: float,
    ) -> tuple[bool, str]:
        bars = self._recent_low_bars(ctx)
        if len(bars) < 2:
            return False, "M15 确认 bars 不足"
        last = bars[-1]
        prev = bars[-2]
        atr = max(ctx.atr, ctx.spread, 1e-8)
        level = float(swing.last_swing_low)
        reclaim_buffer = max(atr * self.config.reclaim_buffer_atr_ratio, ctx.spread)
        confirm_buffer = atr * self.config.confirm_break_buffer_atr_ratio
        touch_band = max(atr * self.config.sweep_touch_atr_ratio, ctx.spread)

        recent = bars[-max(int(self.config.recent_touch_bars), 1):]
        recent_min_low = min(b.low for b in recent)
        if recent_min_low > level + touch_band:
            return False, "最近几根 M15 未真正扫到摆低附近"
        if last.close <= last.open:
            return False, "最后一根不是确认阳线"
        body = last.close - last.open
        if body < atr * self.config.min_confirm_bar_body_atr_ratio:
            return False, f"确认阳线实体过小 body={body:.5f}"
        if last.close <= level + reclaim_buffer:
            return False, "最后一根未有效收回摆低之上"
        if last.close <= (last.high + last.low) / 2:
            return False, "确认阳线收盘位置偏弱"

        swept = recent_min_low <= level + touch_band
        reclaimed = last.close > level + reclaim_buffer
        broke_prev = last.close > prev.high + confirm_buffer
        strong_reject = (
            last.low <= level + touch_band
            and last.close > max(prev.close, level + reclaim_buffer)
        ) or (
            recent_min_low < level and last.close > prev.close + confirm_buffer
        )
        if not swept:
            return False, "最近几根没有充分回踩到摆低"
        if not reclaimed:
            return False, "最近确认 bar 未完成有效收回"
        if not (broke_prev or strong_reject):
            return False, "缺少足够明确的扫低回收确认"
        return True, (
            f"M15 多头确认：摆低={level:.5f}，prev_high={prev.high:.5f}，"
            f"last_close={last.close:.5f}"
        )

    def _confirm_short(
        self,
        swing: StructureState,
        ctx: StrategyContext,
        zone: float,
    ) -> tuple[bool, str]:
        bars = self._recent_low_bars(ctx)
        if len(bars) < 2:
            return False, "M15 确认 bars 不足"
        last = bars[-1]
        prev = bars[-2]
        atr = max(ctx.atr, ctx.spread, 1e-8)
        level = float(swing.last_swing_high)
        reclaim_buffer = max(atr * self.config.reclaim_buffer_atr_ratio, ctx.spread)
        confirm_buffer = atr * self.config.confirm_break_buffer_atr_ratio
        touch_band = max(atr * self.config.sweep_touch_atr_ratio, ctx.spread)

        recent = bars[-max(int(self.config.recent_touch_bars), 1):]
        recent_max_high = max(b.high for b in recent)
        if recent_max_high < level - touch_band:
            return False, "最近几根 M15 未真正扫到摆高附近"
        if last.close >= last.open:
            return False, "最后一根不是确认阴线"
        body = last.open - last.close
        if body < atr * self.config.min_confirm_bar_body_atr_ratio:
            return False, f"确认阴线实体过小 body={body:.5f}"
        if last.close >= level - reclaim_buffer:
            return False, "最后一根未有效收回摆高之下"
        if last.close >= (last.high + last.low) / 2:
            return False, "确认阴线收盘位置偏弱"

        swept = recent_max_high >= level - touch_band
        reclaimed = last.close < level - reclaim_buffer
        broke_prev = last.close < prev.low - confirm_buffer
        strong_reject = (
            last.high >= level - touch_band
            and last.close < min(prev.close, level - reclaim_buffer)
        ) or (
            recent_max_high > level and last.close < prev.close - confirm_buffer
        )
        if not swept:
            return False, "最近几根没有充分反弹到摆高"
        if not reclaimed:
            return False, "最近确认 bar 未完成有效收回"
        if not (broke_prev or strong_reject):
            return False, "缺少足够明确的扫高回落确认"
        return True, (
            f"M15 空头确认：摆高={level:.5f}，prev_low={prev.low:.5f}，"
            f"last_close={last.close:.5f}"
        )

    def generate_signal(
        self,
        high_tf: StructureState,
        mid_tf: StructureState,
        low_tf: StructureState,
        ctx: StrategyContext,
    ) -> Optional[CandidateSignal]:
        swing = self._swing_state(high_tf, mid_tf)
        h_atr = self._higher_atr(ctx)
        zone = h_atr * self.config.entry_zone_atr_ratio

        # --- 多头：价格回测 H4 摆低 ---
        if (
            self.config.allow_long
            and swing.last_swing_low is not None
            and not swing.structure_low_broken
            and self._in_zone(ctx.entry_price, swing.last_swing_low, zone)
        ):
            ok, reason = self._confirm_long(ctx)
            if ok:
                sig = self._build_long_signal(swing, ctx, h_atr)
                if sig is not None:
                    return sig
            else:
                self.logger.info("摆点多头：M15 确认未通过 — %s", reason)

        # --- 空头：价格回测 H4 摆高 ---
        if (
            self.config.allow_short
            and swing.last_swing_high is not None
            and not swing.structure_high_broken
            and self._in_zone(ctx.entry_price, swing.last_swing_high, zone)
        ):
            ok, reason = self._confirm_short(ctx)
            if ok:
                sig = self._build_short_signal(swing, ctx, h_atr)
                if sig is not None:
                    return sig
            else:
                self.logger.info("摆点空头：M15 确认未通过 — %s", reason)

        return None

    def _build_long_signal(
        self,
        swing: StructureState,
        ctx: StrategyContext,
        h_atr: float,
    ) -> Optional[CandidateSignal]:
        entry = ctx.entry_price
        stop_buf = max(
            h_atr * self.config.stop_buffer_atr_ratio,
            ctx.spread * self.config.stop_buffer_spread_multiple,
        )
        stop = swing.last_swing_low - stop_buf  # type: ignore[operator]
        risk = entry - stop
        if risk <= 0:
            self.logger.info("摆点多头：止损计算异常 entry=%.5f sl=%.5f", entry, stop)
            return None

        tp: Optional[float] = swing.last_swing_high or swing.constraint_upper
        if tp is not None and tp <= entry:
            tp = None

        rr = (tp - entry) / risk if tp is not None else 0.0
        if tp is not None and rr < self.config.min_rr:
            self.logger.info(
                "摆点多头：R:R=%.2f < min=%.2f，跳过", rr, self.config.min_rr
            )
            return None

        self.logger.info(
            "已生成摆点多头信号：swing_low=%.5f，entry=%.5f，stop=%.5f，tp=%s，R:R=%.2f",
            swing.last_swing_low,
            entry,
            stop,
            f"{tp:.5f}" if tp else "None",
            rr,
        )
        return CandidateSignal(
            ts=ctx.ts,
            symbol=ctx.symbol,
            direction="long",
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            rr_estimate=rr,
            reason=(
                f"H4摆点多头入场："
                f"摆低={swing.last_swing_low:.5f}，"
                f"entry={entry:.5f}，stop={stop:.5f}，"
                f"tp={f'{tp:.5f}' if tp else 'None'}，R:R={rr:.2f}"
            ),
            tags=["swing_point", "long"],
            meta={
                "swing_tf": swing.timeframe,
                "swing_low": swing.last_swing_low,
                "regime": "swing_point",
                "policy": "swing_point_v1",
            },
        )

    def _build_short_signal(
        self,
        swing: StructureState,
        ctx: StrategyContext,
        h_atr: float,
    ) -> Optional[CandidateSignal]:
        entry = ctx.entry_price
        stop_buf = max(
            h_atr * self.config.stop_buffer_atr_ratio,
            ctx.spread * self.config.stop_buffer_spread_multiple,
        )
        stop = swing.last_swing_high + stop_buf  # type: ignore[operator]
        risk = stop - entry
        if risk <= 0:
            self.logger.info("摆点空头：止损计算异常 entry=%.5f sl=%.5f", entry, stop)
            return None

        tp: Optional[float] = swing.last_swing_low or swing.constraint_lower
        if tp is not None and tp >= entry:
            tp = None

        rr = (entry - tp) / risk if tp is not None else 0.0
        if tp is not None and rr < self.config.min_rr:
            self.logger.info(
                "摆点空头：R:R=%.2f < min=%.2f，跳过", rr, self.config.min_rr
            )
            return None

        self.logger.info(
            "已生成摆点空头信号：swing_high=%.5f，entry=%.5f，stop=%.5f，tp=%s，R:R=%.2f",
            swing.last_swing_high,
            entry,
            stop,
            f"{tp:.5f}" if tp else "None",
            rr,
        )
        return CandidateSignal(
            ts=ctx.ts,
            symbol=ctx.symbol,
            direction="short",
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            rr_estimate=rr,
            reason=(
                f"H4摆点空头入场："
                f"摆高={swing.last_swing_high:.5f}，"
                f"entry={entry:.5f}，stop={stop:.5f}，"
                f"tp={f'{tp:.5f}' if tp else 'None'}，R:R={rr:.2f}"
            ),
            tags=["swing_point", "short"],
            meta={
                "swing_tf": swing.timeframe,
                "swing_high": swing.last_swing_high,
                "regime": "swing_point",
                "policy": "swing_point_v1",
            },
        )


# ===========================================================================
# SwingPointPolicy v2 —— 位置优先、单侧入场带、固定目标
# ===========================================================================

@dataclass(slots=True)
class SwingPointPolicyConfig:
    """高周期摆点入场策略（v2）。

    设计原则：
    - 位置优先：只在高周期摆点附近找交易，不再依赖三周期同时对齐
    - 单侧入场带：多头只在 swing low 上方回踩带内接多；空头只在 swing high 下方反弹带内接空
    - 确认看“拒绝 + 收回”，不是只看 K 线颜色
    - 高周期只做 veto，不做强绑定
    - 出场以摆点到摆点为主，执行层再补时间/PnL 退出
    """

    allow_long: bool = True
    allow_short: bool = True

    dominant_timeframe: str = "low"
    swing_tf: str = "mid"

    entry_zone_atr_ratio: float = 0.45
    m15_to_higher_atr_factor: float = 4.0

    stop_buffer_atr_ratio: float = 0.15
    stop_buffer_spread_multiple: float = 1.5

    min_rr: float = 1.80
    max_rr: float = 4.00

    min_confirm_bar_body_atr_ratio: float = 0.15
    reclaim_buffer_atr_ratio: float = 0.02
    confirm_break_buffer_atr_ratio: float = 0.02
    sweep_touch_atr_ratio: float = 0.08
    recent_touch_bars: int = 3

    max_attempts_per_swing: int = 2
    max_cumulative_loss_r_per_swing: float = 1.50
    require_reset_after_loss: bool = True

    use_higher_tf_veto: bool = True
    veto_timeframe: str = "high"
    veto_min_trend_score: float = 0.45
    veto_phases: tuple[str, ...] = field(default_factory=lambda: ("impulse", "pullback"))


class SwingPointPolicy:
    """高周期摆点入场策略（v2）。"""

    def __init__(
        self,
        config: Optional[SwingPointPolicyConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or SwingPointPolicyConfig()
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.strategy.swing_point")
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

    def _swing_state(self, high_tf: StructureState, mid_tf: StructureState) -> StructureState:
        return {"high": high_tf, "mid": mid_tf}.get(self.config.swing_tf, mid_tf)

    def _higher_atr(self, ctx: StrategyContext) -> float:
        if ctx.higher_tf_atr > 0:
            return ctx.higher_tf_atr
        return max(ctx.atr * self.config.m15_to_higher_atr_factor, ctx.spread, 1e-8)

    def _recent_low_bars(self, ctx: StrategyContext) -> list[GenericBar]:
        return ctx.low_tf_bars or ctx.dominant_tf_bars or []

    def _long_zone_contains(self, price: float, level: float, zone: float) -> bool:
        return level <= price <= level + zone

    def _short_zone_contains(self, price: float, level: float, zone: float) -> bool:
        return level - zone <= price <= level

    def _passes_higher_tf_veto(
        self,
        direction: str,
        high_tf: StructureState,
        mid_tf: StructureState,
    ) -> tuple[bool, str]:
        if not self.config.use_higher_tf_veto:
            return True, "ok"
        veto_state = {"high": high_tf, "mid": mid_tf}.get(self.config.veto_timeframe)
        if veto_state is None:
            return True, "ok"
        opposite = "short" if direction == "long" else "long"
        if (
            veto_state.primary_bias == opposite
            and veto_state.phase in self.config.veto_phases
            and veto_state.trend_score >= self.config.veto_min_trend_score
        ):
            return False, (
                f"高周期 veto：{self.config.veto_timeframe}={veto_state.primary_bias}/{veto_state.phase} "
                f"trend_score={veto_state.trend_score:.2f}"
            )
        return True, "ok"

    def _confirm_long(
        self,
        swing: StructureState,
        ctx: StrategyContext,
        zone: float,
    ) -> tuple[bool, str]:
        bars = self._recent_low_bars(ctx)
        if len(bars) < 2:
            return False, "M15 确认 bars 不足"
        last = bars[-1]
        prev = bars[-2]
        atr = max(ctx.atr, ctx.spread, 1e-8)
        level = float(swing.last_swing_low)
        reclaim_buffer = max(atr * self.config.reclaim_buffer_atr_ratio, ctx.spread)
        confirm_buffer = atr * self.config.confirm_break_buffer_atr_ratio
        touch_band = max(atr * self.config.sweep_touch_atr_ratio, ctx.spread)

        recent = bars[-max(int(self.config.recent_touch_bars), 1):]
        recent_min_low = min(b.low for b in recent)
        if recent_min_low > level + touch_band:
            return False, "最近几根 M15 未真正扫到摆低附近"
        if last.close <= last.open:
            return False, "最后一根不是确认阳线"
        body = last.close - last.open
        if body < atr * self.config.min_confirm_bar_body_atr_ratio:
            return False, f"确认阳线实体过小 body={body:.5f}"
        if last.close <= level + reclaim_buffer:
            return False, "最后一根未有效收回摆低之上"
        if last.close <= (last.high + last.low) / 2:
            return False, "确认阳线收盘位置偏弱"

        swept = recent_min_low <= level + touch_band
        reclaimed = last.close > level + reclaim_buffer
        broke_prev = last.close > prev.high + confirm_buffer
        strong_reject = (
            last.low <= level + touch_band
            and last.close > max(prev.close, level + reclaim_buffer)
        ) or (
            recent_min_low < level and last.close > prev.close + confirm_buffer
        )
        if not swept:
            return False, "最近几根没有充分回踩到摆低"
        if not reclaimed:
            return False, "最近确认 bar 未完成有效收回"
        if not (broke_prev or strong_reject):
            return False, "缺少足够明确的扫低回收确认"
        return True, (
            f"M15 多头确认：摆低={level:.5f}，prev_high={prev.high:.5f}，"
            f"last_close={last.close:.5f}"
        )

    def _confirm_short(
        self,
        swing: StructureState,
        ctx: StrategyContext,
        zone: float,
    ) -> tuple[bool, str]:
        bars = self._recent_low_bars(ctx)
        if len(bars) < 2:
            return False, "M15 确认 bars 不足"
        last = bars[-1]
        prev = bars[-2]
        atr = max(ctx.atr, ctx.spread, 1e-8)
        level = float(swing.last_swing_high)
        reclaim_buffer = max(atr * self.config.reclaim_buffer_atr_ratio, ctx.spread)
        confirm_buffer = atr * self.config.confirm_break_buffer_atr_ratio
        touch_band = max(atr * self.config.sweep_touch_atr_ratio, ctx.spread)

        recent = bars[-max(int(self.config.recent_touch_bars), 1):]
        recent_max_high = max(b.high for b in recent)
        if recent_max_high < level - touch_band:
            return False, "最近几根 M15 未真正扫到摆高附近"
        if last.close >= last.open:
            return False, "最后一根不是确认阴线"
        body = last.open - last.close
        if body < atr * self.config.min_confirm_bar_body_atr_ratio:
            return False, f"确认阴线实体过小 body={body:.5f}"
        if last.close >= level - reclaim_buffer:
            return False, "最后一根未有效收回摆高之下"
        if last.close >= (last.high + last.low) / 2:
            return False, "确认阴线收盘位置偏弱"

        swept = recent_max_high >= level - touch_band
        reclaimed = last.close < level - reclaim_buffer
        broke_prev = last.close < prev.low - confirm_buffer
        strong_reject = (
            last.high >= level - touch_band
            and last.close < min(prev.close, level - reclaim_buffer)
        ) or (
            recent_max_high > level and last.close < prev.close - confirm_buffer
        )
        if not swept:
            return False, "最近几根没有充分反弹到摆高"
        if not reclaimed:
            return False, "最近确认 bar 未完成有效收回"
        if not (broke_prev or strong_reject):
            return False, "缺少足够明确的扫高回落确认"
        return True, (
            f"M15 空头确认：摆高={level:.5f}，prev_low={prev.low:.5f}，"
            f"last_close={last.close:.5f}"
        )

    def generate_signal(
        self,
        high_tf: StructureState,
        mid_tf: StructureState,
        low_tf: StructureState,
        ctx: StrategyContext,
    ) -> Optional[CandidateSignal]:
        swing = self._swing_state(high_tf, mid_tf)
        h_atr = self._higher_atr(ctx)
        zone = h_atr * self.config.entry_zone_atr_ratio

        if (
            self.config.allow_long
            and swing.last_swing_low is not None
            and not swing.structure_low_broken
            and self._long_zone_contains(ctx.entry_price, float(swing.last_swing_low), zone)
        ):
            ok, reason = self._passes_higher_tf_veto("long", high_tf, mid_tf)
            if not ok:
                self.logger.info("摆点多头：%s", reason)
            else:
                ok, reason = self._confirm_long(swing, ctx, zone)
                if ok:
                    sig = self._build_long_signal(swing, ctx, h_atr, zone, reason)
                    if sig is not None:
                        return sig
                else:
                    self.logger.info("摆点多头：M15 确认未通过 — %s", reason)

        if (
            self.config.allow_short
            and swing.last_swing_high is not None
            and not swing.structure_high_broken
            and self._short_zone_contains(ctx.entry_price, float(swing.last_swing_high), zone)
        ):
            ok, reason = self._passes_higher_tf_veto("short", high_tf, mid_tf)
            if not ok:
                self.logger.info("摆点空头：%s", reason)
            else:
                ok, reason = self._confirm_short(swing, ctx, zone)
                if ok:
                    sig = self._build_short_signal(swing, ctx, h_atr, zone, reason)
                    if sig is not None:
                        return sig
                else:
                    self.logger.info("摆点空头：M15 确认未通过 — %s", reason)
        return None

    def _build_long_signal(
        self,
        swing: StructureState,
        ctx: StrategyContext,
        h_atr: float,
        zone: float,
        confirm_reason: str,
    ) -> Optional[CandidateSignal]:
        entry = ctx.entry_price
        stop_buf = max(h_atr * self.config.stop_buffer_atr_ratio, ctx.spread * self.config.stop_buffer_spread_multiple)
        swing_low = float(swing.last_swing_low)
        stop = swing_low - stop_buf
        risk = entry - stop
        if risk <= 0:
            self.logger.info("摆点多头：止损计算异常 entry=%.5f sl=%.5f", entry, stop)
            return None
        tp: Optional[float] = swing.last_swing_high or swing.constraint_upper
        if tp is None or tp <= entry:
            self.logger.info("摆点多头：缺少有效对侧目标，跳过")
            return None
        rr = (tp - entry) / max(risk, 1e-8)
        if rr < self.config.min_rr:
            self.logger.info("摆点多头：R:R=%.2f < min=%.2f，跳过", rr, self.config.min_rr)
            return None
        if self.config.max_rr > 0 and rr > self.config.max_rr:
            tp = entry + risk * self.config.max_rr
            rr = self.config.max_rr
        return CandidateSignal(
            ts=ctx.ts,
            symbol=ctx.symbol,
            direction="long",
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            rr_estimate=rr,
            reason=(
                f"摆点多头入场：{swing.timeframe}摆低={swing_low:.5f}，"
                f"入场带=[{swing_low:.5f},{swing_low + zone:.5f}]，{confirm_reason}，"
                f"entry={entry:.5f}，stop={stop:.5f}，tp={tp:.5f}，R:R={rr:.2f}"
            ),
            tags=["swing_point", "long", "liquidity_reaction"],
            meta={
                "policy": "swing_point_v2",
                "swing_tf": swing.timeframe,
                "dominant_tf": self.config.dominant_timeframe,
                "swing_low": swing_low,
                "swing_high": tp,
                "zone_lower": swing_low,
                "zone_upper": swing_low + zone,
                "swing_id": f"{swing.timeframe}:long:{swing_low:.5f}",
                "max_attempts_per_swing": self.config.max_attempts_per_swing,
                "max_cumulative_loss_r_per_swing": self.config.max_cumulative_loss_r_per_swing,
                "require_reset_after_loss": self.config.require_reset_after_loss,
            },
        )

    def _build_short_signal(
        self,
        swing: StructureState,
        ctx: StrategyContext,
        h_atr: float,
        zone: float,
        confirm_reason: str,
    ) -> Optional[CandidateSignal]:
        entry = ctx.entry_price
        stop_buf = max(h_atr * self.config.stop_buffer_atr_ratio, ctx.spread * self.config.stop_buffer_spread_multiple)
        swing_high = float(swing.last_swing_high)
        stop = swing_high + stop_buf
        risk = stop - entry
        if risk <= 0:
            self.logger.info("摆点空头：止损计算异常 entry=%.5f sl=%.5f", entry, stop)
            return None
        tp: Optional[float] = swing.last_swing_low or swing.constraint_lower
        if tp is None or tp >= entry:
            self.logger.info("摆点空头：缺少有效对侧目标，跳过")
            return None
        rr = (entry - tp) / max(risk, 1e-8)
        if rr < self.config.min_rr:
            self.logger.info("摆点空头：R:R=%.2f < min=%.2f，跳过", rr, self.config.min_rr)
            return None
        if self.config.max_rr > 0 and rr > self.config.max_rr:
            tp = entry - risk * self.config.max_rr
            rr = self.config.max_rr
        return CandidateSignal(
            ts=ctx.ts,
            symbol=ctx.symbol,
            direction="short",
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            rr_estimate=rr,
            reason=(
                f"摆点空头入场：{swing.timeframe}摆高={swing_high:.5f}，"
                f"入场带=[{swing_high - zone:.5f},{swing_high:.5f}]，{confirm_reason}，"
                f"entry={entry:.5f}，stop={stop:.5f}，tp={tp:.5f}，R:R={rr:.2f}"
            ),
            tags=["swing_point", "short", "liquidity_reaction"],
            meta={
                "policy": "swing_point_v2",
                "swing_tf": swing.timeframe,
                "dominant_tf": self.config.dominant_timeframe,
                "swing_high": swing_high,
                "swing_low": tp,
                "zone_lower": swing_high - zone,
                "zone_upper": swing_high,
                "swing_id": f"{swing.timeframe}:short:{swing_high:.5f}",
                "max_attempts_per_swing": self.config.max_attempts_per_swing,
                "max_cumulative_loss_r_per_swing": self.config.max_cumulative_loss_r_per_swing,
                "require_reset_after_loss": self.config.require_reset_after_loss,
            },
        )
