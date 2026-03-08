from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Optional

from dpaitrade.structure.state import Phase, PrimaryBias, StructureState


@dataclass(slots=True)
class GenericBar:
    """
    通用 K 线对象。

    只要能提供 ts/open/high/low/close，
    就可以送入统一结构分析器。
    """

    ts: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(slots=True)
class StructureAnalyzerConfig:
    """
    统一结构分析器配置。

    不同周期只改参数，不改逻辑。
    """

    timeframe: str

    min_bars: int = 24
    lookback: int = 24
    swing_window: int = 2

    trend_threshold: float = 0.38
    range_threshold: float = 0.58

    break_tolerance_atr_ratio: float = 0.15
    boundary_tolerance_atr_ratio: float = 0.20
    near_boundary_atr_ratio: float = 0.35

    min_swing_points: int = 2


@dataclass(slots=True)
class SwingPoint:
    index: int
    price: float
    kind: str  # "high" | "low"


class StructureAnalyzer:
    """
    统一结构分析器。

    输出统一的 StructureState。
    """

    def __init__(
        self,
        config: StructureAnalyzerConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or self._build_logger(config.timeframe)

    @staticmethod
    def _build_logger(timeframe: str) -> logging.Logger:
        logger = logging.getLogger(f"dpaitrade.structure.{timeframe.lower()}")
        if logger.handlers:
            return logger

        logger.setLevel(logging.WARNING)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    def analyze(self, bars: list[GenericBar]) -> StructureState:
        """
        分析最近一段 K 线，输出统一结构状态。
        """
        required = max(self.config.min_bars, self.config.lookback)
        if len(bars) < required:
            reason = (
                f"{self.config.timeframe} 数据不足："
                f"当前 {len(bars)} 根，小于所需 {required} 根"
            )
            self.logger.debug(reason)
            return self._unknown_state(ts=bars[-1].ts if bars else datetime.min, reason=reason)

        window = bars[-self.config.lookback :]
        closes = [b.close for b in window]

        atr = self._calc_atr_proxy(window)
        directional_efficiency = self._calc_directional_efficiency(closes)

        swing_highs, swing_lows = self._detect_swings(window)

        last_swing_high = swing_highs[-1].price if swing_highs else None
        last_swing_low = swing_lows[-1].price if swing_lows else None
        prev_swing_high = swing_highs[-2].price if len(swing_highs) >= 2 else None
        prev_swing_low = swing_lows[-2].price if len(swing_lows) >= 2 else None

        structure_high_broken = self._effective_break_high(window, last_swing_high, atr)
        structure_low_broken = self._effective_break_low(window, last_swing_low, atr)

        hh_hl_ready = self._hh_hl_ready(
            last_swing_high, prev_swing_high, last_swing_low, prev_swing_low
        )
        lh_ll_ready = self._lh_ll_ready(
            last_swing_high, prev_swing_high, last_swing_low, prev_swing_low
        )

        trend_score = self._calc_trend_score(directional_efficiency, hh_hl_ready, lh_ll_ready)
        range_score = self._calc_range_score(directional_efficiency, swing_highs, swing_lows)
        is_range_like = range_score >= self.config.range_threshold

        constraint_upper, constraint_lower, boundary_tolerance = self._estimate_constraints(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            atr=atr,
        )

        last_close = closes[-1]
        near_upper = self._is_near_upper(last_close, constraint_upper, atr, boundary_tolerance)
        near_lower = self._is_near_lower(last_close, constraint_lower, atr, boundary_tolerance)

        primary_bias = self._infer_primary_bias(
            net_move=closes[-1] - closes[0],
            hh_hl_ready=hh_hl_ready,
            lh_ll_ready=lh_ll_ready,
        )

        phase = self._infer_phase(
            primary_bias=primary_bias,
            is_range_like=is_range_like,
            structure_high_broken=structure_high_broken,
            structure_low_broken=structure_low_broken,
            hh_hl_ready=hh_hl_ready,
            lh_ll_ready=lh_ll_ready,
            window=window,
        )

        in_value_zone = (
            (primary_bias == "short" and (near_upper or phase in ("pullback", "range")))
            or (primary_bias == "long" and (near_lower or phase in ("pullback", "range")))
        )

        continuation_ready = (
            primary_bias in ("long", "short")
            and phase in ("impulse", "pullback", "range")
            and not (
                (primary_bias == "short" and structure_high_broken)
                or (primary_bias == "long" and structure_low_broken)
            )
        )

        reversal_warning = (
            phase == "reversal_candidate"
            or (primary_bias == "short" and structure_high_broken)
            or (primary_bias == "long" and structure_low_broken)
        )

        confidence = round(max(trend_score, range_score), 4)

        reason = self._build_reason(
            primary_bias=primary_bias,
            phase=phase,
            trend_score=trend_score,
            range_score=range_score,
            directional_efficiency=directional_efficiency,
            near_upper=near_upper,
            near_lower=near_lower,
            structure_high_broken=structure_high_broken,
            structure_low_broken=structure_low_broken,
        )
        self.logger.debug(reason)

        return StructureState(
            timeframe=self.config.timeframe,
            ts=window[-1].ts,
            primary_bias=primary_bias,
            phase=phase,
            trend_score=trend_score,
            range_score=range_score,
            confidence=confidence,
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            structure_high_broken=structure_high_broken,
            structure_low_broken=structure_low_broken,
            is_range_like=is_range_like,
            constraint_upper=constraint_upper,
            constraint_lower=constraint_lower,
            boundary_tolerance=boundary_tolerance,
            near_upper=near_upper,
            near_lower=near_lower,
            in_value_zone=in_value_zone,
            continuation_ready=continuation_ready,
            reversal_warning=reversal_warning,
            reason=reason,
        )

    def _unknown_state(self, ts: datetime, reason: str) -> StructureState:
        return StructureState(
            timeframe=self.config.timeframe,
            ts=ts,
            primary_bias="neutral",
            phase="unknown",
            trend_score=0.0,
            range_score=0.0,
            confidence=0.0,
            last_swing_high=None,
            last_swing_low=None,
            structure_high_broken=False,
            structure_low_broken=False,
            is_range_like=False,
            constraint_upper=None,
            constraint_lower=None,
            boundary_tolerance=0.0,
            near_upper=False,
            near_lower=False,
            in_value_zone=False,
            continuation_ready=False,
            reversal_warning=False,
            reason=reason,
        )

    def _calc_atr_proxy(self, bars: list[GenericBar]) -> float:
        """
        用简化 TR 均值近似 ATR。
        """
        if len(bars) < 2:
            return 0.0

        trs: list[float] = []
        prev_close = bars[0].close
        for bar in bars[1:]:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev_close),
                abs(bar.low - prev_close),
            )
            trs.append(tr)
            prev_close = bar.close

        return sum(trs) / max(len(trs), 1)

    def _calc_directional_efficiency(self, closes: list[float]) -> float:
        """
        方向效率：
        - 越接近 1，越像单边推进
        - 越接近 0，越像来回震荡
        """
        if len(closes) < 2:
            return 0.0

        net_move = closes[-1] - closes[0]
        total_move = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
        return abs(net_move) / max(total_move, 1e-8)

    def _detect_swings(self, bars: list[GenericBar]) -> tuple[list[SwingPoint], list[SwingPoint]]:
        """
        摆点识别：
        - 某根高点高于两侧若干根 -> swing high
        - 某根低点低于两侧若干根 -> swing low
        """
        highs: list[SwingPoint] = []
        lows: list[SwingPoint] = []

        w = self.config.swing_window
        for i in range(w, len(bars) - w):
            center = bars[i]
            left = bars[i - w : i]
            right = bars[i + 1 : i + 1 + w]

            if all(center.high > b.high for b in left) and all(center.high > b.high for b in right):
                highs.append(SwingPoint(index=i, price=center.high, kind="high"))

            if all(center.low < b.low for b in left) and all(center.low < b.low for b in right):
                lows.append(SwingPoint(index=i, price=center.low, kind="low"))

        return highs, lows

    def _effective_break_high(
        self,
        bars: list[GenericBar],
        level: Optional[float],
        atr: float,
    ) -> bool:
        """
        有效上破：
        - 不是 high > level 就算
        - 要求收盘突破，或连续两根收盘在上方
        """
        if level is None or len(bars) < 2:
            return False

        tolerance = atr * self.config.break_tolerance_atr_ratio
        last_close = bars[-1].close
        prev_close = bars[-2].close

        return (
            last_close > level + tolerance
            or (prev_close > level and last_close > level)
        )

    def _effective_break_low(
        self,
        bars: list[GenericBar],
        level: Optional[float],
        atr: float,
    ) -> bool:
        """
        有效下破：
        - 不是 low < level 就算
        - 要求收盘跌破，或连续两根收盘在下方
        """
        if level is None or len(bars) < 2:
            return False

        tolerance = atr * self.config.break_tolerance_atr_ratio
        last_close = bars[-1].close
        prev_close = bars[-2].close

        return (
            last_close < level - tolerance
            or (prev_close < level and last_close < level)
        )

    def _hh_hl_ready(
        self,
        last_swing_high: Optional[float],
        prev_swing_high: Optional[float],
        last_swing_low: Optional[float],
        prev_swing_low: Optional[float],
    ) -> bool:
        return (
            last_swing_high is not None
            and prev_swing_high is not None
            and last_swing_low is not None
            and prev_swing_low is not None
            and last_swing_high > prev_swing_high
            and last_swing_low > prev_swing_low
        )

    def _lh_ll_ready(
        self,
        last_swing_high: Optional[float],
        prev_swing_high: Optional[float],
        last_swing_low: Optional[float],
        prev_swing_low: Optional[float],
    ) -> bool:
        return (
            last_swing_high is not None
            and prev_swing_high is not None
            and last_swing_low is not None
            and prev_swing_low is not None
            and last_swing_high < prev_swing_high
            and last_swing_low < prev_swing_low
        )

    def _calc_trend_score(
        self,
        directional_efficiency: float,
        hh_hl_ready: bool,
        lh_ll_ready: bool,
    ) -> float:
        """
        趋势评分：
        - 方向效率越高，趋势越强
        - 存在清晰 HH/HL 或 LH/LL 结构，加分
        """
        structure_bonus = 0.25 if (hh_hl_ready or lh_ll_ready) else 0.0
        score = directional_efficiency * 0.75 + structure_bonus
        return round(min(1.0, score), 4)

    def _calc_range_score(
        self,
        directional_efficiency: float,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
    ) -> float:
        """
        区间评分：
        - 方向效率越低，越像整理
        - 摆点越丰富，越像有约束的来回波动
        """
        swing_count_score = min(1.0, (len(swing_highs) + len(swing_lows)) / 8.0)
        score = (1.0 - directional_efficiency) * 0.7 + swing_count_score * 0.3
        return round(min(1.0, score), 4)

    def _estimate_constraints(
        self,
        swing_highs: list[SwingPoint],
        swing_lows: list[SwingPoint],
        atr: float,
    ) -> tuple[Optional[float], Optional[float], float]:
        """
        估计约束边界。

        关键约束：
        - 不直接用单次最高/最低点
        - 优先用最近若干 swing 的中位数近似边界
        """
        recent_highs = [p.price for p in swing_highs[-3:]]
        recent_lows = [p.price for p in swing_lows[-3:]]

        constraint_upper = median(recent_highs) if len(recent_highs) >= self.config.min_swing_points else None
        constraint_lower = median(recent_lows) if len(recent_lows) >= self.config.min_swing_points else None

        tolerance = atr * self.config.boundary_tolerance_atr_ratio
        return constraint_upper, constraint_lower, tolerance

    def _is_near_upper(
        self,
        last_close: float,
        constraint_upper: Optional[float],
        atr: float,
        boundary_tolerance: float,
    ) -> bool:
        if constraint_upper is None:
            return False

        near_distance = max(boundary_tolerance, atr * self.config.near_boundary_atr_ratio)
        return abs(last_close - constraint_upper) <= near_distance or last_close >= constraint_upper - near_distance

    def _is_near_lower(
        self,
        last_close: float,
        constraint_lower: Optional[float],
        atr: float,
        boundary_tolerance: float,
    ) -> bool:
        if constraint_lower is None:
            return False

        near_distance = max(boundary_tolerance, atr * self.config.near_boundary_atr_ratio)
        return abs(last_close - constraint_lower) <= near_distance or last_close <= constraint_lower + near_distance

    def _infer_primary_bias(
        self,
        net_move: float,
        hh_hl_ready: bool,
        lh_ll_ready: bool,
    ) -> PrimaryBias:
        """
        主方向优先看结构，其次看净位移。
        """
        if hh_hl_ready and not lh_ll_ready:
            return "long"
        if lh_ll_ready and not hh_hl_ready:
            return "short"

        if net_move > 0:
            return "long"
        if net_move < 0:
            return "short"
        return "neutral"

    def _infer_phase(
        self,
        primary_bias: PrimaryBias,
        is_range_like: bool,
        structure_high_broken: bool,
        structure_low_broken: bool,
        hh_hl_ready: bool,
        lh_ll_ready: bool,
        window: list[GenericBar],
    ) -> Phase:
        """
        阶段判断：
        - range 不等于 neutral
        - 整理依旧归属于原趋势体系
        """
        if primary_bias == "neutral":
            return "unknown"

        if primary_bias == "short":
            if structure_high_broken and hh_hl_ready:
                return "reversal_candidate"
            if is_range_like:
                return "range"
            if self._is_counter_move(window, direction="short"):
                return "pullback"
            if lh_ll_ready:
                return "impulse"
            return "unknown"

        if primary_bias == "long":
            if structure_low_broken and lh_ll_ready:
                return "reversal_candidate"
            if is_range_like:
                return "range"
            if self._is_counter_move(window, direction="long"):
                return "pullback"
            if hh_hl_ready:
                return "impulse"
            return "unknown"

        return "unknown"

    def _is_counter_move(self, bars: list[GenericBar], direction: str) -> bool:
        """
        判断最近几根是否在走原方向的反向运动。
        """
        if len(bars) < 4:
            return False

        recent = bars[-4:]
        move = recent[-1].close - recent[0].close

        if direction == "short":
            return move > 0
        return move < 0

    def _build_reason(
        self,
        primary_bias: PrimaryBias,
        phase: Phase,
        trend_score: float,
        range_score: float,
        directional_efficiency: float,
        near_upper: bool,
        near_lower: bool,
        structure_high_broken: bool,
        structure_low_broken: bool,
    ) -> str:
        return (
            f"{self.config.timeframe} 结构分析："
            f"bias={primary_bias}，phase={phase}，"
            f"trend_score={trend_score:.4f}，range_score={range_score:.4f}，"
            f"efficiency={directional_efficiency:.4f}，"
            f"near_upper={near_upper}，near_lower={near_lower}，"
            f"high_broken={structure_high_broken}，low_broken={structure_low_broken}"
        )