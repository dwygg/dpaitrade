from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from dpaitrade.core.types import CandidateSignal, MarketState, PortfolioState, RiskDecision, TradeRecord
from dpaitrade.structure import GenericBar, StructureAnalyzer, StructureAnalyzerConfig


@dataclass(slots=True)
class SimulationConfig:
    fee_rate: float = 0.0002
    slippage: float = 0.0
    fallback_exit_to_entry: bool = True
    default_holding_minutes: int = 15

    break_even_trigger_r: float = 1.5
    break_even_offset: float = 0.0

    trailing_buffer_atr_ratio: float = 0.10
    trailing_buffer_spread_multiple: float = 1.0

    dominant_invalid_confirm_bars: int = 1

    progress_check_minutes: int = 180
    min_progress_r_after_progress_check: float = 0.30

    max_holding_minutes: int = 720
    min_unrealized_r_at_max_holding: float = 0.10


class SimpleExecutionSimulator:
    """顺势执行模拟器（主导周期一致性版）。

    逻辑：
    1. 初始止损
    2. 达到 break-even trigger 后移到保本
    3. 保本后，按主导周期已完成 bar 收紧 trailing stop
    4. 主导周期结构失效离场
    5. 持仓时长 + 盈亏判断离场
    6. 到期按 future_close 平仓
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.execution.simulator")
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

    def execute(
        self,
        market_state: MarketState,
        signal: CandidateSignal,
        risk_decision: RiskDecision,
        portfolio_state: PortfolioState,
    ) -> Optional[TradeRecord]:
        if not risk_decision.approved:
            self.logger.info("执行模拟跳过：风控未通过")
            return None
        if signal.direction not in ("long", "short"):
            self.logger.info("执行模拟跳过：候选信号方向无效")
            return None

        quantity = self._calculate_quantity(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            equity=portfolio_state.equity,
            risk_pct=risk_decision.risk_pct,
        )
        if quantity <= 0:
            self.logger.info("执行模拟跳过：计算出的下单数量小于等于 0")
            return None

        future_bars = market_state.meta.get("future_bars", [])
        dominant_exec_bars = market_state.meta.get("dominant_exec_bars", [])
        dominant_state = market_state.meta.get("dominant_state")
        dominant_analyzer_cfg = market_state.meta.get("dominant_analyzer_cfg", {})

        if not future_bars:
            return self._fallback_execute_with_legacy_fields(
                market_state=market_state,
                signal=signal,
                risk_decision=risk_decision,
                portfolio_state=portfolio_state,
                quantity=quantity,
            )

        entry_price = signal.entry_price
        initial_stop = signal.stop_loss
        current_stop = initial_stop
        initial_risk = abs(entry_price - initial_stop)
        if initial_risk <= 0:
            self.logger.info("执行模拟跳过：初始风险距离无效")
            return None

        atr = float(market_state.atr or 0.0)
        spread = float(market_state.spread or 0.0)
        trailing_buffer = max(
            atr * self.config.trailing_buffer_atr_ratio,
            spread * self.config.trailing_buffer_spread_multiple,
        )

        break_even_armed = False
        m15_history: list[dict] = []
        exit_price: Optional[float] = None
        close_reason = ""
        close_time: Optional[datetime] = None

        dominant_timeframe = None
        if dominant_state is not None:
            dominant_timeframe = getattr(dominant_state, "timeframe", None)
        if not dominant_timeframe:
            dominant_timeframe = market_state.meta.get("dominant_timeframe", "M15")

        for bar in future_bars:
            bar_ts = bar["ts"]
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            if signal.direction == "short":
                if bar_high >= current_stop:
                    exit_price = current_stop + self.config.slippage
                    close_time = bar_ts
                    if self._is_break_even_stop(current_stop, entry_price):
                        close_reason = "保本止损"
                    elif current_stop != initial_stop:
                        close_reason = "移动止损"
                    else:
                        close_reason = "命中止损"
                    break
                if signal.take_profit is not None and bar_low <= signal.take_profit:
                    exit_price = signal.take_profit + self.config.slippage
                    close_time = bar_ts
                    close_reason = "命中止盈"
                    break
            else:
                if bar_low <= current_stop:
                    exit_price = current_stop - self.config.slippage
                    close_time = bar_ts
                    if self._is_break_even_stop(current_stop, entry_price):
                        close_reason = "保本止损"
                    elif current_stop != initial_stop:
                        close_reason = "移动止损"
                    else:
                        close_reason = "命中止损"
                    break
                if signal.take_profit is not None and bar_high >= signal.take_profit:
                    exit_price = signal.take_profit - self.config.slippage
                    close_time = bar_ts
                    close_reason = "命中止盈"
                    break

            m15_history.append(
                {
                    "ts": bar_ts,
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                }
            )

            max_favorable_move = self._calc_max_favorable_move(
                direction=signal.direction,
                entry_price=entry_price,
                history=m15_history,
            )
            best_favorable_r = max_favorable_move / max(initial_risk, 1e-8)
            current_unrealized_r = self._calc_unrealized_r(
                direction=signal.direction,
                entry_price=entry_price,
                current_price=bar_close,
                initial_risk=initial_risk,
            )

            if not break_even_armed and best_favorable_r >= self.config.break_even_trigger_r:
                break_even_armed = True
                current_stop = self._break_even_price(entry_price, signal.direction)
                self.logger.info(
                    "触发保本：direction=%s，entry=%.5f，new_stop=%.5f，达到%.2fR",
                    signal.direction,
                    entry_price,
                    current_stop,
                    self.config.break_even_trigger_r,
                )

            closed_exec_bars = self._collect_closed_exec_bars(
                exec_bars=dominant_exec_bars,
                current_ts=bar_ts,
                timeframe=dominant_timeframe,
            )

            if break_even_armed:
                new_stop = self._calc_dominant_trailing_stop(
                    closed_exec_bars=closed_exec_bars,
                    direction=signal.direction,
                    current_stop=current_stop,
                    trailing_buffer=trailing_buffer,
                )
                if new_stop is not None:
                    current_stop = new_stop

            if self._detect_dominant_invalidation_exit(
                direction=signal.direction,
                closed_exec_bars=closed_exec_bars,
                dominant_analyzer_cfg=dominant_analyzer_cfg,
            ):
                exit_price = self._apply_exit_slippage(signal.direction, bar_close)
                close_time = bar_ts
                close_reason = "主导周期失效离场"
                break

            timed_exit_reason = self._detect_time_and_pnl_exit(
                entry_ts=market_state.ts,
                current_ts=bar_ts,
                best_favorable_r=best_favorable_r,
                current_unrealized_r=current_unrealized_r,
            )
            if timed_exit_reason:
                exit_price = self._apply_exit_slippage(signal.direction, bar_close)
                close_time = bar_ts
                close_reason = timed_exit_reason
                break

        if exit_price is None:
            if future_bars:
                last_bar = future_bars[-1]
                exit_price = float(last_bar["close"])
                close_time = last_bar["ts"]
                close_reason = "到期按 future_close 平仓"
            elif self.config.fallback_exit_to_entry:
                exit_price = entry_price
                close_time = market_state.ts + timedelta(minutes=self.config.default_holding_minutes)
                close_reason = "缺少未来价格数据，按入场价平仓"
            else:
                self.logger.info("执行模拟跳过：缺少未来路径，且未启用 fallback")
                return None

        fees = self._calculate_fees(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
        )
        pnl = self._calculate_pnl(
            direction=signal.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            fees=fees,
        )
        pnl_r = self._calculate_r_multiple(
            pnl=pnl,
            equity=portfolio_state.equity,
            risk_pct=risk_decision.risk_pct,
        )

        self.logger.info(
            "执行模拟完成：direction=%s，entry=%.5f，exit=%.5f，quantity=%.5f，pnl=%.2f，close_reason=%s",
            signal.direction,
            entry_price,
            exit_price,
            quantity,
            pnl,
            close_reason,
        )
        return TradeRecord(
            symbol=signal.symbol,
            direction=signal.direction,
            open_time=market_state.ts,
            close_time=close_time or self._estimate_close_time(market_state.ts),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_r=pnl_r,
            fees=fees,
            open_reason=signal.reason,
            close_reason=close_reason,
            meta={
                "risk_pct": risk_decision.risk_pct,
                "initial_stop": initial_stop,
                "final_stop": current_stop,
                "break_even_armed": break_even_armed,
            },
        )

    def _collect_closed_exec_bars(
        self,
        exec_bars: list[dict],
        current_ts: datetime,
        timeframe: str,
    ) -> list[dict]:
        closed: list[dict] = []
        bar_delta = timedelta(minutes=self._timeframe_minutes(timeframe))
        for bar in exec_bars:
            bar_end = bar["ts"] + bar_delta
            if bar_end <= current_ts:
                closed.append(bar)
        return closed

    @staticmethod
    def _timeframe_minutes(timeframe: str) -> int:
        tf = (timeframe or "M15").upper()
        mapping = {
            "M15": 15,
            "H4": 240,
            "D1": 1440,
        }
        return mapping.get(tf, 15)

    def _calc_max_favorable_move(
        self,
        direction: str,
        entry_price: float,
        history: list[dict],
    ) -> float:
        if direction == "short":
            return entry_price - min(h["low"] for h in history)
        return max(h["high"] for h in history) - entry_price

    @staticmethod
    def _calc_unrealized_r(
        direction: str,
        entry_price: float,
        current_price: float,
        initial_risk: float,
    ) -> float:
        if direction == "short":
            return (entry_price - current_price) / max(initial_risk, 1e-8)
        return (current_price - entry_price) / max(initial_risk, 1e-8)

    def _break_even_price(self, entry_price: float, direction: str) -> float:
        if direction == "short":
            return entry_price - self.config.break_even_offset
        return entry_price + self.config.break_even_offset

    @staticmethod
    def _is_break_even_stop(current_stop: float, entry_price: float) -> bool:
        return abs(current_stop - entry_price) < 1e-8

    def _calc_dominant_trailing_stop(
        self,
        closed_exec_bars: list[dict],
        direction: str,
        current_stop: float,
        trailing_buffer: float,
    ) -> Optional[float]:
        if not closed_exec_bars:
            return None

        last_closed = closed_exec_bars[-1]
        if direction == "short":
            candidate = float(last_closed["high"]) + trailing_buffer
            return candidate if candidate < current_stop else current_stop

        candidate = float(last_closed["low"]) - trailing_buffer
        return candidate if candidate > current_stop else current_stop

    def _detect_dominant_invalidation_exit(
        self,
        direction: str,
        closed_exec_bars: list[dict],
        dominant_analyzer_cfg: dict,
    ) -> bool:
        if not closed_exec_bars or not dominant_analyzer_cfg:
            return False

        confirm_n = max(int(self.config.dominant_invalid_confirm_bars), 1)
        lookback = int(dominant_analyzer_cfg.get("lookback", 0) or 0)
        if lookback <= 0 or len(closed_exec_bars) < lookback:
            return False

        invalid_flags: list[bool] = []
        for offset in range(confirm_n):
            usable = len(closed_exec_bars) - offset
            if usable < lookback:
                return False
            window = closed_exec_bars[:usable][-lookback:]
            state = self._analyze_exec_window(window, dominant_analyzer_cfg)
            if direction == "long":
                invalid_flags.append(
                    state.primary_bias == "short" or state.phase == "reversal_candidate"
                )
            else:
                invalid_flags.append(
                    state.primary_bias == "long" or state.phase == "reversal_candidate"
                )
        return all(invalid_flags)

    def _analyze_exec_window(self, window: list[dict], analyzer_cfg: dict):
        bars = [
            GenericBar(
                ts=bar["ts"],
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
            )
            for bar in window
        ]
        analyzer = StructureAnalyzer(
            StructureAnalyzerConfig(
                timeframe=analyzer_cfg.get("timeframe", "M15"),
                min_bars=int(analyzer_cfg.get("min_bars", len(bars))),
                lookback=int(analyzer_cfg.get("lookback", len(bars))),
                swing_window=int(analyzer_cfg.get("swing_window", 2)),
                trend_threshold=float(analyzer_cfg.get("trend_threshold", 0.34)),
                range_threshold=float(analyzer_cfg.get("range_threshold", 0.60)),
                near_boundary_atr_ratio=float(analyzer_cfg.get("near_boundary_atr_ratio", 0.35)),
            )
        )
        return analyzer.analyze(bars)

    def _detect_time_and_pnl_exit(
        self,
        entry_ts: datetime,
        current_ts: datetime,
        best_favorable_r: float,
        current_unrealized_r: float,
    ) -> str:
        elapsed_minutes = int((current_ts - entry_ts).total_seconds() // 60)

        if (
            elapsed_minutes >= self.config.progress_check_minutes
            and best_favorable_r < self.config.min_progress_r_after_progress_check
            and current_unrealized_r <= 0.0
        ):
            return "持仓进展不足离场"

        if (
            elapsed_minutes >= self.config.max_holding_minutes
            and current_unrealized_r < self.config.min_unrealized_r_at_max_holding
        ):
            return "持仓超时离场"

        return ""

    def _apply_exit_slippage(self, direction: str, price: float) -> float:
        if direction == "short":
            return price + self.config.slippage
        return price - self.config.slippage

    def _fallback_execute_with_legacy_fields(
        self,
        market_state: MarketState,
        signal: CandidateSignal,
        risk_decision: RiskDecision,
        portfolio_state: PortfolioState,
        quantity: float,
    ) -> Optional[TradeRecord]:
        future_high = market_state.meta.get("future_high")
        future_low = market_state.meta.get("future_low")
        future_close = market_state.meta.get("future_close")

        exit_price: Optional[float] = None
        close_reason = ""

        if signal.direction == "long":
            if future_low is not None and future_low <= signal.stop_loss:
                exit_price = signal.stop_loss - self.config.slippage
                close_reason = "命中止损"
        else:
            if future_high is not None and future_high >= signal.stop_loss:
                exit_price = signal.stop_loss + self.config.slippage
                close_reason = "命中止损"

        if exit_price is None:
            if future_close is not None:
                exit_price = float(future_close)
                close_reason = "到期按 future_close 平仓"
            elif self.config.fallback_exit_to_entry:
                exit_price = signal.entry_price
                close_reason = "缺少未来价格数据，按入场价平仓"
            else:
                return None

        fees = self._calculate_fees(signal.entry_price, exit_price, quantity)
        pnl = self._calculate_pnl(signal.direction, signal.entry_price, exit_price, quantity, fees)
        pnl_r = self._calculate_r_multiple(pnl, portfolio_state.equity, risk_decision.risk_pct)
        return TradeRecord(
            symbol=signal.symbol,
            direction=signal.direction,
            open_time=market_state.ts,
            close_time=self._estimate_close_time(market_state.ts),
            entry_price=signal.entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_r=pnl_r,
            fees=fees,
            open_reason=signal.reason,
            close_reason=close_reason,
            meta={
                "risk_pct": risk_decision.risk_pct,
                "legacy_mode": True,
            },
        )

    def _calculate_quantity(
        self,
        entry_price: float,
        stop_loss: float,
        equity: float,
        risk_pct: float,
    ) -> float:
        risk_amount = equity * max(risk_pct, 0.0)
        unit_risk = abs(entry_price - stop_loss)
        if unit_risk <= 0:
            return 0.0
        return round(risk_amount / unit_risk, 6)

    def _calculate_fees(self, entry_price: float, exit_price: float, quantity: float) -> float:
        turnover = (entry_price + exit_price) * quantity
        return turnover * self.config.fee_rate

    @staticmethod
    def _calculate_pnl(
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        fees: float,
    ) -> float:
        if direction == "long":
            gross = (exit_price - entry_price) * quantity
        else:
            gross = (entry_price - exit_price) * quantity
        return gross - fees

    @staticmethod
    def _calculate_r_multiple(pnl: float, equity: float, risk_pct: float) -> float:
        one_r = equity * max(risk_pct, 1e-8)
        return pnl / max(one_r, 1e-8)

    def _estimate_close_time(self, ts: datetime) -> datetime:
        return ts + timedelta(minutes=self.config.default_holding_minutes)
