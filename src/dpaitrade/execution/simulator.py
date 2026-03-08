from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from dpaitrade.core.types import CandidateSignal, MarketState, PortfolioState, RiskDecision, TradeRecord


@dataclass(slots=True)
class SimulationConfig:
    fee_rate: float = 0.0002
    slippage: float = 0.0
    fallback_exit_to_entry: bool = True
    default_holding_minutes: int = 15

    # 保留旧字段，避免其他调用方炸掉；SwingPointPolicy 默认不使用它们
    break_even_trigger_r: float = 99.0
    break_even_offset: float = 0.0
    trailing_buffer_atr_ratio: float = 0.10
    trailing_buffer_spread_multiple: float = 1.0
    dominant_invalid_confirm_bars: int = 1

    # 摆点失效确认根数：至少 N 根 M15 收盘越过摆点才确认失效，避免单根假突破触发离场
    swing_invalidation_confirm_bars: int = 2

    progress_check_minutes: int = 240
    min_progress_r_after_progress_check: float = 0.15

    max_holding_minutes: int = 1440
    min_unrealized_r_at_max_holding: float = 0.10


class SimpleExecutionSimulator:
    """统一执行模拟器（v2）。

    对 swing_point_v2：
    - 入场后只看硬止损 / 固定止盈 / 摆点失效 / 时间与盈亏退出
    - 不使用趋势跟踪式 break-even / trailing stop
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
            self.logger.debug("执行模拟跳过：风控未通过")
            return None
        if signal.direction not in ("long", "short"):
            self.logger.debug("执行模拟跳过：候选信号方向无效")
            return None

        quantity = self._calculate_quantity(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            equity=portfolio_state.equity,
            risk_pct=risk_decision.risk_pct,
        )
        if quantity <= 0:
            self.logger.debug("执行模拟跳过：计算出的下单数量小于等于 0")
            return None

        future_bars = market_state.meta.get("future_bars", [])
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
        initial_risk = abs(entry_price - initial_stop)
        if initial_risk <= 0:
            self.logger.debug("执行模拟跳过：初始风险距离无效")
            return None

        swing_policy = str(signal.meta.get("policy", "")).startswith("swing_point")
        swing_low = signal.meta.get("swing_low")
        swing_high = signal.meta.get("swing_high")

        m15_history: list[dict] = []
        exit_price: Optional[float] = None
        close_reason = ""
        close_time: Optional[datetime] = None

        for bar in future_bars:
            bar_ts = bar["ts"]
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            if signal.direction == "short":
                if bar_high >= initial_stop:
                    exit_price = initial_stop + self.config.slippage
                    close_reason = "命中止损"
                    close_time = bar_ts
                    break
                if signal.take_profit is not None and bar_low <= signal.take_profit:
                    exit_price = signal.take_profit + self.config.slippage
                    close_reason = "命中止盈"
                    close_time = bar_ts
                    break
            else:
                if bar_low <= initial_stop:
                    exit_price = initial_stop - self.config.slippage
                    close_reason = "命中止损"
                    close_time = bar_ts
                    break
                if signal.take_profit is not None and bar_high >= signal.take_profit:
                    exit_price = signal.take_profit - self.config.slippage
                    close_reason = "命中止盈"
                    close_time = bar_ts
                    break

            m15_history.append(
                {
                    "ts": bar_ts,
                    "open": float(bar["open"]),
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                }
            )

            max_favorable_move = self._calc_max_favorable_move(signal.direction, entry_price, m15_history)
            best_favorable_r = max_favorable_move / max(initial_risk, 1e-8)
            current_unrealized_r = self._calc_unrealized_r(signal.direction, entry_price, bar_close, initial_risk)

            if swing_policy and self._detect_swing_invalidation_exit(
                direction=signal.direction,
                history=m15_history,
                swing_low=swing_low,
                swing_high=swing_high,
            ):
                exit_price = self._apply_exit_slippage(signal.direction, bar_close)
                close_reason = "摆点失效离场"
                close_time = bar_ts
                self.logger.debug(
                    "摆点失效离场：direction=%s，bar_close=%.5f，swing_low=%s，swing_high=%s",
                    signal.direction, bar_close, swing_low, swing_high,
                )
                break

            timed_exit_reason = self._detect_time_and_pnl_exit(
                entry_ts=market_state.ts,
                current_ts=bar_ts,
                best_favorable_r=best_favorable_r,
                current_unrealized_r=current_unrealized_r,
            )
            if timed_exit_reason:
                exit_price = self._apply_exit_slippage(signal.direction, bar_close)
                close_reason = timed_exit_reason
                close_time = bar_ts
                self.logger.debug("时间/盈亏退出：%s，direction=%s", timed_exit_reason, signal.direction)
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
                self.logger.debug("执行模拟跳过：缺少未来路径，且未启用 fallback")
                return None

        fees = self._calculate_fees(entry_price=entry_price, exit_price=exit_price, quantity=quantity)
        pnl = self._calculate_pnl(signal.direction, entry_price, exit_price, quantity, fees)
        pnl_r = self._calculate_r_multiple(pnl, portfolio_state.equity, risk_decision.risk_pct)

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
                "policy": signal.meta.get("policy"),
                "swing_id": signal.meta.get("swing_id"),
                "swing_low": signal.meta.get("swing_low"),
                "swing_high": signal.meta.get("swing_high"),
                "zone_lower": signal.meta.get("zone_lower"),
                "zone_upper": signal.meta.get("zone_upper"),
                "max_attempts_per_swing": signal.meta.get("max_attempts_per_swing"),
                "max_cumulative_loss_r_per_swing": signal.meta.get("max_cumulative_loss_r_per_swing"),
                "require_reset_after_loss": signal.meta.get("require_reset_after_loss"),
            },
        )

    def _detect_swing_invalidation_exit(
        self,
        direction: str,
        history: list[dict],
        swing_low: Optional[float],
        swing_high: Optional[float],
    ) -> bool:
        confirm_n = max(int(self.config.swing_invalidation_confirm_bars), 1)
        if len(history) < confirm_n:
            return False
        recent = history[-confirm_n:]
        if direction == "long":
            if swing_low is None:
                return False
            return all(float(bar["close"]) < float(swing_low) for bar in recent)
        if swing_high is None:
            return False
        return all(float(bar["close"]) > float(swing_high) for bar in recent)

    def _calc_max_favorable_move(self, direction: str, entry_price: float, history: list[dict]) -> float:
        if direction == "short":
            return entry_price - min(h["low"] for h in history)
        return max(h["high"] for h in history) - entry_price

    @staticmethod
    def _calc_unrealized_r(direction: str, entry_price: float, current_price: float, initial_risk: float) -> float:
        if direction == "short":
            return (entry_price - current_price) / max(initial_risk, 1e-8)
        return (current_price - entry_price) / max(initial_risk, 1e-8)

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
            elif signal.take_profit is not None and future_high is not None and future_high >= signal.take_profit:
                exit_price = signal.take_profit - self.config.slippage
                close_reason = "命中止盈"
        else:
            if future_high is not None and future_high >= signal.stop_loss:
                exit_price = signal.stop_loss + self.config.slippage
                close_reason = "命中止损"
            elif signal.take_profit is not None and future_low is not None and future_low <= signal.take_profit:
                exit_price = signal.take_profit + self.config.slippage
                close_reason = "命中止盈"
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
            meta={"risk_pct": risk_decision.risk_pct, "legacy_mode": True},
        )

    def _calculate_quantity(self, entry_price: float, stop_loss: float, equity: float, risk_pct: float) -> float:
        risk_amount = equity * max(risk_pct, 0.0)
        unit_risk = abs(entry_price - stop_loss)
        if unit_risk <= 0:
            return 0.0
        return round(risk_amount / unit_risk, 6)

    def _calculate_fees(self, entry_price: float, exit_price: float, quantity: float) -> float:
        turnover = (entry_price + exit_price) * quantity
        return turnover * self.config.fee_rate

    @staticmethod
    def _calculate_pnl(direction: str, entry_price: float, exit_price: float, quantity: float, fees: float) -> float:
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
