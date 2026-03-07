from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from dpaitrade.core.types import CandidateSignal, MarketState, PortfolioState, RiskDecision, TradeRecord


@dataclass(slots=True)
class SimulationConfig:
    """
    执行模拟配置（H4结构 trailing 版）。

    逻辑：
    1. 初始止损
    2. 最大浮盈达到 1.5R 后，止损移到保本
    3. 保本后，仅允许 H4 结构 trailing stop 收紧
    4. H4 波段失效离场
    5. 到期按 future_close 平仓
    """

    fee_rate: float = 0.0002
    slippage: float = 0.0

    fallback_exit_to_entry: bool = True
    default_holding_minutes: int = 15

    # 达到 1.5R 后移到保本
    break_even_trigger_r: float = 1.5
    break_even_offset: float = 0.0

    # H4 trailing stop
    h4_bar_hours: int = 4
    trailing_buffer_atr_ratio: float = 0.10
    trailing_buffer_spread_multiple: float = 1.0

    # H4 失效离场确认
    h4_invalid_confirm_bars: int = 2


class SimpleExecutionSimulator:
    """
    顺势执行模拟器（H4 trailing 版）。

    注意：
    - M15 不再参与离场
    - H4 负责波段结构 trailing 与失效离场
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
        h4_exec_bars = market_state.meta.get("h4_exec_bars", [])
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

        mid_tf_state = market_state.meta.get("h4_state")

        for bar in future_bars:
            bar_ts = bar["ts"]
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            # 1) 当前 stop 是否被打到
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

            # 更新 M15 路径历史（仅用于浮盈计算，不用于离场）
            m15_history.append(
                {
                    "ts": bar_ts,
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                }
            )

            # 2) 最大浮盈达到 1.5R 后移到保本
            if not break_even_armed:
                max_favorable_move = self._calc_max_favorable_move(
                    direction=signal.direction,
                    entry_price=entry_price,
                    history=m15_history,
                )
                if max_favorable_move >= initial_risk * self.config.break_even_trigger_r:
                    break_even_armed = True
                    current_stop = self._break_even_price(entry_price, signal.direction)
                    self.logger.info(
                        "触发保本：direction=%s，entry=%.5f，new_stop=%.5f，达到%.2fR",
                        signal.direction,
                        entry_price,
                        current_stop,
                        self.config.break_even_trigger_r,
                    )

            # 3) 获取当前时刻之前“已完成”的 H4 bars
            closed_h4_bars = self._collect_closed_h4_bars(
                h4_exec_bars=h4_exec_bars,
                current_ts=bar_ts,
            )

            # 4) 保本后，只允许 H4 结构 trailing
            if break_even_armed:
                new_stop = self._calc_h4_trailing_stop(
                    closed_h4_bars=closed_h4_bars,
                    direction=signal.direction,
                    current_stop=current_stop,
                    trailing_buffer=trailing_buffer,
                )
                if new_stop is not None:
                    current_stop = new_stop

            # 5) H4 失效离场（不依赖 M15）
            if self._detect_h4_invalidation_exit(
                direction=signal.direction,
                closed_h4_bars=closed_h4_bars,
                mid_tf_state=mid_tf_state,
            ):
                exit_price = (
                    bar_close + self.config.slippage
                    if signal.direction == "short"
                    else bar_close - self.config.slippage
                )
                close_time = bar_ts
                close_reason = "H4失效离场"
                break

        # 6) 到期按最后一根 future_close 离场
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

    def _collect_closed_h4_bars(self, h4_exec_bars: list[dict], current_ts: datetime) -> list[dict]:
        """
        收集当前时刻之前已经完成的 H4 bars。

        约定：
        - h4_exec_bars 中的 ts 是 H4 bar 的开盘时间
        - 只有 ts + 4h <= current_ts，才认为该 H4 bar 已完成
        """
        closed: list[dict] = []
        for bar in h4_exec_bars:
            bar_end = bar["ts"] + timedelta(hours=self.config.h4_bar_hours)
            if bar_end <= current_ts:
                closed.append(bar)
        return closed

    def _calc_max_favorable_move(
        self,
        direction: str,
        entry_price: float,
        history: list[dict],
    ) -> float:
        if direction == "short":
            return entry_price - min(h["low"] for h in history)
        return max(h["high"] for h in history) - entry_price

    def _break_even_price(self, entry_price: float, direction: str) -> float:
        if direction == "short":
            return entry_price - self.config.break_even_offset
        return entry_price + self.config.break_even_offset

    @staticmethod
    def _is_break_even_stop(current_stop: float, entry_price: float) -> bool:
        return abs(current_stop - entry_price) < 1e-8

    def _calc_h4_trailing_stop(
        self,
        closed_h4_bars: list[dict],
        direction: str,
        current_stop: float,
        trailing_buffer: float,
    ) -> Optional[float]:
        """
        H4 结构 trailing stop：

        short:
        - 用最近一根已完成 H4 bar 的高点 + buffer 作为新的结构止损
        - 仅允许向下收紧

        long:
        - 用最近一根已完成 H4 bar 的低点 - buffer
        - 仅允许向上收紧
        """
        if not closed_h4_bars:
            return None

        last_closed = closed_h4_bars[-1]

        if direction == "short":
            candidate = float(last_closed["high"]) + trailing_buffer
            return candidate if candidate < current_stop else current_stop

        candidate = float(last_closed["low"]) - trailing_buffer
        return candidate if candidate > current_stop else current_stop

    def _detect_h4_invalidation_exit(self, direction: str, closed_h4_bars: list[dict], mid_tf_state) -> bool:
        """
        H4 波段失效离场：

        short:
        - 连续 N 根已完成 H4 收盘 > constraint_upper + tolerance

        long:
        - 连续 N 根已完成 H4 收盘 < constraint_lower - tolerance
        """
        if mid_tf_state is None:
            return False

        confirm_n = self.config.h4_invalid_confirm_bars
        if len(closed_h4_bars) < confirm_n:
            return False

        recent = closed_h4_bars[-confirm_n:]

        upper = getattr(mid_tf_state, "constraint_upper", None)
        lower = getattr(mid_tf_state, "constraint_lower", None)
        tol = float(getattr(mid_tf_state, "boundary_tolerance", 0.0) or 0.0)

        if direction == "short":
            if upper is None:
                return False
            return all(float(bar["close"]) > upper + tol for bar in recent)

        if lower is None:
            return False
        return all(float(bar["close"]) < lower - tol for bar in recent)

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