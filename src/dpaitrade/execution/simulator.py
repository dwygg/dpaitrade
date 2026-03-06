from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from dpaitrade.core.types import CandidateSignal, MarketState, PortfolioState, RiskDecision, TradeRecord


@dataclass(slots=True)
class SimulationConfig:
    """
    执行模拟配置。

    当前版本先做“单步近似成交模拟”：
    - 通过 market_state.meta 中预先提供的 future_high / future_low / future_close
      来模拟本次候选信号后续的价格表现
    - 若未来价格先到达止盈或止损，则按命中处理
    - 若都未命中，则按 future_close 平仓

    这是为了先把回测主链路打通。
    后续可以升级为：
    - 多 bar 持仓生命周期
    - 分批止盈
    - 移动止损
    - 滑点与手续费的更细建模
    """

    fee_rate: float = 0.0002
    slippage: float = 0.0
    default_holding_minutes: int = 15
    fallback_exit_to_entry: bool = True


class SimpleExecutionSimulator:
    """
    简单执行模拟器。

    设计说明：
    - 当前用于第一版工程链路验证
    - 它不追求完整市场微观结构，只追求“可测、可扩展、能出交易记录”
    - 后续你可以用更严格的逐 bar / 逐 tick 模拟器替换它
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
        """
        执行一次近似交易模拟。

        约定：
        - future_high / future_low / future_close 从 market_state.meta 中读取
        - 若缺失这些字段，则按 fallback 逻辑处理
        """
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
            direction=signal.direction,
        )
        if quantity <= 0:
            self.logger.info("执行模拟跳过：计算出的下单数量小于等于 0")
            return None

        future_high = market_state.meta.get("future_high")
        future_low = market_state.meta.get("future_low")
        future_close = market_state.meta.get("future_close")

        exit_price: Optional[float] = None
        close_reason = ""

        # 对多头：
        # 先检查 future_low 是否触发止损，再检查 future_high 是否触发止盈。
        # 这里是一种偏保守处理，默认先止损后止盈，避免回测过于乐观。
        if signal.direction == "long":
            if future_low is not None and future_low <= signal.stop_loss:
                exit_price = signal.stop_loss - self.config.slippage
                close_reason = "命中止损"
            elif signal.take_profit is not None and future_high is not None and future_high >= signal.take_profit:
                exit_price = signal.take_profit - self.config.slippage
                close_reason = "命中止盈"

        # 对空头：
        # 同样采用偏保守顺序，默认先检查止损，再检查止盈。
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
                self.logger.info("执行模拟跳过：缺少未来价格数据，且未启用 fallback")
                return None

        fees = self._calculate_fees(
            entry_price=signal.entry_price,
            exit_price=exit_price,
            quantity=quantity,
        )
        pnl = self._calculate_pnl(
            direction=signal.direction,
            entry_price=signal.entry_price,
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
            signal.entry_price,
            exit_price,
            quantity,
            pnl,
            close_reason,
        )

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
                "future_high": future_high,
                "future_low": future_low,
                "future_close": future_close,
                "risk_pct": risk_decision.risk_pct,
            },
        )

    def _calculate_quantity(
        self,
        entry_price: float,
        stop_loss: float,
        equity: float,
        risk_pct: float,
        direction: str,
    ) -> float:
        """
        根据固定风险比例估算下单数量。

        计算思路：
        - 风险金额 = equity * risk_pct
        - 单位风险 = |entry - stop_loss|
        - 数量 = 风险金额 / 单位风险
        """
        risk_amount = equity * max(risk_pct, 0.0)
        unit_risk = abs(entry_price - stop_loss)
        if unit_risk <= 0:
            return 0.0
        return round(risk_amount / unit_risk, 6)

    def _calculate_fees(self, entry_price: float, exit_price: float, quantity: float) -> float:
        """
        计算手续费。

        当前采用非常简单的双边费率模型。
        """
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
        """
        计算净盈亏。
        """
        if direction == "long":
            gross = (exit_price - entry_price) * quantity
        else:
            gross = (entry_price - exit_price) * quantity
        return gross - fees

    @staticmethod
    def _calculate_r_multiple(pnl: float, equity: float, risk_pct: float) -> float:
        """
        计算 R 值。

        当前定义：
        - 1R = equity * risk_pct
        """
        one_r = equity * max(risk_pct, 1e-8)
        return pnl / max(one_r, 1e-8)

    def _estimate_close_time(self, ts: datetime) -> datetime:
        """
        估算平仓时间。

        由于当前是近似模拟，不追踪真实持仓生命周期，
        因此默认给一个固定持仓时长。
        """
        return ts + timedelta(minutes=self.config.default_holding_minutes)
