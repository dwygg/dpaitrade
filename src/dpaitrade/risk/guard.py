from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from dpaitrade.core.types import AgentDecision, CandidateSignal, MarketState, PortfolioState, RiskDecision


@dataclass(slots=True)
class RiskGuardConfig:
    """
    风控守卫配置。

    设计原则：
    - 所有约束尽量显式配置
    - 风控必须是确定性的，不允许由 AI 自由发挥
    - 当前先覆盖最基础但最常用的风险限制
    """

    default_risk_pct: float = 0.005
    max_risk_pct_per_trade: float = 0.01
    max_used_risk_pct: float = 0.02
    max_consecutive_losses: int = 3
    max_daily_loss_pct: float = 0.02
    max_open_positions: int = 1
    max_spread: float = 5.0
    min_setup_score: float = 0.45
    min_rr: float = 1.2
    reject_direction_conflict: bool = True


class GuardRiskManager:
    """
    风控守卫。

    当前模块负责把：
    - 市场环境约束
    - 账户风险约束
    - Agent 输出约束
    - 候选信号质量约束
    统一收口到一个 review() 方法中。

    设计重点：
    1. 风控审批必须可解释
    2. 每个拒绝分支都给出中文原因
    3. 后续可以继续扩展更细的风险项，例如：
       - 新闻时段禁开仓
       - 同方向仓位限制
       - 分级冷却
       - 策略级熔断
    """

    def __init__(
        self,
        config: Optional[RiskGuardConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or RiskGuardConfig()
        self.logger = logger or self._build_logger()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("dpaitrade.risk.guard")
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

    def review(
        self,
        market_state: MarketState,
        signal: CandidateSignal,
        agent_decision: AgentDecision,
        portfolio_state: PortfolioState,
    ) -> RiskDecision:
        """
        审核当前候选信号是否允许进入执行阶段。

        返回：
        - RiskDecision.approve(): 风控通过
        - RiskDecision.reject(): 风控拒绝
        """
        self.logger.info(
            "开始风控审核：symbol=%s，direction=%s，setup_score=%.4f，spread=%.4f，used_risk_pct=%.4f",
            signal.symbol,
            signal.direction,
            agent_decision.setup_score,
            market_state.spread,
            portfolio_state.used_risk_pct,
        )

        if not agent_decision.allow_trade:
            reason = "风控拒绝：Agent 已明确拒绝该交易，不进入风险审批"
            self.logger.info(reason)
            return RiskDecision.reject(reject_reason=reason)

        if signal.direction == "neutral":
            reason = "风控拒绝：候选信号方向为 neutral"
            self.logger.info(reason)
            return RiskDecision.reject(reject_reason=reason)

        if market_state.spread > self.config.max_spread:
            reason = (
                f"风控拒绝：当前点差过大（spread={market_state.spread:.4f} > "
                f"{self.config.max_spread:.4f}）"
            )
            self.logger.info(reason)
            return RiskDecision.reject(reject_reason=reason, meta={"spread": market_state.spread})

        if portfolio_state.consecutive_losses >= self.config.max_consecutive_losses:
            reason = (
                f"风控拒绝：连续亏损次数过多（{portfolio_state.consecutive_losses} >= "
                f"{self.config.max_consecutive_losses}）"
            )
            self.logger.info(reason)
            return RiskDecision.reject(
                reject_reason=reason,
                meta={"consecutive_losses": portfolio_state.consecutive_losses},
            )

        if portfolio_state.daily_loss_pct >= self.config.max_daily_loss_pct:
            reason = (
                f"风控拒绝：日内亏损比例过高（{portfolio_state.daily_loss_pct:.4f} >= "
                f"{self.config.max_daily_loss_pct:.4f}）"
            )
            self.logger.info(reason)
            return RiskDecision.reject(
                reject_reason=reason,
                meta={"daily_loss_pct": portfolio_state.daily_loss_pct},
            )

        if portfolio_state.open_positions >= self.config.max_open_positions:
            reason = (
                f"风控拒绝：当前持仓数量过多（{portfolio_state.open_positions} >= "
                f"{self.config.max_open_positions}）"
            )
            self.logger.info(reason)
            return RiskDecision.reject(
                reject_reason=reason,
                meta={"open_positions": portfolio_state.open_positions},
            )

        if portfolio_state.used_risk_pct >= self.config.max_used_risk_pct:
            reason = (
                f"风控拒绝：风险占用过高（{portfolio_state.used_risk_pct:.4f} >= "
                f"{self.config.max_used_risk_pct:.4f}）"
            )
            self.logger.info(reason)
            return RiskDecision.reject(
                reject_reason=reason,
                meta={"used_risk_pct": portfolio_state.used_risk_pct},
            )

        if signal.rr_estimate < self.config.min_rr:
            reason = (
                f"风控拒绝：候选信号 RR 过低（{signal.rr_estimate:.4f} < "
                f"{self.config.min_rr:.4f}）"
            )
            self.logger.info(reason)
            return RiskDecision.reject(reject_reason=reason, meta={"rr_estimate": signal.rr_estimate})

        if agent_decision.setup_score < self.config.min_setup_score:
            reason = (
                f"风控拒绝：Agent 评分过低（{agent_decision.setup_score:.4f} < "
                f"{self.config.min_setup_score:.4f}）"
            )
            self.logger.info(reason)
            return RiskDecision.reject(
                reject_reason=reason,
                meta={"setup_score": agent_decision.setup_score},
            )

        if self.config.reject_direction_conflict:
            if agent_decision.direction_bias not in ("neutral", signal.direction):
                reason = (
                    f"风控拒绝：Agent 方向偏置（{agent_decision.direction_bias}）"
                    f"与候选信号方向（{signal.direction}）冲突"
                )
                self.logger.info(reason)
                return RiskDecision.reject(
                    reject_reason=reason,
                    meta={
                        "direction_bias": agent_decision.direction_bias,
                        "signal_direction": signal.direction,
                    },
                )

        adjusted_risk = self.config.default_risk_pct * max(agent_decision.risk_adjustment, 0.0)
        adjusted_risk = min(adjusted_risk, self.config.max_risk_pct_per_trade)

        remaining_risk_capacity = self.config.max_used_risk_pct - portfolio_state.used_risk_pct
        approved_risk = min(adjusted_risk, remaining_risk_capacity)

        if approved_risk <= 0:
            reason = "风控拒绝：剩余风险容量不足，无法分配单笔风险"
            self.logger.info(reason)
            return RiskDecision.reject(reject_reason=reason)

        self.logger.info(
            "风控通过：approved_risk=%.4f，default_risk_pct=%.4f，risk_adjustment=%.2f",
            approved_risk,
            self.config.default_risk_pct,
            agent_decision.risk_adjustment,
        )
        return RiskDecision.approve(
            risk_pct=approved_risk,
            meta={
                "default_risk_pct": self.config.default_risk_pct,
                "approved_risk": approved_risk,
                "risk_adjustment": agent_decision.risk_adjustment,
            },
        )
