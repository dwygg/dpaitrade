from __future__ import annotations

from abc import ABC, abstractmethod

from dpaitrade.core.types import AgentDecision, CandidateSignal, MarketState, PortfolioState


class BaseAgent(ABC):
    """
    Agent 抽象基类。

    当前项目约定：
    Agent 只负责“受限决策”，不直接负责成交执行。
    """

    name: str = "基础代理"

    @abstractmethod
    def evaluate(
        self,
        market_state: MarketState,
        signal: CandidateSignal,
        portfolio_state: PortfolioState,
    ) -> AgentDecision:
        """
        对候选信号进行评估，并输出结构化决策。

        参数：
        - market_state: 当前市场状态
        - signal: 当前候选信号
        - portfolio_state: 当前账户状态

        返回：
        - AgentDecision
        """
        raise NotImplementedError


class PassthroughAgent(BaseAgent):
    """
    直通代理。

    用途：
    - 调试回测主链路
    - 作为“无 AI 过滤”的最小代理
    - 后续可作为纯规则基线对照
    """

    name = "直通代理"

    def evaluate(
        self,
        market_state: MarketState,
        signal: CandidateSignal,
        portfolio_state: PortfolioState,
    ) -> AgentDecision:
        return AgentDecision.allow(
            direction_bias=signal.direction,
            setup_score=0.5,
            risk_adjustment=1.0,
            reason="直通代理未做过滤，默认放行候选信号",
            meta={
                "agent_name": self.name,
                "d1_regime": market_state.d1_regime,
                "d1_bias": market_state.d1_bias,
            },
        )


class SimpleRuleFilterAgent(BaseAgent):
    """
    简单规则过滤代理。

    这个类不是最终 AI，而是一个“占位代理”。
    在正式接入 LLM 之前，可以先用它模拟：
    - 评分
    - 拒绝交易
    - 风险缩放

    这样可以先把：
    规则信号 -> Agent 过滤 -> 风控 -> 回测
    这一整条链路跑通。
    """

    name = "简单规则过滤代理"

    def __init__(
        self,
        min_rr: float = 1.5,
        max_spread: float = 3.0,
        min_volatility_score: float = 0.2,
        reject_when_direction_conflict: bool = True,
    ) -> None:
        self.min_rr = min_rr
        self.max_spread = max_spread
        self.min_volatility_score = min_volatility_score
        self.reject_when_direction_conflict = reject_when_direction_conflict

    def evaluate(
        self,
        market_state: MarketState,
        signal: CandidateSignal,
        portfolio_state: PortfolioState,
    ) -> AgentDecision:
        """
        按简单规则对候选信号评分。

        当前评分逻辑较朴素，主要用于占位：
        - RR 越高，加分越多
        - 点差越低，加分越多
        - 波动率过低时减分
        - 若 D1 偏置与信号方向冲突，可直接拒绝
        """

        if signal.direction == "neutral":
            return AgentDecision.reject(
                reason="候选信号方向为中性，代理拒绝放行",
                meta={"agent_name": self.name},
            )

        if market_state.spread > self.max_spread:
            return AgentDecision.reject(
                reason=f"当前点差过大（spread={market_state.spread:.4f}），代理拒绝交易",
                meta={
                    "agent_name": self.name,
                    "spread": market_state.spread,
                    "max_spread": self.max_spread,
                },
            )

        if self.reject_when_direction_conflict:
            if market_state.d1_bias != "neutral" and market_state.d1_bias != signal.direction:
                return AgentDecision.reject(
                    reason=(
                        f"D1 方向偏置（{market_state.d1_bias}）"
                        f"与候选信号方向（{signal.direction}）冲突，代理拒绝交易"
                    ),
                    meta={
                        "agent_name": self.name,
                        "d1_bias": market_state.d1_bias,
                        "signal_direction": signal.direction,
                    },
                )

        score = 0.0

        # RR 评分
        if signal.rr_estimate >= self.min_rr:
            score += 0.4
        else:
            score += max(0.0, signal.rr_estimate / max(self.min_rr, 1e-6)) * 0.4

        # 点差评分
        spread_score = max(0.0, 1.0 - (market_state.spread / max(self.max_spread, 1e-6)))
        score += spread_score * 0.3

        # 波动率评分
        vol_score = min(1.0, market_state.volatility_score / max(self.min_volatility_score, 1e-6))
        score += vol_score * 0.3

        score = min(1.0, round(score, 4))

        if score < 0.45:
            return AgentDecision.reject(
                reason=f"候选信号综合评分过低（score={score:.4f}），代理拒绝交易",
                setup_score=score,
                direction_bias=signal.direction,
                meta={"agent_name": self.name},
            )

        # 简单风险缩放逻辑：评分越高，允许风险略大
        risk_adjustment = 0.8
        if score >= 0.8:
            risk_adjustment = 1.2
        elif score >= 0.6:
            risk_adjustment = 1.0

        return AgentDecision.allow(
            direction_bias=signal.direction,
            setup_score=score,
            risk_adjustment=risk_adjustment,
            reason=f"候选信号通过简单规则过滤，综合评分={score:.4f}",
            meta={
                "agent_name": self.name,
                "spread": market_state.spread,
                "rr_estimate": signal.rr_estimate,
                "volatility_score": market_state.volatility_score,
            },
        )
