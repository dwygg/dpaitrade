from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from typing import Iterable, Optional, Protocol

from dpaitrade.agent.interface import BaseAgent
from dpaitrade.core.types import (
    AgentDecision,
    BacktestResult,
    BacktestStep,
    PortfolioState,
    RiskDecision,
    TradeRecord,
)


class RiskManager(Protocol):
    """
    风控器协议。

    后续真正的风控模块只要实现 review 方法即可接入。
    """

    def review(
        self,
        market_state,
        signal,
        agent_decision: AgentDecision,
        portfolio_state: PortfolioState,
    ) -> RiskDecision:
        """
        审核当前候选信号是否允许进入执行阶段。
        """
        ...


class ExecutionSimulator(Protocol):
    """
    执行模拟器协议。

    后续真正的成交仿真模块只要实现 execute 方法即可接入。
    """

    def execute(
        self,
        market_state,
        signal,
        risk_decision: RiskDecision,
        portfolio_state: PortfolioState,
    ) -> Optional[TradeRecord]:
        """
        执行模拟交易。

        返回：
        - TradeRecord: 有成交且已形成记录
        - None: 本步未发生可记录的交易
        """
        ...


class DefaultRiskManager:
    """
    默认风控器。

    这不是最终版本，只是一个最小可用占位实现，
    用于先打通主流程。
    """

    def __init__(
        self,
        default_risk_pct: float = 0.005,
        max_used_risk_pct: float = 0.02,
        max_consecutive_losses: int = 3,
        max_spread: float = 200.0,
    ) -> None:
        self.default_risk_pct = default_risk_pct
        self.max_used_risk_pct = max_used_risk_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_spread = max_spread

    def review(
        self,
        market_state,
        signal,
        agent_decision: AgentDecision,
        portfolio_state: PortfolioState,
    ) -> RiskDecision:
        """
        默认风控审批逻辑：

        - 点差过大，拒绝
        - 连续亏损过多，拒绝
        - 当前风险占用过高，拒绝
        - 否则按 agent 的 risk_adjustment 计算单笔风险
        """

        if market_state.spread > self.max_spread:
            return RiskDecision.reject(
                reject_reason=(
                    f"风控拒绝：当前点差过大（spread={market_state.spread:.4f} > "
                    f"{self.max_spread:.4f}）"
                ),
                meta={"spread": market_state.spread},
            )

        if portfolio_state.consecutive_losses >= self.max_consecutive_losses:
            return RiskDecision.reject(
                reject_reason=(
                    f"风控拒绝：连续亏损次数过多（{portfolio_state.consecutive_losses}）"
                ),
                meta={"consecutive_losses": portfolio_state.consecutive_losses},
            )

        if portfolio_state.used_risk_pct >= self.max_used_risk_pct:
            return RiskDecision.reject(
                reject_reason=(
                    f"风控拒绝：当前风险占用过高（{portfolio_state.used_risk_pct:.4f}）"
                ),
                meta={"used_risk_pct": portfolio_state.used_risk_pct},
            )

        adjusted_risk = self.default_risk_pct * max(agent_decision.risk_adjustment, 0.0)
        adjusted_risk = min(adjusted_risk, self.max_used_risk_pct)

        if adjusted_risk <= 0:
            return RiskDecision.reject(
                reject_reason="风控拒绝：计算后的单笔风险小于等于 0",
                meta={"adjusted_risk": adjusted_risk},
            )

        return RiskDecision.approve(
            risk_pct=adjusted_risk,
            meta={
                "default_risk_pct": self.default_risk_pct,
                "agent_risk_adjustment": agent_decision.risk_adjustment,
            },
        )


class NoopExecutionSimulator:
    """
    空执行模拟器。

    当前仅用于先把主链路打通，不产生真实交易记录。
    后续可替换为真正的成交仿真模块。
    """

    def execute(
        self,
        market_state,
        signal,
        risk_decision: RiskDecision,
        portfolio_state: PortfolioState,
    ) -> Optional[TradeRecord]:
        return None


class BacktestEngine:
    """
    回测主引擎。

    当前职责：
    1. 遍历回测步
    2. 调用 Agent 评估
    3. 调用风控审批
    4. 调用执行模拟器
    5. 汇总统计结果

    注意：
    当前版本重点在于打通“主链路”，
    不追求一次性实现完整成交仿真细节。
    """

    def __init__(
        self,
        agent: BaseAgent,
        risk_manager: Optional[RiskManager] = None,
        execution_simulator: Optional[ExecutionSimulator] = None,
        logger: Optional[logging.Logger] = None,
        cooldown_steps: int = 0,
    ) -> None:
        self.agent = agent
        self.risk_manager = risk_manager or DefaultRiskManager()
        self.execution_simulator = execution_simulator or NoopExecutionSimulator()
        self.logger = logger or self._build_logger()
        self.cooldown_steps = max(0, cooldown_steps)

    @staticmethod
    def _build_logger() -> logging.Logger:
        """
        创建默认日志对象。

        日志打印统一使用中文，方便后续排查。
        """
        logger = logging.getLogger("dpaitrade.backtest")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)  # DEBUG 级别由此过滤，仅输出关键事件

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    def run(
        self,
        steps: Iterable[BacktestStep],
        initial_portfolio: Optional[PortfolioState] = None,
    ) -> BacktestResult:
        """
        执行回测。

        参数：
        - steps: 回测步序列
        - initial_portfolio: 初始账户状态

        返回：
        - BacktestResult
        """
        initial_portfolio = initial_portfolio or PortfolioState(
            cash=10000.0,
            equity=10000.0,
        )

        portfolio_state = replace(initial_portfolio)

        started_at = datetime.now()
        result = BacktestResult(
            started_at=started_at,
            finished_at=started_at,
            initial_equity=portfolio_state.equity,
            final_equity=portfolio_state.equity,
        )

        busy_until: Optional[datetime] = None
        cooldown_remaining = 0
        self.logger.info("开始执行回测，初始资金：%.2f", portfolio_state.equity)
        self.logger.info("当前 Agent：%s", self.agent.name)

        for idx, step in enumerate(steps, start=1):
            result.total_steps += 1

            market_state = step.market_state
            signal = step.candidate_signal
            step_portfolio_state = step.portfolio_state or portfolio_state

            self.logger.debug(
                "处理回测步[%s]，时间=%s，品种=%s，D1状态=%s，D1偏置=%s",
                idx,
                market_state.ts,
                market_state.symbol,
                market_state.d1_regime,
                market_state.d1_bias,
            )

            self._update_swing_trackers(portfolio_state=portfolio_state, market_state=market_state)

            if busy_until is not None and market_state.ts < busy_until:
                self.logger.debug("持仓中，跳过（busy_until=%s）", busy_until)
                continue

            if cooldown_remaining > 0:
                self.logger.debug("冷却期剩余=%s，跳过", cooldown_remaining)
                cooldown_remaining -= 1
                continue
            if signal is None:
                self.logger.debug("无候选信号，跳过")
                continue

            result.total_candidate_signals += 1
            self.logger.info(
                "候选信号 [%s %s] entry=%.2f sl=%.2f RR=%.2f",
                signal.direction,
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                signal.rr_estimate,
            )

            swing_reject_reason = self._swing_reuse_reject_reason(
                portfolio_state=portfolio_state,
                signal=signal,
            )
            if swing_reject_reason:
                result.total_risk_rejected += 1
                self.logger.info("摆点复用拒绝：%s", swing_reject_reason)
                continue

            # --------------------------
            # 第一步：Agent 过滤与评分
            # --------------------------
            agent_decision = self.agent.evaluate(
                market_state=market_state,
                signal=signal,
                portfolio_state=step_portfolio_state,
            )

            self.logger.debug(
                "Agent 决策：allow_trade=%s，setup_score=%.4f，原因=%s",
                agent_decision.allow_trade,
                agent_decision.setup_score,
                agent_decision.reason,
            )

            if not agent_decision.allow_trade:
                result.total_agent_rejected += 1
                self.logger.info("Agent 拒绝 [%s %s]：%s", signal.direction, signal.symbol, agent_decision.reason)
                continue

            # --------------------------
            # 第二步：风控审批
            # --------------------------
            risk_decision = self.risk_manager.review(
                market_state=market_state,
                signal=signal,
                agent_decision=agent_decision,
                portfolio_state=step_portfolio_state,
            )

            if not risk_decision.approved:
                result.total_risk_rejected += 1
                self.logger.info("风控拒绝 [%s %s]：%s", signal.direction, signal.symbol, risk_decision.reject_reason)
                continue

            # --------------------------
            # 第三步：执行模拟
            # --------------------------
            trade_record = self.execution_simulator.execute(
                market_state=market_state,
                signal=signal,
                risk_decision=risk_decision,
                portfolio_state=step_portfolio_state,
            )

            if trade_record is None:
                self.logger.info("本步未形成成交记录，进入下一步")
                continue

            result.trade_records.append(trade_record)
            busy_until = trade_record.close_time
            result.total_trades += 1

            self._apply_trade_record(
                portfolio_state=portfolio_state,
                trade_record=trade_record,
                risk_pct=risk_decision.risk_pct,
            )
            self._update_swing_tracker_after_trade(
                portfolio_state=portfolio_state,
                trade_record=trade_record,
            )
            if self.cooldown_steps > 0:
                cooldown_remaining = self.cooldown_steps
                self.logger.info("成交后进入冷却期，冷却步数=%s", self.cooldown_steps)
                
            self.logger.info(
                "成交 [%s %s] entry=%.2f exit=%.2f 原因=%s pnl=%.2f (%.2fR) 权益=%.2f",
                trade_record.direction,
                trade_record.symbol,
                trade_record.entry_price,
                trade_record.exit_price,
                trade_record.close_reason,
                trade_record.pnl,
                trade_record.pnl_r,
                portfolio_state.equity,
            )

        # 回测结束，计算汇总指标
        result.finished_at = datetime.now()
        result.final_equity = portfolio_state.equity
        result.net_pnl = result.final_equity - result.initial_equity

        self._finalize_metrics(result=result, initial_equity=result.initial_equity)

        self.logger.info("回测结束")
        self.logger.info(
            "结果汇总：总步数=%s，候选信号=%s，Agent拒绝=%s，风控拒绝=%s，成交笔数=%s，净盈亏=%.2f",
            result.total_steps,
            result.total_candidate_signals,
            result.total_agent_rejected,
            result.total_risk_rejected,
            result.total_trades,
            result.net_pnl,
        )

        return result


    def _update_swing_trackers(
        self,
        portfolio_state: PortfolioState,
        market_state,
    ) -> None:
        trackers = portfolio_state.meta.setdefault("swing_trackers", {})
        current_price = float(market_state.meta.get("current_price", 0.0) or 0.0)
        if current_price <= 0:
            return
        for tracker in trackers.values():
            if not tracker.get("needs_reset"):
                continue
            zone_lower = tracker.get("zone_lower")
            zone_upper = tracker.get("zone_upper")
            if zone_lower is None or zone_upper is None:
                continue
            if current_price < float(zone_lower) or current_price > float(zone_upper):
                tracker["zone_released"] = True

    def _swing_reuse_reject_reason(
        self,
        portfolio_state: PortfolioState,
        signal,
    ) -> Optional[str]:
        if str(signal.meta.get("policy", "")).startswith("swing_point") is False:
            return None
        swing_id = signal.meta.get("swing_id")
        if not swing_id:
            return None
        trackers = portfolio_state.meta.setdefault("swing_trackers", {})
        tracker = trackers.get(swing_id)
        if not tracker:
            return None
        if tracker.get("invalidated"):
            return f"摆点复用规则拒绝：{swing_id} 已失效"
        max_attempts = int(signal.meta.get("max_attempts_per_swing") or 0)
        if max_attempts > 0 and int(tracker.get("attempts", 0)) >= max_attempts:
            return f"摆点复用规则拒绝：{swing_id} 尝试次数已达上限 {max_attempts}"
        max_cum_loss = float(signal.meta.get("max_cumulative_loss_r_per_swing") or 0.0)
        cum_loss_r = float(tracker.get("cumulative_loss_r", 0.0) or 0.0)
        if max_cum_loss > 0 and cum_loss_r >= max_cum_loss:
            return f"摆点复用规则拒绝：{swing_id} 累计亏损已达 {cum_loss_r:.2f}R"
        require_reset = bool(signal.meta.get("require_reset_after_loss", False))
        if require_reset and tracker.get("needs_reset") and not tracker.get("zone_released"):
            return f"摆点复用规则拒绝：{swing_id} 上次亏损后尚未完成 zone reset"
        return None

    def _update_swing_tracker_after_trade(
        self,
        portfolio_state: PortfolioState,
        trade_record: TradeRecord,
    ) -> None:
        swing_id = trade_record.meta.get("swing_id")
        if not swing_id:
            return
        trackers = portfolio_state.meta.setdefault("swing_trackers", {})
        tracker = trackers.setdefault(swing_id, {
            "attempts": 0,
            "cumulative_loss_r": 0.0,
            "needs_reset": False,
            "zone_released": False,
            "invalidated": False,
        })
        tracker["attempts"] = int(tracker.get("attempts", 0)) + 1
        tracker["zone_lower"] = trade_record.meta.get("zone_lower")
        tracker["zone_upper"] = trade_record.meta.get("zone_upper")
        tracker["last_close_reason"] = trade_record.close_reason
        tracker["last_close_time"] = trade_record.close_time
        if trade_record.close_reason == "摆点失效离场":
            tracker["invalidated"] = True
        if trade_record.pnl < 0:
            tracker["cumulative_loss_r"] = float(tracker.get("cumulative_loss_r", 0.0)) + abs(float(trade_record.pnl_r))
            tracker["needs_reset"] = True
            tracker["zone_released"] = False
        else:
            tracker["needs_reset"] = False
            tracker["zone_released"] = False

    def _apply_trade_record(
        self,
        portfolio_state: PortfolioState,
        trade_record: TradeRecord,
        risk_pct: float,
    ) -> None:
        """
        将成交记录反映到账户状态中。

        当前先做最基础更新：
        - 资金
        - 权益
        - 连续亏损次数
        - 当前风险占用（简化处理）

        后续可以扩展：
        - 已开仓列表
        - 日内亏损限制
        - 最大回撤跟踪
        """
        portfolio_state.cash += trade_record.pnl
        portfolio_state.equity += trade_record.pnl
        portfolio_state.used_risk_pct = max(0.0, portfolio_state.used_risk_pct - risk_pct)

        if trade_record.pnl < 0:
            portfolio_state.consecutive_losses += 1
        else:
            portfolio_state.consecutive_losses = 0

    def _finalize_metrics(self, result: BacktestResult, initial_equity: float) -> None:
        """
        计算回测汇总指标。
        """
        trades = result.trade_records
        if not trades:
            result.win_rate = 0.0
            result.profit_factor = 0.0
            result.max_drawdown_pct = 0.0
            return

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]

        result.win_rate = len(wins) / len(trades)

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        if gross_loss == 0:
            result.profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            result.profit_factor = gross_profit / gross_loss

        result.max_drawdown_pct = self._calculate_max_drawdown_pct(
            initial_equity=initial_equity,
            trades=trades,
        )

    def _calculate_max_drawdown_pct(
        self,
        initial_equity: float,
        trades: list[TradeRecord],
    ) -> float:
        """
        根据成交记录计算最大回撤百分比。

        当前为基于已实现净盈亏的简化计算。
        """
        peak = initial_equity
        equity = initial_equity
        max_drawdown = 0.0

        for trade in trades:
            equity += trade.pnl
            if equity > peak:
                peak = equity

            if peak <= 0:
                continue

            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown
