from __future__ import annotations

from datetime import datetime

from dpaitrade.agent import PassthroughAgent, SimpleRuleFilterAgent
from dpaitrade.backtest import BacktestEngine
from dpaitrade.core import BacktestStep, PortfolioState
from dpaitrade.execution import SimpleExecutionSimulator
from dpaitrade.risk import GuardRiskManager, RiskGuardConfig
from dpaitrade.strategy import (
    D1Bar,
    D1RegimeDetector,
    H4Bar,
    H4PullbackDetector,
    M15Bar,
    M15EntryDetector,
    SignalBuildContext,
    SignalBuilder,
)


def build_demo_step() -> BacktestStep:
    """
    构造一个最小可运行的演示回测步。

    目标：
    - 不依赖外部数据文件
    - 先把整条工程链跑通
    - 便于后续替换为真实历史数据
    """
    print("[信息] 开始构造演示数据")

    d1_detector = D1RegimeDetector(lookback=20, trend_threshold=0.18, range_threshold=0.08)
    h4_detector = H4PullbackDetector(
        swing_lookback=24,
        pullback_min_ratio=0.25,
        pullback_max_ratio=0.70,
        boundary_near_ratio=0.15,
    )
    m15_detector = M15EntryDetector(structure_lookback=12, min_rr=1.5)
    signal_builder = SignalBuilder()

    # --------------------------
    # 一、构造 D1 趋势数据
    # --------------------------
    d1_bars = [
        D1Bar(open=100 + i * 0.8, high=101 + i * 0.8, low=99.8 + i * 0.8, close=100.6 + i * 0.8)
        for i in range(20)
    ]
    d1_result = d1_detector.detect(d1_bars)

    # --------------------------
    # 二、构造 H4 回调数据
    # --------------------------
    h4_bars = []
    for i in range(24):
        if i < 16:
            base = 100 + i * 1.2
        else:
            base = 119 - (i - 16) * 0.8
        h4_bars.append(H4Bar(open=base, high=base + 0.8, low=base - 0.8, close=base + 0.2))
    h4_result = h4_detector.detect(h4_bars, d1_regime=d1_result.regime, d1_bias=d1_result.bias)

    # --------------------------
    # 三、构造 M15 入场确认数据
    # --------------------------
    m15_bars = [
        M15Bar(open=110.0, high=110.5, low=109.8, close=110.2),
        M15Bar(open=110.2, high=110.6, low=110.0, close=110.4),
        M15Bar(open=110.4, high=110.7, low=110.1, close=110.3),
        M15Bar(open=110.3, high=110.8, low=110.0, close=110.5),
        M15Bar(open=110.5, high=110.9, low=110.2, close=110.6),
        M15Bar(open=110.6, high=111.0, low=110.3, close=110.7),
        M15Bar(open=110.7, high=111.1, low=110.4, close=110.8),
        M15Bar(open=110.8, high=111.2, low=110.5, close=110.9),
        M15Bar(open=110.9, high=111.25, low=110.6, close=111.0),
        M15Bar(open=111.0, high=111.3, low=110.7, close=111.05),
        M15Bar(open=111.05, high=111.35, low=110.8, close=111.1),
        M15Bar(open=111.1, high=111.8, low=111.0, close=111.7),
    ]
    m15_result = m15_detector.detect(m15_bars, preferred_direction=h4_result.preferred_direction)

    ctx = SignalBuildContext(
        ts=datetime(2026, 3, 6, 10, 0, 0),
        symbol="EURUSD",
        atr=1.2,
        spread=1.1,
        volatility_score=0.45,
    )

    market_state = signal_builder.build_market_state(ctx, d1_result, h4_result, m15_result)

    # 这里预先埋入未来价格，用于简单执行模拟器估算结果
    market_state.meta["future_high"] = 114.5
    market_state.meta["future_low"] = 111.1
    market_state.meta["future_close"] = 114.0

    candidate_signal = signal_builder.build_candidate_signal(ctx, d1_result, h4_result, m15_result)
    if candidate_signal is None:
        raise RuntimeError("演示数据未生成候选信号，请检查策略参数或样例数据")

    print("[信息] 演示数据构造完成，已生成候选信号")
    print(
        f"[信息] 候选信号：symbol={candidate_signal.symbol}，direction={candidate_signal.direction}，"
        f"entry={candidate_signal.entry_price:.5f}，sl={candidate_signal.stop_loss:.5f}，"
        f"tp={candidate_signal.take_profit:.5f}，rr={candidate_signal.rr_estimate:.4f}"
    )

    return BacktestStep(
        market_state=market_state,
        candidate_signal=candidate_signal,
        portfolio_state=PortfolioState(cash=10000.0, equity=10000.0),
    )


def run_demo_backtest(use_simple_rule_agent: bool = True) -> None:
    """
    运行最小演示回测。

    参数：
    - use_simple_rule_agent:
      True  -> 使用简单规则过滤代理
      False -> 使用直通代理
    """
    print("[信息] 开始运行演示回测")

    step = build_demo_step()

    if use_simple_rule_agent:
        agent = SimpleRuleFilterAgent(
            min_rr=1.2,
            max_spread=3.0,
            min_volatility_score=0.2,
            reject_when_direction_conflict=True,
        )
        print("[信息] 当前使用代理：简单规则过滤代理")
    else:
        agent = PassthroughAgent()
        print("[信息] 当前使用代理：直通代理")

    risk_manager = GuardRiskManager(
        config=RiskGuardConfig(
            default_risk_pct=0.005,
            max_risk_pct_per_trade=0.01,
            max_used_risk_pct=0.02,
            max_consecutive_losses=3,
            max_daily_loss_pct=0.02,
            max_open_positions=1,
            max_spread=2.0,
            min_setup_score=0.45,
            min_rr=1.2,
            reject_direction_conflict=True,
        )
    )
    simulator = SimpleExecutionSimulator()
    engine = BacktestEngine(agent=agent, risk_manager=risk_manager, execution_simulator=simulator)

    result = engine.run([step])

    print("\n========== 回测结果 ==========")
    print(f"总步数：{result.total_steps}")
    print(f"候选信号数：{result.total_candidate_signals}")
    print(f"Agent 拒绝数：{result.total_agent_rejected}")
    print(f"风控拒绝数：{result.total_risk_rejected}")
    print(f"成交笔数：{result.total_trades}")
    print(f"初始权益：{result.initial_equity:.2f}")
    print(f"最终权益：{result.final_equity:.2f}")
    print(f"净盈亏：{result.net_pnl:.2f}")
    print(f"胜率：{result.win_rate:.4f}")
    print(f"Profit Factor：{result.profit_factor}")
    print(f"最大回撤：{result.max_drawdown_pct:.4%}")

    if result.trade_records:
        print("\n========== 成交记录 ==========")
        for idx, trade in enumerate(result.trade_records, start=1):
            print(
                f"第{idx}笔：symbol={trade.symbol}，direction={trade.direction}，"
                f"entry={trade.entry_price:.5f}，exit={trade.exit_price:.5f}，"
                f"pnl={trade.pnl:.2f}，pnl_r={trade.pnl_r:.4f}，"
                f"open_reason={trade.open_reason}，close_reason={trade.close_reason}"
            )
    else:
        print("\n[信息] 本次回测没有成交记录")


if __name__ == "__main__":
    run_demo_backtest(use_simple_rule_agent=True)
