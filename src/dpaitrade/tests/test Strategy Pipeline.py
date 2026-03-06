from __future__ import annotations

from datetime import datetime

from dpaitrade.agent.interface import PassthroughAgent, SimpleRuleFilterAgent
from dpaitrade.backtest.engine import BacktestEngine
from dpaitrade.core.types import BacktestStep, PortfolioState
from dpaitrade.execution.simulator import SimpleExecutionSimulator
from dpaitrade.risk.guard import GuardRiskManager, RiskGuardConfig
from dpaitrade.strategy.d1_regime import D1Bar, D1RegimeDetector
from dpaitrade.strategy.h4_pullback import H4Bar, H4PullbackDetector
from dpaitrade.strategy.m15_entry import M15Bar, M15EntryDetector
from dpaitrade.strategy.signal_builder import SignalBuildContext, SignalBuilder


class TestStrategyPipeline:
    """
    策略流水线测试。

    当前目标不是验证“策略一定赚钱”，而是验证：
    1. 各模块是否能顺利串起来
    2. 是否能生成 MarketState 与 CandidateSignal
    3. Agent / 风控 / 执行模块是否能走完整链路
    4. 不同代理模式下结果是否符合预期
    """

    def test_full_pipeline_should_generate_trade_record_when_conditions_align(self) -> None:
        """
        当 D1 / H4 / M15 条件对齐，并且 future_close 对结果有利时，
        应当能生成至少一笔成交记录。
        """
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
        # 构造 D1 趋势数据：整体向上
        # --------------------------
        d1_bars = [
            D1Bar(open=100 + i * 0.8, high=101 + i * 0.8, low=99.8 + i * 0.8, close=100.6 + i * 0.8)
            for i in range(20)
        ]
        d1_result = d1_detector.detect(d1_bars)

        # --------------------------
        # 构造 H4 回调数据：
        # 整体位于上升区间内，但最近回落到中段偏下，符合“趋势中的回调”
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
        # 构造 M15 多头确认数据：
        # 最后一根收盘价突破近端高点
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
        market_state.meta["future_high"] = 114.5
        market_state.meta["future_low"] = 111.1
        market_state.meta["future_close"] = 114.0

        candidate_signal = signal_builder.build_candidate_signal(ctx, d1_result, h4_result, m15_result)

        assert d1_result.regime == "trend"
        assert d1_result.bias == "long"
        assert h4_result.pullback_active is True
        assert h4_result.preferred_direction == "long"
        assert m15_result.entry_ready is True
        assert candidate_signal is not None
        assert candidate_signal.direction == "long"

        step = BacktestStep(
            market_state=market_state,
            candidate_signal=candidate_signal,
            portfolio_state=PortfolioState(cash=10000.0, equity=10000.0),
        )

        agent = PassthroughAgent()
        risk_manager = GuardRiskManager(config=RiskGuardConfig(max_spread=2.0, min_rr=1.2))
        simulator = SimpleExecutionSimulator()
        engine = BacktestEngine(agent=agent, risk_manager=risk_manager, execution_simulator=simulator)

        result = engine.run([step])

        assert result.total_steps == 1
        assert result.total_candidate_signals == 1
        assert result.total_agent_rejected == 0
        assert result.total_risk_rejected == 0
        assert result.total_trades == 1
        assert len(result.trade_records) == 1
        assert result.trade_records[0].pnl != 0

    def test_rule_filter_agent_should_reject_when_spread_is_too_high(self) -> None:
        """
        当点差过大时，简单规则过滤代理应当拒绝放行。
        """
        d1_detector = D1RegimeDetector(lookback=20, trend_threshold=0.18, range_threshold=0.08)
        h4_detector = H4PullbackDetector(swing_lookback=24)
        m15_detector = M15EntryDetector(structure_lookback=12, min_rr=1.5)
        signal_builder = SignalBuilder()

        d1_bars = [
            D1Bar(open=100 + i * 0.6, high=101 + i * 0.6, low=99.7 + i * 0.6, close=100.5 + i * 0.6)
            for i in range(20)
        ]
        d1_result = d1_detector.detect(d1_bars)

        h4_bars = []
        for i in range(24):
            if i < 16:
                base = 100 + i * 1.0
            else:
                base = 116 - (i - 16) * 0.7
            h4_bars.append(H4Bar(open=base, high=base + 0.7, low=base - 0.7, close=base + 0.2))
        h4_result = h4_detector.detect(h4_bars, d1_regime=d1_result.regime, d1_bias=d1_result.bias)

        m15_bars = [
            M15Bar(open=110.0, high=110.5, low=109.8, close=110.1),
            M15Bar(open=110.1, high=110.6, low=109.9, close=110.2),
            M15Bar(open=110.2, high=110.7, low=110.0, close=110.3),
            M15Bar(open=110.3, high=110.8, low=110.1, close=110.4),
            M15Bar(open=110.4, high=110.9, low=110.2, close=110.5),
            M15Bar(open=110.5, high=111.0, low=110.3, close=110.6),
            M15Bar(open=110.6, high=111.1, low=110.4, close=110.7),
            M15Bar(open=110.7, high=111.2, low=110.5, close=110.8),
            M15Bar(open=110.8, high=111.3, low=110.6, close=110.9),
            M15Bar(open=110.9, high=111.35, low=110.7, close=111.0),
            M15Bar(open=111.0, high=111.4, low=110.8, close=111.1),
            M15Bar(open=111.1, high=111.9, low=111.0, close=111.8),
        ]
        m15_result = m15_detector.detect(m15_bars, preferred_direction=h4_result.preferred_direction)

        ctx = SignalBuildContext(
            ts=datetime(2026, 3, 6, 11, 0, 0),
            symbol="GBPUSD",
            atr=1.1,
            spread=6.0,
            volatility_score=0.5,
        )

        market_state = signal_builder.build_market_state(ctx, d1_result, h4_result, m15_result)
        candidate_signal = signal_builder.build_candidate_signal(ctx, d1_result, h4_result, m15_result)

        assert candidate_signal is not None

        agent = SimpleRuleFilterAgent(max_spread=3.0)
        decision = agent.evaluate(
            market_state=market_state,
            signal=candidate_signal,
            portfolio_state=PortfolioState(cash=10000.0, equity=10000.0),
        )

        assert decision.allow_trade is False
        assert "点差过大" in decision.reason

    def test_signal_builder_should_return_none_when_cross_cycle_conditions_do_not_align(self) -> None:
        """
        当跨周期方向不一致时，不应生成候选信号。
        """
        signal_builder = SignalBuilder()

        # 这里直接构造一个“D1 偏多，但 M15 给出空头方向”的场景
        d1_result = D1RegimeDetector(lookback=20, trend_threshold=0.18, range_threshold=0.08).detect(
            [
                D1Bar(open=100 + i * 0.7, high=101 + i * 0.7, low=99.8 + i * 0.7, close=100.6 + i * 0.7)
                for i in range(20)
            ]
        )

        h4_result = H4PullbackDetector(swing_lookback=24).detect(
            [
                H4Bar(
                    open=(100 + i * 1.1) if i < 16 else (118 - (i - 16) * 0.8),
                    high=((100 + i * 1.1) if i < 16 else (118 - (i - 16) * 0.8)) + 0.7,
                    low=((100 + i * 1.1) if i < 16 else (118 - (i - 16) * 0.8)) - 0.7,
                    close=((100 + i * 1.1) if i < 16 else (118 - (i - 16) * 0.8)) + 0.2,
                )
                for i in range(24)
            ],
            d1_regime=d1_result.regime,
            d1_bias=d1_result.bias,
        )

        # 故意指定 preferred_direction=short，制造与 D1 偏多冲突的场景
        m15_result = M15EntryDetector(structure_lookback=12, min_rr=1.5).detect(
            [
                M15Bar(open=111.8, high=112.0, low=111.6, close=111.7),
                M15Bar(open=111.7, high=111.9, low=111.5, close=111.6),
                M15Bar(open=111.6, high=111.8, low=111.4, close=111.5),
                M15Bar(open=111.5, high=111.7, low=111.3, close=111.4),
                M15Bar(open=111.4, high=111.6, low=111.2, close=111.3),
                M15Bar(open=111.3, high=111.5, low=111.1, close=111.2),
                M15Bar(open=111.2, high=111.4, low=111.0, close=111.1),
                M15Bar(open=111.1, high=111.3, low=110.9, close=111.0),
                M15Bar(open=111.0, high=111.2, low=110.8, close=110.9),
                M15Bar(open=110.9, high=111.1, low=110.7, close=110.8),
                M15Bar(open=110.8, high=111.0, low=110.6, close=110.7),
                M15Bar(open=110.7, high=110.8, low=110.1, close=110.2),
            ],
            preferred_direction="short",
        )

        ctx = SignalBuildContext(
            ts=datetime(2026, 3, 6, 12, 0, 0),
            symbol="USDJPY",
            atr=1.0,
            spread=1.0,
            volatility_score=0.4,
        )

        candidate_signal = signal_builder.build_candidate_signal(ctx, d1_result, h4_result, m15_result)
        assert d1_result.bias == "long"
        assert m15_result.direction == "short"
        assert candidate_signal is None
