from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from dpaitrade.agent import PassthroughAgent, SimpleRuleFilterAgent
from dpaitrade.backtest import BacktestEngine
from dpaitrade.core import BacktestStep, PortfolioState
from dpaitrade.data import OHLCRow, load_ohlc_csv, slice_recent_rows
from dpaitrade.execution import SimpleExecutionSimulator
from dpaitrade.risk import GuardRiskManager, RiskGuardConfig
from dpaitrade.strategy import (
    StrategyContext,
    TrendContinuationPolicy,
    TrendContinuationPolicyConfig,
)
from dpaitrade.structure import GenericBar, StructureAnalyzer, StructureAnalyzerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dpaitrade 统一结构策略回测入口")

    parser.add_argument("--d1", type=str, required=True, help="D1 CSV 路径")
    parser.add_argument("--h4", type=str, required=True, help="H4 CSV 路径")
    parser.add_argument("--m15", type=str, required=True, help="M15 CSV 路径")

    parser.add_argument("--symbol", type=str, default="UNKNOWN", help="品种名称")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["simple", "passthrough"],
        default="simple",
        help="选择代理类型",
    )
    parser.add_argument("--initial-equity", type=float, default=10000.0, help="初始权益")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个 M15 步，0 表示不限制")
    parser.add_argument(
        "--future-window",
        type=int,
        default=64,
        help="执行模拟时，用后续多少根 M15 生成 future_high/future_low/future_close",
    )

    parser.add_argument("--d1-lookback", type=int, default=24, help="D1 回看窗口")
    parser.add_argument("--h4-lookback", type=int, default=24, help="H4 回看窗口")
    parser.add_argument("--m15-lookback", type=int, default=24, help="M15 回看窗口")

    parser.add_argument("--cooldown-steps", type=int, default=4, help="成交后冷却步数")

    parser.add_argument("--allow-long", action="store_true", help="允许做多")
    parser.add_argument("--allow-short", action="store_true", help="允许做空")
    return parser.parse_args()


def to_generic_bars(rows: list[OHLCRow]) -> list[GenericBar]:
    return [
        GenericBar(
            ts=r.ts,
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
        )
        for r in rows
    ]


def estimate_atr_from_rows(rows: list[OHLCRow], period: int = 14) -> float:
    """
    当 CSV 没有 atr 列时，用最近 period 根 M15 估算一个 ATR，
    避免策略中的 ATR 距离保护退化成接近 0。
    """
    if len(rows) < 2:
        return 0.0

    window = rows[-max(period + 1, 2) :]
    true_ranges: list[float] = []
    for prev_row, row in zip(window[:-1], window[1:]):
        tr = max(
            row.high - row.low,
            abs(row.high - prev_row.close),
            abs(row.low - prev_row.close),
        )
        true_ranges.append(tr)
    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


def build_future_path(
    m15_rows: list[OHLCRow],
    current_index: int,
    future_window: int,
) -> tuple[list[dict], float | None, float | None, float | None]:
    start = current_index + 1
    end = min(len(m15_rows), current_index + 1 + future_window)
    future_rows = m15_rows[start:end]

    if not future_rows:
        return [], None, None, None

    future_bars = [
        {
            "ts": row.ts,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
        }
        for row in future_rows
    ]

    future_high = max(row.high for row in future_rows)
    future_low = min(row.low for row in future_rows)
    future_close = future_rows[-1].close
    return future_bars, future_high, future_low, future_close

def build_backtest_steps_from_csv(
    d1_rows: list[OHLCRow],
    h4_rows: list[OHLCRow],
    m15_rows: list[OHLCRow],
    symbol: str,
    d1_lookback: int,
    h4_lookback: int,
    m15_lookback: int,
    future_window: int,
    allow_long: bool,
    allow_short: bool,
    limit: int = 0,
) -> list[BacktestStep]:
    print("[信息] 开始构造多步回测数据")

    d1_analyzer = StructureAnalyzer(
        StructureAnalyzerConfig(
            timeframe="D1",
            min_bars=d1_lookback,
            lookback=d1_lookback,
            swing_window=2,
            trend_threshold=0.38,
            range_threshold=0.58,
        )
    )
    h4_analyzer = StructureAnalyzer(
        StructureAnalyzerConfig(
            timeframe="H4",
            min_bars=h4_lookback,
            lookback=h4_lookback,
            swing_window=2,
            trend_threshold=0.36,
            range_threshold=0.60,
            near_boundary_atr_ratio=0.60,
        )
    )
    m15_analyzer = StructureAnalyzer(
        StructureAnalyzerConfig(
            timeframe="M15",
            min_bars=m15_lookback,
            lookback=m15_lookback,
            swing_window=2,
            trend_threshold=0.34,
            range_threshold=0.60,
        )
    )

    policy = TrendContinuationPolicy(
        TrendContinuationPolicyConfig(
            allow_long=allow_long,
            allow_short=allow_short,
            low_tf_range_entry_guard_atr_ratio=1.20,
        )
    )

    d1_ts = [r.ts for r in d1_rows]
    h4_ts = [r.ts for r in h4_rows]
    m15_ts = [r.ts for r in m15_rows]

    steps: list[BacktestStep] = []
    processed = 0

    # ── 诊断计数器 ──────────────────────────────────────────────
    from collections import Counter
    diag: Counter = Counter()
    # ────────────────────────────────────────────────────────────

    for idx, current_m15 in enumerate(m15_rows):
        current_ts = current_m15.ts

        d1_window_rows = slice_recent_rows(d1_rows, d1_ts, current_ts, d1_lookback)
        h4_window_rows = slice_recent_rows(h4_rows, h4_ts, current_ts, h4_lookback)
        m15_window_rows = slice_recent_rows(m15_rows, m15_ts, current_ts, m15_lookback)

        if len(d1_window_rows) < d1_lookback:
            continue
        if len(h4_window_rows) < h4_lookback:
            continue
        if len(m15_window_rows) < m15_lookback:
            continue

        atr = float(current_m15.meta.get("atr", 0.0) or 0.0)
        if atr <= 0.0:
            atr = estimate_atr_from_rows(m15_window_rows)

        spread = float(current_m15.meta.get("spread", 0.0) or 0.0)
        volatility_score = float(current_m15.meta.get("volatility_score", 0.0) or 0.0)

        d1_state = d1_analyzer.analyze(to_generic_bars(d1_window_rows))
        h4_state = h4_analyzer.analyze(to_generic_bars(h4_window_rows))
        m15_state = m15_analyzer.analyze(to_generic_bars(m15_window_rows))

        # ── 诊断：D1多头路径逐级过滤 ────────────────────────────
        diag[f"D1 bias={d1_state.primary_bias}"] += 1
        if d1_state.primary_bias == "long":
            # 过滤1: reversal_candidate
            if d1_state.phase == "reversal_candidate":
                diag["LONG过滤@D1 phase=reversal_candidate"] += 1
            # 过滤2: trend_score
            elif d1_state.trend_score < policy.config.min_trend_score_high_tf:
                diag[f"LONG过滤@D1 trend_score不足(score={d1_state.trend_score:.2f})"] += 1
            else:
                # D1 通过
                # 过滤3: H4 in_value_zone
                if not h4_state.in_value_zone:
                    diag["LONG过滤@H4 not in_value_zone"] += 1
                # 过滤4: H4 reversal_warning
                elif h4_state.reversal_warning:
                    diag["LONG过滤@H4 reversal_warning"] += 1
                else:
                    # H4 通过，检查 constraint_lower 距离
                    _atr_tmp = float(current_m15.meta.get("atr", 0.0) or 0.0)
                    _mid_price = h4_window_rows[-1].close if h4_window_rows else current_m15.close
                    _max_dist = max(_atr_tmp * policy.config.low_tf_range_entry_guard_atr_ratio, 1e-8)
                    if h4_state.constraint_lower is not None:
                        _dist = max(_mid_price - h4_state.constraint_lower, 0.0)
                        if _dist > _max_dist:
                            diag[f"LONG过滤@H4距下缘过远(dist={_dist:.1f} max={_max_dist:.1f})"] += 1
                            diag["LONG过滤@H4距下缘过远[汇总]"] += 1
                        else:
                            diag["LONG通过H4距离检查→进入M15触发器"] += 1
                    else:
                        diag["LONG通过H4距离检查(无constraint_lower)→进入M15触发器"] += 1
        # ────────────────────────────────────────────────────────

        ctx = StrategyContext(
            ts=current_ts,
            symbol=symbol,
            entry_price=current_m15.close,
            atr=atr,
            spread=spread,
            volatility_score=volatility_score,
            mid_price=h4_window_rows[-1].close if h4_window_rows else current_m15.close,
            low_tf_bars=to_generic_bars(m15_window_rows),
        )

        candidate_signal = policy.generate_signal(
            high_tf=d1_state,
            mid_tf=h4_state,
            low_tf=m15_state,
            ctx=ctx,
        )

        future_bars, future_high, future_low, future_close = build_future_path(
            m15_rows=m15_rows,
            current_index=idx,
            future_window=future_window,
        )

        h4_exec_bars = build_h4_execution_path(
                h4_rows=h4_rows,
                current_ts=current_ts,
                history_window=h4_lookback,
                forward_window=max(8, future_window // 4 + 2),
        )
        # 使用 D1 作为主市场状态，中周期/低周期状态写入 meta，供后续分析使用
        market_state = _build_market_state_from_structure(
            symbol=symbol,
            state=d1_state,
            current_ts=current_ts,
            atr=atr,
            spread=spread,
            volatility_score=volatility_score,

            extra_meta={
                "h4_state": h4_state,
                "m15_state": m15_state,
                "h4_exec_bars": h4_exec_bars,
                "future_bars": future_bars,
                "future_high": future_high,
                "future_low": future_low,
                "future_close": future_close,
            },
        )

        step = BacktestStep(
            market_state=market_state,
            candidate_signal=candidate_signal,
            portfolio_state=None,
        )
        steps.append(step)

        processed += 1
        if limit > 0 and processed >= limit:
            break

    print(f"[信息] 多步回测数据构造完成，总步数={len(steps)}")

    # ── 诊断报告写入文件 ─────────────────────────────────────────
    import sys
    diag_path = "backtest_diag.txt"
    with open(diag_path, "w", encoding="utf-8") as _f:
        _f.write("========== [诊断] 信号过滤分布 ==========\n")
        for key, count in sorted(diag.items(), key=lambda x: -x[1]):
            _f.write(f"  {key}: {count}\n")
        _f.write("==========================================\n")
    print(f"[诊断] 过滤分布已写入 {diag_path}", flush=True)
    sys.stderr.write(f"[诊断] 过滤分布已写入 {diag_path}\n")
    sys.stderr.flush()
    # ────────────────────────────────────────────────────────────

    return steps


def _build_market_state_from_structure(
    symbol: str,
    state,
    current_ts,
    atr: float,
    spread: float,
    volatility_score: float,
    extra_meta: dict,
):
    from dpaitrade.core import MarketState

    d1_regime = "trend" if state.phase in ("impulse", "pullback") else ("range" if state.phase == "range" else "unknown")
    d1_bias = state.primary_bias

    market_state = MarketState(
        ts=current_ts,
        symbol=symbol,
        d1_regime=d1_regime,
        d1_bias=d1_bias if d1_bias in ("long", "short", "neutral") else "neutral",
        h4_pullback_active=False,
        h4_boundary_zone=False,
        m15_entry_ready=False,
        m15_entry_direction="neutral",
        atr=atr,
        spread=spread,
        volatility_score=volatility_score,
        meta={
            "structure_state": state,
            **extra_meta,
        },
    )
    return market_state

def build_h4_execution_path(
    h4_rows: list[OHLCRow],
    current_ts,
    history_window: int = 24,
    forward_window: int = 8,
) -> list[dict]:
    """
    为执行模拟器构造 H4 路径：
    - 包含当前时刻之前的历史 H4 bars
    - 以及当前时刻之后的未来 H4 bars

    用途：
    - 让模拟器在持仓过程中按“已完成 H4 bar”更新 trailing stop
    - 判断 H4 波段失效
    """
    history = [r for r in h4_rows if r.ts <= current_ts]
    future = [r for r in h4_rows if r.ts > current_ts]

    history = history[-history_window:]
    future = future[:forward_window]

    rows = history + future
    return [
        {
            "ts": r.ts,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
        }
        for r in rows
    ]


def build_agent(agent_name: str):
    if agent_name == "passthrough":
        print("[信息] 当前使用代理：直通代理")
        return PassthroughAgent()

    print("[信息] 当前使用代理：简单规则过滤代理")
    return SimpleRuleFilterAgent(
        min_rr=0.0,
        max_spread=100000.0,
        min_volatility_score=0.0,
        reject_when_direction_conflict=False,
    )


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def _infer_regime_from_open_reason(open_reason: str) -> str:
    if "趋势延续空头候选信号" in open_reason:
        return "trend"
    if "趋势延续多头候选信号" in open_reason:
        return "trend"
    return "unknown"


def build_trade_breakdown(trade_records: list) -> dict:
    stats: dict = {
        "total_trades": len(trade_records),
        "avg_pnl": 0.0,
        "avg_pnl_r": 0.0,
        "close_reason": defaultdict(lambda: {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "avg_pnl_r": 0.0,
            "sum_pnl_r": 0.0,
        }),
        "direction": defaultdict(lambda: {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "avg_pnl_r": 0.0,
            "sum_pnl_r": 0.0,
        }),
        "regime": defaultdict(lambda: {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "avg_pnl_r": 0.0,
            "sum_pnl_r": 0.0,
        }),
    }

    if not trade_records:
        return stats

    total_pnl = 0.0
    total_pnl_r = 0.0

    for trade in trade_records:
        total_pnl += trade.pnl
        total_pnl_r += trade.pnl_r

        close_reason_key = trade.close_reason
        direction_key = trade.direction
        regime_key = _infer_regime_from_open_reason(trade.open_reason)

        for bucket_key, bucket_name in (
            (close_reason_key, "close_reason"),
            (direction_key, "direction"),
            (regime_key, "regime"),
        ):
            bucket = stats[bucket_name][bucket_key]
            bucket["count"] += 1
            bucket["net_pnl"] += trade.pnl
            bucket["sum_pnl_r"] += trade.pnl_r

            if trade.pnl > 0:
                bucket["wins"] += 1
            elif trade.pnl < 0:
                bucket["losses"] += 1

    stats["avg_pnl"] = total_pnl / len(trade_records)
    stats["avg_pnl_r"] = total_pnl_r / len(trade_records)

    for section in ("close_reason", "direction", "regime"):
        for _, bucket in stats[section].items():
            bucket["avg_pnl_r"] = _safe_div(bucket["sum_pnl_r"], bucket["count"])

    return stats


def print_trade_breakdown(result) -> None:
    stats = build_trade_breakdown(result.trade_records)

    print("\n========== 结果拆解统计 ==========")
    print(f"总成交笔数：{stats['total_trades']}")
    print(f"平均每笔净盈亏：{stats['avg_pnl']:.2f}")
    print(f"平均每笔 R 值：{stats['avg_pnl_r']:.4f}")

    print("\n------ 按平仓原因拆解 ------")
    if stats["close_reason"]:
        for reason, item in stats["close_reason"].items():
            win_rate = _safe_div(item["wins"], item["count"])
            print(
                f"{reason}：count={item['count']}，wins={item['wins']}，"
                f"losses={item['losses']}，win_rate={win_rate:.4f}，"
                f"net_pnl={item['net_pnl']:.2f}，avg_pnl_r={item['avg_pnl_r']:.4f}"
            )
    else:
        print("无数据")

    print("\n------ 按方向拆解 ------")
    if stats["direction"]:
        for direction, item in stats["direction"].items():
            win_rate = _safe_div(item["wins"], item["count"])
            print(
                f"{direction}：count={item['count']}，wins={item['wins']}，"
                f"losses={item['losses']}，win_rate={win_rate:.4f}，"
                f"net_pnl={item['net_pnl']:.2f}，avg_pnl_r={item['avg_pnl_r']:.4f}"
            )
    else:
        print("无数据")

    print("\n------ 按场景拆解 ------")
    if stats["regime"]:
        for regime, item in stats["regime"].items():
            win_rate = _safe_div(item["wins"], item["count"])
            print(
                f"{regime}：count={item['count']}，wins={item['wins']}，"
                f"losses={item['losses']}，win_rate={win_rate:.4f}，"
                f"net_pnl={item['net_pnl']:.2f}，avg_pnl_r={item['avg_pnl_r']:.4f}"
            )
    else:
        print("无数据")


def print_result(result) -> None:
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
        print("\n========== 成交记录（前10笔） ==========")
        for idx, trade in enumerate(result.trade_records[:10], start=1):
            print(
                f"第{idx}笔：symbol={trade.symbol}，direction={trade.direction}，"
                f"entry={trade.entry_price:.5f}，exit={trade.exit_price:.5f}，"
                f"pnl={trade.pnl:.2f}，pnl_r={trade.pnl_r:.4f}，"
                f"open_time={trade.open_time}，close_time={trade.close_time}，"
                f"open_reason={trade.open_reason}，close_reason={trade.close_reason}"
            )
    else:
        print("\n[信息] 本次回测没有成交记录")

    print_trade_breakdown(result)


def main() -> None:
    args = parse_args()

    print("[信息] 开始加载 CSV 数据")
    d1_rows = load_ohlc_csv(args.d1)
    h4_rows = load_ohlc_csv(args.h4)
    m15_rows = load_ohlc_csv(args.m15)
    print(
        f"[信息] CSV 加载完成：D1={len(d1_rows)} 行，H4={len(h4_rows)} 行，M15={len(m15_rows)} 行"
    )

    if not d1_rows or not h4_rows or not m15_rows:
        raise RuntimeError("CSV 数据为空，无法回测")

    explicit_direction_selected = args.allow_long or args.allow_short
    allow_long = args.allow_long or (not explicit_direction_selected)
    allow_short = args.allow_short or (not explicit_direction_selected)

    print(
        f"[信息] 方向开关：allow_long={allow_long}，allow_short={allow_short}"
    )

    steps = build_backtest_steps_from_csv(
        d1_rows=d1_rows,
        h4_rows=h4_rows,
        m15_rows=m15_rows,
        symbol=args.symbol,
        d1_lookback=args.d1_lookback,
        h4_lookback=args.h4_lookback,
        m15_lookback=args.m15_lookback,
        future_window=args.future_window,
        allow_long=allow_long,
        allow_short=allow_short,
        limit=args.limit,
    )
    if not steps:
        raise RuntimeError("未生成任何回测步，请检查 CSV 数据长度或参数设置")

    agent = build_agent(args.agent)

    risk_manager = GuardRiskManager(
        config=RiskGuardConfig(
            default_risk_pct=0.005,
            max_risk_pct_per_trade=0.01,
            max_used_risk_pct=0.02,
            max_consecutive_losses=100,
            max_daily_loss_pct=0.99,
            max_open_positions=1,
            max_spread=200.0,
            min_setup_score=0.0,
            min_rr=0.0,
            reject_direction_conflict=False,
        )
    )
    simulator = SimpleExecutionSimulator()
    engine = BacktestEngine(
        agent=agent,
        risk_manager=risk_manager,
        execution_simulator=simulator,
        cooldown_steps=args.cooldown_steps,
    )

    result = engine.run(
        steps=steps,
        initial_portfolio=PortfolioState(
            cash=args.initial_equity,
            equity=args.initial_equity,
        ),
    )
    print_result(result)


if __name__ == "__main__":
    main()