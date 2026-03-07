from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from dpaitrade.agent import PassthroughAgent, SimpleRuleFilterAgent
from dpaitrade.backtest import BacktestEngine
from dpaitrade.core import BacktestStep, PortfolioState
from dpaitrade.data import OHLCRow, load_ohlc_csv, slice_recent_rows
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
from collections import defaultdict

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="dpaitrade 多步 CSV 回测入口")

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
        default=8,
        help="执行模拟时，用后续多少根 M15 生成 future_high/future_low/future_close",
    )

    # 各周期回看窗口
    parser.add_argument("--d1-lookback", type=int, default=20, help="D1 回看窗口")
    parser.add_argument("--h4-lookback", type=int, default=24, help="H4 回看窗口")
    parser.add_argument("--m15-lookback", type=int, default=12, help="M15 回看窗口")

    return parser.parse_args()


def to_d1_bars(rows: list[OHLCRow]) -> list[D1Bar]:
    return [D1Bar(open=r.open, high=r.high, low=r.low, close=r.close) for r in rows]


def to_h4_bars(rows: list[OHLCRow]) -> list[H4Bar]:
    return [H4Bar(open=r.open, high=r.high, low=r.low, close=r.close) for r in rows]


def to_m15_bars(rows: list[OHLCRow]) -> list[M15Bar]:
    return [M15Bar(open=r.open, high=r.high, low=r.low, close=r.close) for r in rows]


def infer_future_prices(
    m15_rows: list[OHLCRow],
    current_index: int,
    future_window: int,
) -> tuple[float | None, float | None, float | None]:
    """
    用当前 M15 之后的 future_window 根 M15，推导 future_high / future_low / future_close。

    返回：
    - future_high
    - future_low
    - future_close
    """
    start = current_index + 1
    end = min(len(m15_rows), current_index + 1 + future_window)
    future_rows = m15_rows[start:end]

    if not future_rows:
        return None, None, None

    future_high = max(row.high for row in future_rows)
    future_low = min(row.low for row in future_rows)
    future_close = future_rows[-1].close
    return future_high, future_low, future_close


def build_backtest_steps_from_csv(
    d1_rows: list[OHLCRow],
    h4_rows: list[OHLCRow],
    m15_rows: list[OHLCRow],
    symbol: str,
    d1_lookback: int,
    h4_lookback: int,
    m15_lookback: int,
    future_window: int,
    limit: int = 0,
) -> list[BacktestStep]:
    """
    从三份 CSV 数据构造多步回测输入。

    规则：
    - 以 M15 为主循环
    - 每一步取“当前时刻之前”的最近 D1/H4/M15 窗口
    - 用这些窗口生成 MarketState 与 CandidateSignal
    - 再把未来 N 根 M15 的高低收，放入 market_state.meta 给执行模拟器使用
    """
    print("[信息] 开始构造多步回测数据")

    d1_detector = D1RegimeDetector(
        lookback=d1_lookback,
        trend_threshold=0.18,
        range_threshold=0.08,
    )
    h4_detector = H4PullbackDetector(
        swing_lookback=h4_lookback,
        pullback_min_ratio=0.25,
        pullback_max_ratio=0.70,
        boundary_near_ratio=0.15,
    )
    m15_detector = M15EntryDetector(
        structure_lookback=m15_lookback,
        min_rr=1.5,
    )
    signal_builder = SignalBuilder()

    d1_ts = [r.ts for r in d1_rows]
    h4_ts = [r.ts for r in h4_rows]
    m15_ts = [r.ts for r in m15_rows]

    steps: list[BacktestStep] = []
    processed = 0

    for idx, current_m15 in enumerate(m15_rows):
        current_ts = current_m15.ts

        d1_window_rows = slice_recent_rows(d1_rows, d1_ts, current_ts, d1_lookback)
        h4_window_rows = slice_recent_rows(h4_rows, h4_ts, current_ts, h4_lookback)
        m15_window_rows = slice_recent_rows(m15_rows, m15_ts, current_ts, m15_lookback)

        # 窗口不足时，跳过
        if len(d1_window_rows) < d1_lookback:
            continue
        if len(h4_window_rows) < h4_lookback:
            continue
        if len(m15_window_rows) < m15_lookback:
            continue

        d1_result = d1_detector.detect(to_d1_bars(d1_window_rows))
        h4_result = h4_detector.detect(
            to_h4_bars(h4_window_rows),
            d1_regime=d1_result.regime,
            d1_bias=d1_result.bias,
        )
        m15_result = m15_detector.detect(
            to_m15_bars(m15_window_rows),
            preferred_direction=h4_result.preferred_direction,
        )

        # spread / atr / volatility_score 优先从当前 m15 行 meta 中读取
        atr = float(current_m15.meta.get("atr", 0.0) or 0.0)
        spread = float(current_m15.meta.get("spread", 0.0) or 0.0)
        volatility_score = float(current_m15.meta.get("volatility_score", 0.0) or 0.0)

        ctx = SignalBuildContext(
            ts=current_ts,
            symbol=symbol,
            atr=atr,
            spread=spread,
            volatility_score=volatility_score,
        )

        market_state = signal_builder.build_market_state(ctx, d1_result, h4_result, m15_result)

        future_high, future_low, future_close = infer_future_prices(
            m15_rows=m15_rows,
            current_index=idx,
            future_window=future_window,
        )
        market_state.meta["future_high"] = future_high
        market_state.meta["future_low"] = future_low
        market_state.meta["future_close"] = future_close

        candidate_signal = signal_builder.build_candidate_signal(
            ctx,
            d1_result,
            h4_result,
            m15_result,
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
    return steps


def build_agent(agent_name: str):
    """
    根据参数创建代理。
    """
    if agent_name == "passthrough":
        print("[信息] 当前使用代理：直通代理")
        return PassthroughAgent()

    print("[信息] 当前使用代理：简单规则过滤代理")
    return SimpleRuleFilterAgent(
        min_rr=1.2,
        max_spread=100000.0,
        min_volatility_score=0.2,
        reject_when_direction_conflict=True,
    )




def _safe_div(a: float, b: float) -> float:
    """
    安全除法，避免分母为 0。
    """
    if b == 0:
        return 0.0
    return a / b


def _infer_regime_from_open_reason(open_reason: str) -> str:
    """
    从 open_reason 里推断信号来源场景。

    当前阶段先使用字符串判断：
    - 包含“趋势候选信号” -> trend
    - 包含“边界候选信号” -> range
    - 否则 -> unknown

    后续更推荐把 regime 直接写入 TradeRecord.meta。
    """
    if "趋势候选信号" in open_reason:
        return "trend"
    if "边界候选信号" in open_reason:
        return "range"
    return "unknown"


def build_trade_breakdown(trade_records: list) -> dict:
    """
    构建成交记录拆解统计。

    输出结构包括：
    - close_reason 维度统计
    - direction 维度统计
    - regime 维度统计
    - 全局平均 pnl / pnl_r
    """
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

    # 计算各桶平均值
    for section in ("close_reason", "direction", "regime"):
        for _, bucket in stats[section].items():
            bucket["avg_pnl_r"] = _safe_div(bucket["sum_pnl_r"], bucket["count"])

    return stats

def print_trade_breakdown(result) -> None:
    """
    打印回测结果拆解统计。
    """
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
    """
    打印回测结果。
    """
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
    # 新增：打印拆解统计
    print_trade_breakdown(result)

def main() -> None:
    """
    主入口。
    """
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

    steps = build_backtest_steps_from_csv(
        d1_rows=d1_rows,
        h4_rows=h4_rows,
        m15_rows=m15_rows,
        symbol=args.symbol,
        d1_lookback=args.d1_lookback,
        h4_lookback=args.h4_lookback,
        m15_lookback=args.m15_lookback,
        future_window=args.future_window,
        limit=args.limit,
    )
    if not steps:
        raise RuntimeError("未生成任何回测步，请检查 CSV 数据长度或 lookback 参数")

    agent = build_agent(args.agent)

    risk_manager = GuardRiskManager(
        config=RiskGuardConfig(
            default_risk_pct=0.005,
            max_risk_pct_per_trade=0.01,
            max_used_risk_pct=0.02,
            max_consecutive_losses=100,
            max_daily_loss_pct=0.02,
            max_open_positions=1,
            max_spread=200.0,
            min_setup_score=0.45,
            min_rr=1.2,
            reject_direction_conflict=True,
        )
    )
    simulator = SimpleExecutionSimulator()
    engine = BacktestEngine(
        agent=agent,
        risk_manager=risk_manager,
        execution_simulator=simulator,
        cooldown_steps=4,
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