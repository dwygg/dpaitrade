from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

# ------------------------------
# 方向与市场状态类型定义
# ------------------------------
Direction = Literal["long", "short", "neutral"]
Regime = Literal["trend", "range", "unknown"]

@dataclass(slots=True)
class MarketState:
    """
    市场状态对象。

    这是规则模块整理后的"结构化市场快照"，
    后续 Agent 不应直接处理原始 K 线，而应尽量处理这个对象。

    字段说明：
    - d1_regime: D1 级别市场状态（趋势 / 震荡 / 未知）
    - d1_bias: D1 方向偏置（多 / 空 / 中性）
    - h4_pullback_active: H4 是否处于趋势中的回调阶段
    - h4_boundary_zone: H4 是否位于震荡边界区域
    - m15_entry_ready: M15 是否出现可考虑的入场条件
    - m15_entry_direction: M15 当前入场方向
    - atr / spread / volatility_score: 基础环境特征
    """

    ts: datetime
    symbol: str

    d1_regime: Regime = "unknown"
    d1_bias: Direction = "neutral"

    h4_pullback_active: bool = False
    h4_boundary_zone: bool = False

    m15_entry_ready: bool = False
    m15_entry_direction: Direction = "neutral"

    atr: float = 0.0
    spread: float = 0.0
    volatility_score: float = 0.0

    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class CandidateSignal:
    """
    候选信号对象。

    注意：
    这不是最终交易指令，而是规则系统初步筛出的"可考虑信号"。
    后续还需要经过 Agent 与风控审核。
    """

    ts: datetime
    symbol: str
    direction: Direction

    entry_price: float
    stop_loss: float
    take_profit: Optional[float]

    rr_estimate: float
    reason: str

    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class AgentDecision:
    """
    Agent 决策结果。

    设计原则：
    - Agent 输出必须受限
    - 不允许无限自由发挥
    - 尽量保持结构化，便于回测和统计
    """

    allow_trade: bool
    direction_bias: Direction
    setup_score: float
    risk_adjustment: float
    reason: str

    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(
        cls,
        direction_bias: Direction,
        setup_score: float,
        reason: str,
        risk_adjustment: float = 1.0,
        meta: Optional[dict[str, Any]] = None,
    ) -> "AgentDecision":
        """
        创建一个"允许交易"的决策对象。
        """
        return cls(
            allow_trade=True,
            direction_bias=direction_bias,
            setup_score=setup_score,
            risk_adjustment=risk_adjustment,
            reason=reason,
            meta=meta or {},
        )

    @classmethod
    def reject(
        cls,
        reason: str,
        direction_bias: Direction = "neutral",
        setup_score: float = 0.0,
        meta: Optional[dict[str, Any]] = None,
    ) -> "AgentDecision":
        """
        创建一个"拒绝交易"的决策对象。
        """
        return cls(
            allow_trade=False,
            direction_bias=direction_bias,
            setup_score=setup_score,
            risk_adjustment=0.0,
            reason=reason,
            meta=meta or {},
        )

@dataclass(slots=True)
class RiskDecision:
    """
    风控审批结果。

    风控模块必须是确定性的，不交给 AI 自由控制。
    """

    approved: bool
    risk_pct: float
    reject_reason: Optional[str] = None

    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def approve(cls, risk_pct: float, meta: Optional[dict[str, Any]] = None) -> "RiskDecision":
        """
        创建一个"通过风控审批"的对象。
        """
        return cls(
            approved=True,
            risk_pct=risk_pct,
            reject_reason=None,
            meta=meta or {},
        )

    @classmethod
    def reject(
        cls,
        reject_reason: str,
        meta: Optional[dict[str, Any]] = None,
    ) -> "RiskDecision":
        """
        创建一个"风控拒绝"的对象。
        """
        return cls(
            approved=False,
            risk_pct=0.0,
            reject_reason=reject_reason,
            meta=meta or {},
        )

@dataclass(slots=True)
class PortfolioState:
    """
    账户状态。

    当前先保留最基础字段，后面可以逐步扩展：
    - 持仓列表
    - 日内累计亏损
    - 最大回撤
    - 保证金占用
    """

    cash: float
    equity: float
    used_risk_pct: float = 0.0
    open_positions: int = 0
    consecutive_losses: int = 0
    daily_loss_pct: float = 0.0

    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class TradeRecord:
    """
    成交记录。

    约定：
    - pnl 使用"净盈亏"
    - fees 单独记录，便于后期复盘
    """

    symbol: str
    direction: Direction

    open_time: datetime
    close_time: datetime

    entry_price: float
    exit_price: float
    quantity: float

    pnl: float
    pnl_r: float
    fees: float

    open_reason: str
    close_reason: str

    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class BacktestStep:
    """
    回测步对象。

    一次回测循环至少要知道：
    - 当前市场状态
    - 当前规则模块是否给出了候选信号
    - 当前账户状态（可选）

    后续 D1 / H4 / M15 模块接入后，
    可以逐根 bar 或逐个事件生成这个对象。
    """

    market_state: MarketState
    candidate_signal: Optional[CandidateSignal] = None
    portfolio_state: Optional[PortfolioState] = None

    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class BacktestResult:
    """
    回测结果汇总对象。
    """

    started_at: datetime
    finished_at: datetime

    initial_equity: float
    final_equity: float

    total_steps: int = 0
    total_candidate_signals: int = 0
    total_agent_rejected: int = 0
    total_risk_rejected: int = 0
    total_trades: int = 0

    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    net_pnl: float = 0.0

    trade_records: list[TradeRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)