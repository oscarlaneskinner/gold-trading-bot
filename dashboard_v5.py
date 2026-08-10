"""
Dashboard v5

Unified research dashboard for the gold-trading-bot platform.

Dashboard v5 integrates:
- Master Controller v2
- Market Regime Lab
- Strategy Hall of Fame
- Strategy Learning Engine
- Portfolio Commander
- Intelligent Position Sizer
- Championship Scanner
- Two-Bot Shadow Controller
- Position Manager v2
- Position Valuation
- Performance Analytics
- Risk Manager v2
- Trade Execution Manager v1
- Platform safety

Research-only:
- does not modify the trade ledger,
- does not create a trading client,
- does not submit orders.

Run:
    python -m streamlit run dashboard_v5.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Trading Platform Dashboard v5",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS = PROJECT_ROOT / "reports"

MASTER_V2_PATH = (
    REPORTS
    / "master_controller_v2"
    / "latest_summary.json"
)

MARKET_REGIME_PATH = (
    REPORTS
    / "market_regime"
    / "market_regime_lab_v1.json"
)

HALL_OF_FAME_PATH = (
    REPORTS
    / "hall_of_fame"
    / "strategy_hall_of_fame.json"
)

STRATEGY_LEARNING_PATH = (
    REPORTS
    / "strategy_learning"
    / "latest_summary.json"
)

STRATEGY_LEARNING_CSV_PATH = (
    REPORTS
    / "strategy_learning"
    / "strategy_learning_scores.csv"
)

ALLOCATION_GUIDANCE_PATH = (
    REPORTS
    / "strategy_learning"
    / "allocation_guidance.csv"
)

PORTFOLIO_PATH = (
    REPORTS
    / "portfolio"
    / "portfolio_commander_v1.json"
)

POSITION_SIZER_PATH = (
    REPORTS
    / "position_sizer"
    / "latest_summary.json"
)

POSITION_SIZER_CSV_PATH = (
    REPORTS
    / "position_sizer"
    / "position_sizing_decisions.csv"
)

SCANNER_PATH = (
    REPORTS
    / "scanner"
    / "championship_scanner_v1.json"
)

SHADOW_CONTROLLER_PATH = (
    REPORTS
    / "shadow"
    / "two_bot_shadow_controller_v1.json"
)

POSITION_MANAGER_PATH = (
    REPORTS
    / "position_manager"
    / "latest_summary.json"
)

POSITION_VALUATION_PATH = (
    REPORTS
    / "position_valuation"
    / "latest_summary.json"
)

PERFORMANCE_ANALYTICS_PATH = (
    REPORTS
    / "performance_analytics"
    / "latest_summary.json"
)

EQUITY_CURVE_PATH = (
    REPORTS
    / "performance_analytics"
    / "equity_curve.csv"
)

STRATEGY_ANALYTICS_PATH = (
    REPORTS
    / "performance_analytics"
    / "strategy_analytics.csv"
)

SYMBOL_ANALYTICS_PATH = (
    REPORTS
    / "performance_analytics"
    / "symbol_analytics.csv"
)

RISK_MANAGER_PATH = (
    REPORTS
    / "risk_manager"
    / "latest_summary.json"
)

RISK_POSITION_PATH = (
    REPORTS
    / "risk_manager"
    / "position_risk.csv"
)

TRADE_EXECUTION_PATH = (
    REPORTS
    / "trade_execution_manager"
    / "latest_summary.json"
)

TRADE_DECISIONS_PATH = (
    REPORTS
    / "trade_execution_manager"
    / "trade_decisions.csv"
)


# ============================================================
# LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_json(
    path_text: str,
    default: Any,
) -> Any:

    path = Path(path_text)

    if not path.exists():
        return default

    try:

        raw = path.read_text(
            encoding="utf-8"
        ).strip()

        if not raw:
            return default

        return json.loads(
            raw
        )

    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):

        return default


@st.cache_data(show_spinner=False)
def load_csv(
    path_text: str,
) -> pd.DataFrame:

    path = Path(path_text)

    if not path.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):

        return pd.DataFrame()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:

        if value is None:
            return default

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def money(
    value: Any,
) -> str:

    return (
        f"${safe_float(value):,.2f}"
    )


def percent(
    value: Any,
) -> str:

    return (
        f"{safe_float(value):,.2f}%"
    )


def yes_no(
    value: Any,
) -> str:

    return (
        "Yes"
        if bool(value)
        else "No"
    )


def text(
    value: Any,
    default: str = "Not available",
) -> str:

    result = str(
        value
        if value is not None
        else ""
    ).strip()

    return result or default


def modified_time(
    path: Path,
) -> str:

    if not path.exists():
        return "Not available"

    timestamp = datetime.fromtimestamp(
        path.stat().st_mtime
    )

    return timestamp.strftime(
        "%Y-%m-%d %I:%M:%S %p"
    )


# ============================================================
# LOAD DATA
# ============================================================

master = load_json(
    str(MASTER_V2_PATH),
    {},
)

market_regime = load_json(
    str(MARKET_REGIME_PATH),
    {},
)

hall_of_fame = load_json(
    str(HALL_OF_FAME_PATH),
    {},
)

strategy_learning = load_json(
    str(STRATEGY_LEARNING_PATH),
    {},
)

portfolio = load_json(
    str(PORTFOLIO_PATH),
    {},
)

position_sizer = load_json(
    str(POSITION_SIZER_PATH),
    {},
)

scanner = load_json(
    str(SCANNER_PATH),
    {},
)

shadow_controller = load_json(
    str(SHADOW_CONTROLLER_PATH),
    {},
)

position_manager = load_json(
    str(POSITION_MANAGER_PATH),
    {},
)

position_valuation = load_json(
    str(POSITION_VALUATION_PATH),
    {},
)

performance_analytics = load_json(
    str(PERFORMANCE_ANALYTICS_PATH),
    {},
)

risk_manager = load_json(
    str(RISK_MANAGER_PATH),
    {},
)

trade_execution = load_json(
    str(TRADE_EXECUTION_PATH),
    {},
)

learning_scores = load_csv(
    str(STRATEGY_LEARNING_CSV_PATH)
)

allocation_guidance = load_csv(
    str(ALLOCATION_GUIDANCE_PATH)
)

position_sizing_decisions = load_csv(
    str(POSITION_SIZER_CSV_PATH)
)

equity_curve = load_csv(
    str(EQUITY_CURVE_PATH)
)

strategy_analytics = load_csv(
    str(STRATEGY_ANALYTICS_PATH)
)

symbol_analytics = load_csv(
    str(SYMBOL_ANALYTICS_PATH)
)

risk_positions = load_csv(
    str(RISK_POSITION_PATH)
)

trade_decisions = load_csv(
    str(TRADE_DECISIONS_PATH)
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "📊 AI Trading Platform"
    )

    st.caption(
        "Dashboard v5"
    )

    st.divider()

    if st.button(
        "Refresh Dashboard",
        width="stretch",
    ):

        st.cache_data.clear()
        st.rerun()

    st.subheader(
        "Mode"
    )

    st.success(
        "Research / Shadow"
    )

    safety = master.get(
        "safety",
        {},
    )

    st.write(
        "**Apply Used:** "
        + yes_no(
            safety.get(
                "position_manager_apply_used",
                False,
            )
        )
    )

    st.write(
        "**Trading Client:** "
        + yes_no(
            safety.get(
                "trading_client_created",
                False,
            )
        )
    )

    st.write(
        "**Order Submitted:** "
        + yes_no(
            safety.get(
                "order_submitted",
                False,
            )
        )
    )

    st.divider()

    st.subheader(
        "Latest Reports"
    )

    st.caption(
        "Master Controller v2: "
        + modified_time(
            MASTER_V2_PATH
        )
    )

    st.caption(
        "Strategy Learning: "
        + modified_time(
            STRATEGY_LEARNING_PATH
        )
    )

    st.caption(
        "Position Sizer: "
        + modified_time(
            POSITION_SIZER_PATH
        )
    )

    st.caption(
        "Risk Manager: "
        + modified_time(
            RISK_MANAGER_PATH
        )
    )

    st.caption(
        "Execution Manager: "
        + modified_time(
            TRADE_EXECUTION_PATH
        )
    )

    st.divider()

    st.caption(
        "Read-only dashboard."
    )

    st.caption(
        "No broker orders are submitted."
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "AI Trading Platform Dashboard v5"
)

st.caption(
    "Master Controller v2 • Strategy Learning • "
    "Adaptive Position Sizing • Risk • Execution Preview"
)

st.info(
    "Research-only platform. "
    "No paper or live broker orders are submitted "
    "from this dashboard."
)


# ============================================================
# MASTER CONTROLLER V2
# ============================================================

st.header(
    "Master Controller v2"
)

master_status = text(
    master.get(
        "overall_status",
        "UNKNOWN",
    )
).upper()

controller_columns = st.columns(
    6
)

with controller_columns[0]:

    st.metric(
        "Status",
        master_status,
    )

with controller_columns[1]:

    st.metric(
        "Required Steps",
        (
            f"{safe_int(master.get('required_steps_passed', 0))}/"
            f"{safe_int(master.get('required_step_count', 0))}"
        ),
    )

with controller_columns[2]:

    st.metric(
        "Failed",
        safe_int(
            master.get(
                "failed_step_count",
                0,
            )
        ),
    )

with controller_columns[3]:

    st.metric(
        "Duration",
        (
            f"{safe_float(master.get('duration_seconds', 0.0)):.1f}s"
        ),
    )

with controller_columns[4]:

    st.metric(
        "Safety Passed",
        yes_no(
            safety.get(
                "safety_passed",
                False,
            )
        ),
    )

with controller_columns[5]:

    st.metric(
        "Order Submitted",
        yes_no(
            safety.get(
                "order_submitted",
                False,
            )
        ),
    )


if master_status == "PASSED":

    st.success(
        "Master Controller v2 passed all required modules."
    )

elif master_status == "FAILED":

    st.error(
        "Master Controller v2 reported a failure."
    )

else:

    st.warning(
        f"Master Controller status: {master_status}"
    )


step_results = master.get(
    "step_results",
    [],
)

if isinstance(
    step_results,
    list,
) and step_results:

    step_rows = []

    for result in step_results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        step_rows.append(
            {
                "Module": result.get(
                    "name",
                    "",
                ),

                "Status": result.get(
                    "status",
                    "",
                ),

                "Required": yes_no(
                    result.get(
                        "required",
                        False,
                    )
                ),

                "Return Code": result.get(
                    "returncode",
                    "",
                ),

                "Duration": round(
                    safe_float(
                        result.get(
                            "duration_seconds",
                            0.0,
                        )
                    ),
                    3,
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            step_rows
        ),
        width="stretch",
        hide_index=True,
    )


st.divider()


# ============================================================
# MARKET + PORTFOLIO
# ============================================================

st.header(
    "Market & Portfolio"
)

regime = text(
    market_regime.get(
        "regime",
        "UNKNOWN",
    )
).upper()

estimated_equity = safe_float(
    position_manager.get(
        "estimated_equity",
        position_valuation.get(
            "estimated_equity",
            0.0,
        ),
    )
)

unrealized_pnl = safe_float(
    position_manager.get(
        "unrealized_pnl_dollars",
        position_valuation.get(
            "unrealized_pnl_dollars",
            0.0,
        ),
    )
)

realized_pnl = safe_float(
    position_manager.get(
        "realized_pnl_dollars",
        0.0,
    )
)

portfolio_columns = st.columns(
    6
)

with portfolio_columns[0]:

    st.metric(
        "Market Regime",
        regime,
    )

with portfolio_columns[1]:

    st.metric(
        "Estimated Equity",
        money(
            estimated_equity
        ),
    )

with portfolio_columns[2]:

    st.metric(
        "Unrealized P/L",
        money(
            unrealized_pnl
        ),
    )

with portfolio_columns[3]:

    st.metric(
        "Realized P/L",
        money(
            realized_pnl
        ),
    )

with portfolio_columns[4]:

    st.metric(
        "Open Positions",
        safe_int(
            position_manager.get(
                "open_position_count",
                0,
            )
        ),
    )

with portfolio_columns[5]:

    st.metric(
        "Cash Reserve",
        money(
            portfolio.get(
                "cash_reserve_dollars",
                0.0,
            )
        ),
    )


permissions = market_regime.get(
    "permissions",
    {},
)

permission_columns = st.columns(
    4
)

with permission_columns[0]:

    st.metric(
        "GLD",
        (
            "Allowed"
            if permissions.get(
                "allow_gld_bot",
                False,
            )
            else "Blocked"
        ),
    )

with permission_columns[1]:

    st.metric(
        "Long",
        (
            "Allowed"
            if permissions.get(
                "allow_long_bot",
                False,
            )
            else "Blocked"
        ),
    )

with permission_columns[2]:

    st.metric(
        "Short",
        (
            "Allowed"
            if permissions.get(
                "allow_short_bot",
                False,
            )
            else "Blocked"
        ),
    )

with permission_columns[3]:

    st.metric(
        "Reduce Size",
        yes_no(
            permissions.get(
                "reduce_position_size",
                False,
            )
        ),
    )


st.divider()


# ============================================================
# STRATEGY LEARNING
# ============================================================

st.header(
    "Strategy Learning Engine v1"
)

learning_status = text(
    strategy_learning.get(
        "learning_system_status",
        "UNKNOWN",
    )
).upper()

learning_columns = st.columns(
    7
)

with learning_columns[0]:

    st.metric(
        "Status",
        learning_status,
    )

with learning_columns[1]:

    st.metric(
        "Strategies",
        safe_int(
            strategy_learning.get(
                "strategy_count",
                0,
            )
        ),
    )

with learning_columns[2]:

    st.metric(
        "Top Strategy",
        text(
            strategy_learning.get(
                "top_strategy",
                "None",
            )
        ),
    )

with learning_columns[3]:

    st.metric(
        "Top Score",
        (
            f"{safe_float(strategy_learning.get('top_learning_score', 0.0)):.2f}"
        ),
    )

with learning_columns[4]:

    st.metric(
        "Promote",
        safe_int(
            strategy_learning.get(
                "promote_count",
                0,
            )
        ),
    )

with learning_columns[5]:

    st.metric(
        "Reduce",
        safe_int(
            strategy_learning.get(
                "reduce_count",
                0,
            )
        ),
    )

with learning_columns[6]:

    st.metric(
        "Retire",
        safe_int(
            strategy_learning.get(
                "retire_count",
                0,
            )
        ),
    )


if learning_status == "WAITING_FOR_CLOSED_TRADES":

    st.info(
        "Learning Engine is using research priors only. "
        "Allocation changes remain neutral until closed "
        "shadow-trade evidence accumulates."
    )

elif learning_status == "ACTIVE_LEARNING":

    st.success(
        "Strategy Learning Engine has enough live evidence "
        "for adaptive research recommendations."
    )

else:

    st.caption(
        strategy_learning.get(
            "learning_system_message",
            "",
        )
    )


if (
    isinstance(
        learning_scores,
        pd.DataFrame,
    )
    and not learning_scores.empty
):

    preferred_columns = [
        column
        for column in [
            "strategy_name",
            "role",
            "learning_score",
            "evidence_label",
            "live_trade_count",
            "recommendation",
            "suggested_allocation_multiplier",
            "hall_profit_factor",
            "hall_drawdown_percent",
            "hall_consistency_percent",
        ]
        if column
        in learning_scores.columns
    ]

    st.dataframe(
        learning_scores[
            preferred_columns
        ].head(
            20
        ),
        width="stretch",
        hide_index=True,
    )


with st.expander(
    "Allocation Guidance"
):

    if (
        isinstance(
            allocation_guidance,
            pd.DataFrame,
        )
        and not allocation_guidance.empty
    ):

        st.dataframe(
            allocation_guidance.head(
                30
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.caption(
            "No allocation guidance available."
        )


st.divider()


# ============================================================
# INTELLIGENT POSITION SIZER
# ============================================================

st.header(
    "Intelligent Position Sizer v1"
)

sizer_columns = st.columns(
    6
)

with sizer_columns[0]:

    st.metric(
        "Proposals",
        safe_int(
            position_sizer.get(
                "proposal_count",
                0,
            )
        ),
    )

with sizer_columns[1]:

    st.metric(
        "Positive Sizes",
        safe_int(
            position_sizer.get(
                "positive_size_count",
                0,
            )
        ),
    )

with sizer_columns[2]:

    st.metric(
        "Zero Sizes",
        safe_int(
            position_sizer.get(
                "zero_size_count",
                0,
            )
        ),
    )

with sizer_columns[3]:

    st.metric(
        "Recommended $",
        money(
            position_sizer.get(
                "total_recommended_dollars",
                0.0,
            )
        ),
    )

with sizer_columns[4]:

    st.metric(
        "Recommended %",
        percent(
            position_sizer.get(
                "total_recommended_percent",
                0.0,
            )
        ),
    )

with sizer_columns[5]:

    st.metric(
        "Research Only",
        yes_no(
            position_sizer.get(
                "research_only",
                True,
            )
        ),
    )


if (
    isinstance(
        position_sizing_decisions,
        pd.DataFrame,
    )
    and not position_sizing_decisions.empty
):

    st.dataframe(
        position_sizing_decisions,
        width="stretch",
        hide_index=True,
    )


st.divider()


# ============================================================
# RISK MANAGER
# ============================================================

st.header(
    "Risk Manager v2"
)

risk_metrics = risk_manager.get(
    "portfolio_metrics",
    {},
)

risk_columns = st.columns(
    6
)

with risk_columns[0]:

    st.metric(
        "Risk Score",
        (
            f"{safe_int(risk_manager.get('risk_score', 100))}/100"
        ),
    )

with risk_columns[1]:

    st.metric(
        "Risk Level",
        text(
            risk_manager.get(
                "risk_level",
                "UNKNOWN",
            )
        ),
    )

with risk_columns[2]:

    st.metric(
        "Exposure",
        percent(
            risk_metrics.get(
                "total_exposure_percent",
                0.0,
            )
        ),
    )

with risk_columns[3]:

    st.metric(
        "Portfolio Heat",
        percent(
            risk_metrics.get(
                "portfolio_heat_percent",
                0.0,
            )
        ),
    )

with risk_columns[4]:

    st.metric(
        "Combined P/L",
        money(
            risk_metrics.get(
                "combined_pnl_dollars",
                0.0,
            )
        ),
    )

with risk_columns[5]:

    st.metric(
        "Warnings",
        safe_int(
            risk_manager.get(
                "warning_count",
                0,
            )
        ),
    )


risk_trade_status = text(
    risk_manager.get(
        "trade_acceptance_status",
        "UNKNOWN",
    )
).upper()

if risk_trade_status == "SHADOW_RESEARCH_ALLOWED":

    st.success(
        "Risk Manager permits shadow-research proposals."
    )

elif risk_trade_status == "REVIEW_BEFORE_NEW_SHADOW_TRADE":

    st.warning(
        "Risk Manager requires review before new shadow trades."
    )

elif risk_trade_status == "BLOCK_NEW_SHADOW_TRADES":

    st.error(
        "Risk Manager is blocking new shadow trades."
    )


if (
    isinstance(
        risk_positions,
        pd.DataFrame,
    )
    and not risk_positions.empty
):

    with st.expander(
        "Position Risk Detail"
    ):

        st.dataframe(
            risk_positions,
            width="stretch",
            hide_index=True,
        )


st.divider()


# ============================================================
# TRADE EXECUTION PREVIEW
# ============================================================

st.header(
    "Trade Execution Manager v1"
)

execution_columns = st.columns(
    5
)

with execution_columns[0]:

    st.metric(
        "Proposals",
        safe_int(
            trade_execution.get(
                "proposal_count",
                0,
            )
        ),
    )

with execution_columns[1]:

    st.metric(
        "Approved",
        safe_int(
            trade_execution.get(
                "approved_count",
                0,
            )
        ),
    )

with execution_columns[2]:

    st.metric(
        "Rejected",
        safe_int(
            trade_execution.get(
                "rejected_count",
                0,
            )
        ),
    )

with execution_columns[3]:

    st.metric(
        "Preview Only",
        yes_no(
            trade_execution.get(
                "preview_only",
                True,
            )
        ),
    )

with execution_columns[4]:

    st.metric(
        "Order Submitted",
        yes_no(
            trade_execution.get(
                "order_submitted",
                False,
            )
        ),
    )


if (
    isinstance(
        trade_decisions,
        pd.DataFrame,
    )
    and not trade_decisions.empty
):

    st.dataframe(
        trade_decisions,
        width="stretch",
        hide_index=True,
    )


st.divider()


# ============================================================
# OPEN POSITIONS
# ============================================================

st.header(
    "Open Positions"
)

open_positions = position_manager.get(
    "open_positions",
    [],
)

if isinstance(
    open_positions,
    list,
) and open_positions:

    position_rows = []

    for position in open_positions:

        if not isinstance(
            position,
            dict,
        ):
            continue

        position_rows.append(
            {
                "Symbol": position.get(
                    "symbol",
                    "",
                ),

                "Strategy": position.get(
                    "strategy_name",
                    "",
                ),

                "Side": position.get(
                    "side",
                    "",
                ),

                "Entry": safe_float(
                    position.get(
                        "entry_price",
                        0.0,
                    )
                ),

                "Current": safe_float(
                    position.get(
                        "current_price",
                        0.0,
                    )
                ),

                "Unrealized P/L": safe_float(
                    position.get(
                        "unrealized_pnl_dollars",
                        0.0,
                    )
                ),

                "Unrealized %": safe_float(
                    position.get(
                        "unrealized_pnl_percent",
                        0.0,
                    )
                ),

                "Stop": safe_float(
                    position.get(
                        "stop_price",
                        0.0,
                    )
                ),

                "Target": safe_float(
                    position.get(
                        "target_price",
                        0.0,
                    )
                ),

                "Holding Days": safe_int(
                    position.get(
                        "holding_days",
                        0,
                    )
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            position_rows
        ),
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No open shadow positions."
    )


st.divider()


# ============================================================
# SCANNER
# ============================================================

st.header(
    "Championship Scanner"
)

top_longs = scanner.get(
    "top_longs",
    [],
)

top_shorts = scanner.get(
    "top_shorts",
    [],
)

left, right = st.columns(
    2
)

with left:

    st.subheader(
        "Top Longs"
    )

    if isinstance(
        top_longs,
        list,
    ) and top_longs:

        frame = pd.DataFrame(
            top_longs
        )

        wanted = [
            column
            for column in [
                "symbol",
                "score",
                "close",
                "rsi_14",
                "atr_percent",
                "suggested_stop",
                "suggested_target",
            ]
            if column in frame.columns
        ]

        st.dataframe(
            frame[
                wanted
            ].head(
                10
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.caption(
            "No long candidates."
        )


with right:

    st.subheader(
        "Top Shorts"
    )

    if isinstance(
        top_shorts,
        list,
    ) and top_shorts:

        frame = pd.DataFrame(
            top_shorts
        )

        wanted = [
            column
            for column in [
                "symbol",
                "score",
                "close",
                "rsi_14",
                "atr_percent",
                "suggested_stop",
                "suggested_target",
            ]
            if column in frame.columns
        ]

        st.dataframe(
            frame[
                wanted
            ].head(
                10
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.caption(
            "No short candidates."
        )


st.divider()


# ============================================================
# PERFORMANCE
# ============================================================

st.header(
    "Performance Analytics"
)

performance_columns = st.columns(
    6
)

with performance_columns[0]:

    st.metric(
        "Status",
        text(
            performance_analytics.get(
                "analytics_status",
                "UNKNOWN",
            )
        ),
    )

with performance_columns[1]:

    st.metric(
        "Closed Trades",
        safe_int(
            performance_analytics.get(
                "closed_position_count",
                0,
            )
        ),
    )

with performance_columns[2]:

    st.metric(
        "Win Rate",
        percent(
            performance_analytics.get(
                "win_rate_percent",
                0.0,
            )
        ),
    )

with performance_columns[3]:

    st.metric(
        "Profit Factor",
        (
            f"{safe_float(performance_analytics.get('profit_factor', 0.0)):.2f}"
        ),
    )

with performance_columns[4]:

    st.metric(
        "Max Drawdown",
        percent(
            performance_analytics.get(
                "maximum_drawdown_percent",
                0.0,
            )
        ),
    )

with performance_columns[5]:

    st.metric(
        "Sharpe",
        (
            f"{safe_float(performance_analytics.get('sharpe_ratio', 0.0)):.2f}"
        ),
    )


if (
    isinstance(
        equity_curve,
        pd.DataFrame,
    )
    and not equity_curve.empty
    and "equity" in equity_curve.columns
):

    chart_frame = equity_curve.copy()

    chart_frame[
        "equity"
    ] = pd.to_numeric(
        chart_frame[
            "equity"
        ],
        errors="coerce",
    )

    chart_frame = chart_frame.dropna(
        subset=[
            "equity"
        ]
    )

    if "date" in chart_frame.columns:

        chart_frame[
            "date"
        ] = chart_frame[
            "date"
        ].astype(
            str
        )

        chart_frame = chart_frame.set_index(
            "date"
        )

    st.line_chart(
        chart_frame[
            [
                "equity"
            ]
        ],
        width="stretch",
    )

else:

    st.info(
        "Equity curve will become more meaningful "
        "after trades begin closing."
    )


analytics_left, analytics_right = st.columns(
    2
)

with analytics_left:

    st.subheader(
        "Strategy Analytics"
    )

    if (
        isinstance(
            strategy_analytics,
            pd.DataFrame,
        )
        and not strategy_analytics.empty
    ):

        st.dataframe(
            strategy_analytics.head(
                20
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.caption(
            "Waiting for closed trades."
        )


with analytics_right:

    st.subheader(
        "Symbol Analytics"
    )

    if (
        isinstance(
            symbol_analytics,
            pd.DataFrame,
        )
        and not symbol_analytics.empty
    ):

        st.dataframe(
            symbol_analytics.head(
                20
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.caption(
            "Waiting for closed trades."
        )


st.divider()


# ============================================================
# HALL OF FAME
# ============================================================

st.header(
    "Strategy Hall of Fame"
)

hall_columns = st.columns(
    4
)

with hall_columns[0]:

    st.metric(
        "Strategies",
        safe_int(
            hall_of_fame.get(
                "strategy_count",
                0,
            )
        ),
    )

with hall_columns[1]:

    st.metric(
        "Scanner Records Excluded",
        yes_no(
            hall_of_fame.get(
                "scanner_records_excluded",
                False,
            )
        ),
    )

with hall_columns[2]:

    st.metric(
        "Production Changed",
        yes_no(
            hall_of_fame.get(
                "production_strategy_changed",
                False,
            )
        ),
    )

with hall_columns[3]:

    st.metric(
        "Orders Submitted",
        yes_no(
            hall_of_fame.get(
                "order_submitted",
                False,
            )
        ),
    )


top_strategies = hall_of_fame.get(
    "top_strategies",
    [],
)

if isinstance(
    top_strategies,
    list,
) and top_strategies:

    frame = pd.DataFrame(
        top_strategies
    )

    preferred = [
        column
        for column in [
            "rank",
            "strategy_name",
            "role",
            "symbol",
            "score",
            "return_percent",
            "drawdown_percent",
            "profit_factor",
            "consistency_percent",
            "trade_count",
            "status",
        ]
        if column in frame.columns
    ]

    st.dataframe(
        frame[
            preferred
        ].head(
            25
        ),
        width="stretch",
        hide_index=True,
    )


st.divider()


# ============================================================
# SAFETY PANEL
# ============================================================

st.header(
    "Platform Safety"
)

safety_rows = [
    {
        "Check": "Shadow Mode",
        "Value": yes_no(
            safety.get(
                "shadow_mode",
                True,
            )
        ),
    },

    {
        "Check": "Position Manager Mode",
        "Value": text(
            safety.get(
                "position_manager_mode",
                "UNKNOWN",
            )
        ),
    },

    {
        "Check": "Apply Used",
        "Value": yes_no(
            safety.get(
                "position_manager_apply_used",
                False,
            )
        ),
    },

    {
        "Check": "Trading Client Created",
        "Value": yes_no(
            safety.get(
                "trading_client_created",
                False,
            )
        ),
    },

    {
        "Check": "Order Submitted",
        "Value": yes_no(
            safety.get(
                "order_submitted",
                False,
            )
        ),
    },

    {
        "Check": "Safety Passed",
        "Value": yes_no(
            safety.get(
                "safety_passed",
                False,
            )
        ),
    },
]

st.dataframe(
    pd.DataFrame(
        safety_rows
    ),
    width="stretch",
    hide_index=True,
)


if bool(
    safety.get(
        "safety_passed",
        False,
    )
):

    st.success(
        "Safety checks passed. "
        "Research/shadow mode is active and no order was submitted."
    )

else:

    st.error(
        "One or more safety checks require review."
    )


# ============================================================
# REPORT INSPECTOR
# ============================================================

st.divider()

st.header(
    "Report Inspector"
)

report_options = {
    "Master Controller v2": master,
    "Market Regime": market_regime,
    "Hall of Fame": hall_of_fame,
    "Strategy Learning": strategy_learning,
    "Portfolio Commander": portfolio,
    "Position Sizer": position_sizer,
    "Scanner": scanner,
    "Shadow Controller": shadow_controller,
    "Position Manager": position_manager,
    "Position Valuation": position_valuation,
    "Performance Analytics": performance_analytics,
    "Risk Manager": risk_manager,
    "Trade Execution": trade_execution,
}

selected_name = st.selectbox(
    "Choose a report",
    list(
        report_options.keys()
    ),
)

with st.expander(
    f"View {selected_name}",
    expanded=False,
):

    st.json(
        report_options[
            selected_name
        ]
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.success(
    "Dashboard v5 loaded successfully."
)

st.caption(
    "AI Trading Platform Dashboard v5 • "
    "Research and shadow monitoring only • "
    "No broker orders are submitted from this dashboard."
)