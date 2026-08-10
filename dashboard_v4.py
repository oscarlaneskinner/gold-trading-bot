"""
Dashboard v4

Unified research dashboard for the gold-trading-bot platform.

Dashboard v4 combines:
- Master Controller status
- Market Regime
- Portfolio status
- Position Manager
- Risk Manager v2
- Trade Execution Manager v1
- Performance Analytics v1
- Scanner leaders
- Hall of Fame
- Open positions
- Shadow observation progress
- Safety checks

Research-only:
- does not modify the trade ledger,
- does not create a trading client,
- does not submit orders.

Run:
    python -m streamlit run dashboard_v4.py
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
    page_title="AI Trading Platform Dashboard v4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS = PROJECT_ROOT / "reports"

MASTER_PATH = (
    REPORTS
    / "master_controller"
    / "latest_summary.json"
)

MARKET_REGIME_PATH = (
    REPORTS
    / "market_regime"
    / "market_regime_lab_v1.json"
)

PORTFOLIO_PATH = (
    REPORTS
    / "portfolio"
    / "portfolio_commander_v1.json"
)

SCANNER_PATH = (
    REPORTS
    / "scanner"
    / "championship_scanner_v1.json"
)

HALL_OF_FAME_PATH = (
    REPORTS
    / "hall_of_fame"
    / "strategy_hall_of_fame.json"
)

SHADOW_CONTROLLER_PATH = (
    REPORTS
    / "shadow"
    / "two_bot_shadow_controller_v1.json"
)

PERFORMANCE_PATH = (
    REPORTS
    / "performance"
    / "latest_summary.json"
)

TRADE_LEDGER_PATH = (
    REPORTS
    / "performance"
    / "trade_ledger.json"
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

PERFORMANCE_EQUITY_PATH = (
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

POSITION_RISK_PATH = (
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

OBSERVATION_STATE_PATH = (
    REPORTS
    / "shadow_observation"
    / "observation_state.json"
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

        return json.loads(raw)

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
        return pd.read_csv(path)

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

        return float(value)

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

        return int(value)

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


def file_modified(
    path: Path,
) -> str:
    if not path.exists():
        return "Not available"

    timestamp = (
        datetime.fromtimestamp(
            path.stat().st_mtime
        )
    )

    return timestamp.strftime(
        "%Y-%m-%d %I:%M:%S %p"
    )


# ============================================================
# LOAD PLATFORM REPORTS
# ============================================================

master = load_json(
    str(MASTER_PATH),
    {},
)

market_regime = load_json(
    str(MARKET_REGIME_PATH),
    {},
)

portfolio = load_json(
    str(PORTFOLIO_PATH),
    {},
)

scanner = load_json(
    str(SCANNER_PATH),
    {},
)

hall_of_fame = load_json(
    str(HALL_OF_FAME_PATH),
    {},
)

shadow_controller = load_json(
    str(SHADOW_CONTROLLER_PATH),
    {},
)

performance = load_json(
    str(PERFORMANCE_PATH),
    {},
)

trade_ledger = load_json(
    str(TRADE_LEDGER_PATH),
    [],
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

observation_state = load_json(
    str(OBSERVATION_STATE_PATH),
    {},
)

equity_curve = load_csv(
    str(PERFORMANCE_EQUITY_PATH)
)

strategy_analytics = load_csv(
    str(STRATEGY_ANALYTICS_PATH)
)

symbol_analytics = load_csv(
    str(SYMBOL_ANALYTICS_PATH)
)

position_risk = load_csv(
    str(POSITION_RISK_PATH)
)

trade_decisions = load_csv(
    str(TRADE_DECISIONS_PATH)
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "📈 AI Trading Platform"
    )

    st.caption(
        "Dashboard v4"
    )

    st.divider()

    if st.button(
        "Refresh Dashboard",
        width="stretch",
    ):
        st.cache_data.clear()
        st.rerun()

    st.subheader(
        "Operating Mode"
    )

    st.success(
        "Research / Shadow Mode"
    )

    safety = master.get(
        "safety",
        {},
    )

    st.write(
        "**Position Manager Apply Used:** "
        + yes_no(
            safety.get(
                "position_manager_apply_used",
                False,
            )
        )
    )

    st.write(
        "**Trading Client Created:** "
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
        "Latest Updates"
    )

    st.caption(
        "Master Controller: "
        + file_modified(
            MASTER_PATH
        )
    )

    st.caption(
        "Risk Manager: "
        + file_modified(
            RISK_MANAGER_PATH
        )
    )

    st.caption(
        "Trade Execution: "
        + file_modified(
            TRADE_EXECUTION_PATH
        )
    )

    st.caption(
        "Position Manager: "
        + file_modified(
            POSITION_MANAGER_PATH
        )
    )

    st.caption(
        "Performance Analytics: "
        + file_modified(
            PERFORMANCE_ANALYTICS_PATH
        )
    )

    st.divider()

    st.caption(
        "Dashboard v4 reads research reports only."
    )

    st.caption(
        "It does not place trades."
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "AI Trading Platform Dashboard v4"
)

st.caption(
    "Unified research, risk, execution-preview, "
    "portfolio, strategy, and performance monitoring"
)

st.info(
    "Research-only system. "
    "No live or paper broker orders are submitted "
    "from this dashboard."
)


# ============================================================
# MASTER CONTROLLER
# ============================================================

st.header(
    "Master Controller"
)

master_status = text(
    master.get(
        "overall_status",
        "NOT AVAILABLE",
    )
).upper()

master_columns = st.columns(
    5
)

with master_columns[0]:

    st.metric(
        "Controller Status",
        master_status,
    )

with master_columns[1]:

    st.metric(
        "Required Steps",
        (
            f"{safe_int(master.get('required_steps_passed', 0))}/"
            f"{safe_int(master.get('required_step_count', 0))}"
        ),
    )

with master_columns[2]:

    st.metric(
        "Failed Steps",
        safe_int(
            master.get(
                "failed_step_count",
                0,
            )
        ),
    )

with master_columns[3]:

    st.metric(
        "Run Duration",
        (
            f"{safe_float(master.get('duration_seconds', 0.0)):.1f}s"
        ),
    )

with master_columns[4]:

    st.metric(
        "Orders Submitted",
        yes_no(
            safety.get(
                "order_submitted",
                False,
            )
        ),
    )

if master_status == "PASSED":

    st.success(
        "Latest Master Controller run passed "
        "all required modules."
    )

elif master_status == "FAILED":

    st.error(
        "Latest Master Controller run failed."
    )

else:

    st.warning(
        "Master Controller status is not available "
        "or requires review."
    )


# ============================================================
# CONTROLLER STEP RESULTS
# ============================================================

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
                "Module": text(
                    result.get(
                        "name",
                        "",
                    ),
                    "",
                ),

                "Status": text(
                    result.get(
                        "status",
                        "",
                    ),
                    "",
                ),

                "Required": yes_no(
                    result.get(
                        "required",
                        False,
                    )
                ),

                "Return Code": str(
                    result.get(
                        "returncode",
                        "",
                    )
                ),

                "Duration Seconds": round(
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

market_columns = st.columns(
    6
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
            performance.get(
                "ending_equity",
                2000.0,
            ),
        ),
    ),
    2000.0,
)

starting_capital = safe_float(
    position_manager.get(
        "starting_capital",
        performance.get(
            "starting_capital",
            2000.0,
        ),
    ),
    2000.0,
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
        performance.get(
            "total_pnl_dollars",
            0.0,
        ),
    )
)

with market_columns[0]:

    st.metric(
        "Market Regime",
        regime,
    )

with market_columns[1]:

    st.metric(
        "Estimated Equity",
        money(
            estimated_equity
        ),
        delta=money(
            estimated_equity
            - starting_capital
        ),
    )

with market_columns[2]:

    st.metric(
        "Unrealized P/L",
        money(
            unrealized_pnl
        ),
    )

with market_columns[3]:

    st.metric(
        "Realized P/L",
        money(
            realized_pnl
        ),
    )

with market_columns[4]:

    st.metric(
        "Open Positions",
        safe_int(
            position_manager.get(
                "open_position_count",
                performance.get(
                    "open_positions",
                    0,
                ),
            )
        ),
    )

with market_columns[5]:

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
        "GLD Bot",
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
        "Long Bot",
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
        "Short Bot",
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
        "Reduce Position Size",
        yes_no(
            permissions.get(
                "reduce_position_size",
                False,
            )
        ),
    )


st.divider()


# ============================================================
# RISK MANAGER
# ============================================================

st.header(
    "Risk Manager v2"
)

risk_score = safe_int(
    risk_manager.get(
        "risk_score",
        100,
    )
)

risk_level = text(
    risk_manager.get(
        "risk_level",
        "UNKNOWN",
    )
).upper()

risk_status = text(
    risk_manager.get(
        "trade_acceptance_status",
        "UNKNOWN",
    )
).upper()

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
        f"{risk_score}/100",
    )

with risk_columns[1]:

    st.metric(
        "Risk Level",
        risk_level,
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


if risk_status == "SHADOW_RESEARCH_ALLOWED":

    st.success(
        "Risk Manager currently permits new "
        "shadow-research trades."
    )

elif risk_status == "REVIEW_BEFORE_NEW_SHADOW_TRADE":

    st.warning(
        "Risk Manager requires review before "
        "a new shadow trade."
    )

elif risk_status == "BLOCK_NEW_SHADOW_TRADES":

    st.error(
        "Risk Manager is blocking new shadow trades."
    )

else:

    st.info(
        f"Risk status: {risk_status}"
    )


warnings = risk_manager.get(
    "warnings",
    [],
)

if isinstance(
    warnings,
    list,
) and warnings:

    warning_rows = []

    for warning in warnings:

        if isinstance(
            warning,
            dict,
        ):

            warning_rows.append(
                {
                    "Severity": text(
                        warning.get(
                            "severity",
                            "",
                        ),
                        "",
                    ),

                    "Category": text(
                        warning.get(
                            "category",
                            "",
                        ),
                        "",
                    ),

                    "Symbol": text(
                        warning.get(
                            "symbol",
                            "",
                        ),
                        "",
                    ),

                    "Message": text(
                        warning.get(
                            "message",
                            "",
                        ),
                        "",
                    ),
                }
            )

    st.dataframe(
        pd.DataFrame(
            warning_rows
        ),
        width="stretch",
        hide_index=True,
    )

else:

    st.caption(
        "No active Risk Manager warnings."
    )


st.divider()


# ============================================================
# TRADE EXECUTION MANAGER
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

else:

    decisions = trade_execution.get(
        "decisions",
        [],
    )

    if isinstance(
        decisions,
        list,
    ) and decisions:

        decision_rows = []

        for decision in decisions:

            if not isinstance(
                decision,
                dict,
            ):
                continue

            decision_rows.append(
                {
                    "Symbol": decision.get(
                        "symbol",
                        "",
                    ),

                    "Side": decision.get(
                        "side",
                        "",
                    ),

                    "Strategy": decision.get(
                        "strategy_name",
                        "",
                    ),

                    "Dollars": safe_float(
                        decision.get(
                            "proposed_dollars",
                            0.0,
                        )
                    ),

                    "Decision": decision.get(
                        "decision",
                        "",
                    ),

                    "Duplicate": yes_no(
                        decision.get(
                            "duplicate_open_position",
                            False,
                        )
                    ),

                    "Reasons": " | ".join(
                        decision.get(
                            "reasons",
                            [],
                        )
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                decision_rows
            ),
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

manager_positions = position_manager.get(
    "open_positions",
    [],
)

if isinstance(
    manager_positions,
    list,
) and manager_positions:

    position_rows = []

    for position in manager_positions:

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

                "Entry Date": position.get(
                    "entry_date",
                    "",
                ),

                "Entry Price": safe_float(
                    position.get(
                        "entry_price",
                        0.0,
                    )
                ),

                "Current Price": safe_float(
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

    positions_frame = pd.DataFrame(
        position_rows
    )

    st.dataframe(
        positions_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Entry Price": (
                st.column_config.NumberColumn(
                    format="$%.2f"
                )
            ),

            "Current Price": (
                st.column_config.NumberColumn(
                    format="$%.2f"
                )
            ),

            "Unrealized P/L": (
                st.column_config.NumberColumn(
                    format="$%.2f"
                )
            ),

            "Unrealized %": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),

            "Stop": (
                st.column_config.NumberColumn(
                    format="$%.2f"
                )
            ),

            "Target": (
                st.column_config.NumberColumn(
                    format="$%.2f"
                )
            ),
        },
    )

else:

    st.info(
        "No open shadow positions."
    )


if (
    isinstance(
        position_risk,
        pd.DataFrame,
    )
    and not position_risk.empty
):

    with st.expander(
        "View Position Risk Detail"
    ):

        st.dataframe(
            position_risk,
            width="stretch",
            hide_index=True,
        )


st.divider()


# ============================================================
# SCANNER LEADERS
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

scanner_left, scanner_right = st.columns(
    2
)

with scanner_left:

    st.subheader(
        "Top Longs"
    )

    if isinstance(
        top_longs,
        list,
    ) and top_longs:

        long_frame = pd.DataFrame(
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
            if column
            in long_frame.columns
        ]

        st.dataframe(
            (
                long_frame[
                    wanted
                ].head(
                    10
                )
                if wanted
                else long_frame.head(
                    10
                )
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No long candidates."
        )


with scanner_right:

    st.subheader(
        "Top Shorts"
    )

    if isinstance(
        top_shorts,
        list,
    ) and top_shorts:

        short_frame = pd.DataFrame(
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
            if column
            in short_frame.columns
        ]

        st.dataframe(
            (
                short_frame[
                    wanted
                ].head(
                    10
                )
                if wanted
                else short_frame.head(
                    10
                )
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No short candidates."
        )


st.divider()


# ============================================================
# PERFORMANCE ANALYTICS
# ============================================================

st.header(
    "Performance Analytics"
)

analytics_columns = st.columns(
    6
)

with analytics_columns[0]:

    st.metric(
        "Analytics Status",
        text(
            performance_analytics.get(
                "analytics_status",
                "UNKNOWN",
            )
        ),
    )

with analytics_columns[1]:

    st.metric(
        "Closed Trades",
        safe_int(
            performance_analytics.get(
                "closed_position_count",
                0,
            )
        ),
    )

with analytics_columns[2]:

    st.metric(
        "Win Rate",
        percent(
            performance_analytics.get(
                "win_rate_percent",
                0.0,
            )
        ),
    )

with analytics_columns[3]:

    st.metric(
        "Profit Factor",
        (
            f"{safe_float(performance_analytics.get('profit_factor', 0.0)):.2f}"
        ),
    )

with analytics_columns[4]:

    st.metric(
        "Max Drawdown",
        percent(
            performance_analytics.get(
                "maximum_drawdown_percent",
                0.0,
            )
        ),
    )

with analytics_columns[5]:

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
    and "equity"
    in equity_curve.columns
):

    chart_frame = (
        equity_curve.copy()
    )

    chart_frame[
        "equity"
    ] = pd.to_numeric(
        chart_frame[
            "equity"
        ],
        errors="coerce",
    )

    chart_frame = (
        chart_frame.dropna(
            subset=[
                "equity"
            ]
        )
    )

    if (
        "date"
        in chart_frame.columns
    ):

        chart_frame[
            "date"
        ] = (
            chart_frame[
                "date"
            ].astype(
                str
            )
        )

        chart_frame = (
            chart_frame.set_index(
                "date"
            )
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
        "Equity curve will become more useful "
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
                15
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
                15
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

top_strategies = hall_of_fame.get(
    "top_strategies",
    [],
)

hall_columns = st.columns(
    3
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
        "Records Imported",
        safe_int(
            hall_of_fame.get(
                "records_imported",
                0,
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


if isinstance(
    top_strategies,
    list,
) and top_strategies:

    hall_frame = pd.DataFrame(
        top_strategies
    )

    wanted = [
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
        if column
        in hall_frame.columns
    ]

    st.dataframe(
        (
            hall_frame[
                wanted
            ].head(
                20
            )
            if wanted
            else hall_frame.head(
                20
            )
        ),
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No Hall of Fame entries available."
    )


st.divider()


# ============================================================
# SHADOW OBSERVATION
# ============================================================

st.header(
    "Shadow Observation"
)

target_days = safe_int(
    observation_state.get(
        "target_days",
        8,
    ),
    8,
)

completed_days = safe_int(
    observation_state.get(
        "completed_days",
        0,
    )
)

progress = (
    min(
        max(
            completed_days
            / target_days,
            0.0,
        ),
        1.0,
    )
    if target_days > 0
    else 0.0
)

st.progress(
    progress,
    text=(
        f"{completed_days} of "
        f"{target_days} successful observation days"
    ),
)

observation_columns = st.columns(
    4
)

with observation_columns[0]:

    st.metric(
        "Completed",
        completed_days,
    )

with observation_columns[1]:

    st.metric(
        "Target",
        target_days,
    )

with observation_columns[2]:

    st.metric(
        "Status",
        text(
            observation_state.get(
                "status",
                "UNKNOWN",
            )
        ),
    )

with observation_columns[3]:

    st.metric(
        "Failed Dates",
        len(
            observation_state.get(
                "failed_dates",
                [],
            )
        )
        if isinstance(
            observation_state.get(
                "failed_dates",
                [],
            ),
            list,
        )
        else 0,
    )


with st.expander(
    "Observation Details"
):

    st.write(
        "**Successful dates:**",
        observation_state.get(
            "successful_dates",
            [],
        ),
    )

    st.write(
        "**Failed dates:**",
        observation_state.get(
            "failed_dates",
            [],
        ),
    )

    st.write(
        "**Last run:**",
        observation_state.get(
            "last_run_timestamp",
            "Not available",
        ),
    )


st.divider()


# ============================================================
# FINAL SAFETY PANEL
# ============================================================

st.header(
    "Platform Safety"
)

safety_rows = [
    {
        "Safety Check": (
            "Shadow Mode"
        ),
        "Value": yes_no(
            safety.get(
                "shadow_mode",
                True,
            )
        ),
    },

    {
        "Safety Check": (
            "Position Manager Apply Used"
        ),
        "Value": yes_no(
            safety.get(
                "position_manager_apply_used",
                False,
            )
        ),
    },

    {
        "Safety Check": (
            "Trading Client Created"
        ),
        "Value": yes_no(
            safety.get(
                "trading_client_created",
                False,
            )
        ),
    },

    {
        "Safety Check": (
            "Order Submitted"
        ),
        "Value": yes_no(
            safety.get(
                "order_submitted",
                False,
            )
        ),
    },

    {
        "Safety Check": (
            "Execution Preview Only"
        ),
        "Value": yes_no(
            trade_execution.get(
                "preview_only",
                True,
            )
        ),
    },

    {
        "Safety Check": (
            "Execution Ledger Modified"
        ),
        "Value": yes_no(
            trade_execution.get(
                "trade_ledger_modified",
                False,
            )
        ),
    },
]

safety_frame = pd.DataFrame(
    safety_rows
)

safety_frame[
    "Value"
] = (
    safety_frame[
        "Value"
    ].astype(
        str
    )
)

st.dataframe(
    safety_frame,
    width="stretch",
    hide_index=True,
)


safe_operation = (
    bool(
        safety.get(
            "shadow_mode",
            True,
        )
    )
    and not bool(
        safety.get(
            "position_manager_apply_used",
            False,
        )
    )
    and not bool(
        safety.get(
            "trading_client_created",
            False,
        )
    )
    and not bool(
        safety.get(
            "order_submitted",
            False,
        )
    )
)


if safe_operation:

    st.success(
        "Safety checks passed: shadow mode is active, "
        "Position Manager apply mode was not used, "
        "no trading client was created, "
        "and no order was submitted."
    )

else:

    st.error(
        "One or more platform safety conditions require review."
    )


# ============================================================
# RAW REPORT INSPECTOR
# ============================================================

st.divider()

st.header(
    "Report Inspector"
)

report_options = {

    "Master Controller": master,

    "Market Regime": market_regime,

    "Portfolio Commander": portfolio,

    "Risk Manager": risk_manager,

    "Trade Execution Manager": trade_execution,

    "Position Manager": position_manager,

    "Performance Analytics": performance_analytics,

    "Scanner": scanner,

    "Hall of Fame": hall_of_fame,

    "Shadow Controller": shadow_controller,

    "Observation State": observation_state,
}

selected_report_name = st.selectbox(
    "Choose a report",
    list(
        report_options.keys()
    ),
)

with st.expander(
    f"View {selected_report_name}",
    expanded=False,
):

    st.json(
        report_options[
            selected_report_name
        ]
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.success(
    "Dashboard v4 loaded successfully."
)

st.caption(
    "AI Trading Platform Dashboard v4 • "
    "Research and shadow monitoring only • "
    "No broker orders are submitted from this dashboard."
)