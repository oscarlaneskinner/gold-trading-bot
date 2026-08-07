"""
AI Trading Platform Dashboard v3

Research-only dashboard for the gold-trading-bot project.

This dashboard:
- reads existing research reports,
- displays portfolio and performance information,
- shows open shadow positions,
- does not change the trade ledger,
- does not create a broker client,
- does not submit orders.
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
    page_title="AI Trading Platform Dashboard v3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

PERFORMANCE_DIRECTORY = REPORTS_DIRECTORY / "performance"
POSITION_MANAGER_DIRECTORY = REPORTS_DIRECTORY / "position_manager"
POSITION_VALUATION_DIRECTORY = REPORTS_DIRECTORY / "position_valuation"
MARKET_REGIME_DIRECTORY = REPORTS_DIRECTORY / "market_regime"
PORTFOLIO_DIRECTORY = REPORTS_DIRECTORY / "portfolio"
SCANNER_DIRECTORY = REPORTS_DIRECTORY / "scanner"
SHADOW_DIRECTORY = REPORTS_DIRECTORY / "shadow"
SHADOW_OBSERVATION_DIRECTORY = REPORTS_DIRECTORY / "shadow_observation"
HALL_OF_FAME_DIRECTORY = REPORTS_DIRECTORY / "hall_of_fame"

PERFORMANCE_SUMMARY_PATH = PERFORMANCE_DIRECTORY / "latest_summary.json"
TRADE_LEDGER_PATH = PERFORMANCE_DIRECTORY / "trade_ledger.json"
EQUITY_CURVE_PATH = PERFORMANCE_DIRECTORY / "equity_curve.csv"

POSITION_MANAGER_SUMMARY_PATH = POSITION_MANAGER_DIRECTORY / "latest_summary.json"
POSITION_MANAGER_OPEN_POSITIONS_PATH = POSITION_MANAGER_DIRECTORY / "open_positions.csv"
POSITION_MANAGER_CLOSED_TODAY_PATH = POSITION_MANAGER_DIRECTORY / "closed_today.csv"

POSITION_VALUATION_SUMMARY_PATH = POSITION_VALUATION_DIRECTORY / "latest_summary.json"

MARKET_REGIME_PATH = MARKET_REGIME_DIRECTORY / "market_regime_lab_v1.json"

PORTFOLIO_COMMANDER_PATH = PORTFOLIO_DIRECTORY / "portfolio_commander_v1.json"
PORTFOLIO_ALLOCATIONS_PATH = PORTFOLIO_DIRECTORY / "portfolio_allocations.csv"

SCANNER_SUMMARY_PATH = SCANNER_DIRECTORY / "championship_scanner_v1.json"
TOP_LONGS_PATH = SCANNER_DIRECTORY / "top_longs.csv"
TOP_SHORTS_PATH = SCANNER_DIRECTORY / "top_shorts.csv"

SHADOW_CONTROLLER_PATH = SHADOW_DIRECTORY / "two_bot_shadow_controller_v1.json"
SHADOW_PROPOSALS_PATH = SHADOW_DIRECTORY / "two_bot_shadow_proposals.csv"

SHADOW_OBSERVATION_PATH = SHADOW_OBSERVATION_DIRECTORY / "latest_summary.json"
OBSERVATION_STATE_PATH = SHADOW_OBSERVATION_DIRECTORY / "observation_state.json"

HALL_OF_FAME_PATH = HALL_OF_FAME_DIRECTORY / "strategy_hall_of_fame.json"
HALL_OF_FAME_CSV_PATH = HALL_OF_FAME_DIRECTORY / "strategy_hall_of_fame.csv"

PLATFORM_CONTROLLER_DIRECTORY = REPORTS_DIRECTORY / "platform_controller"
PLATFORM_CONTROLLER_SUMMARY_PATH = (
    PLATFORM_CONTROLLER_DIRECTORY / "latest_summary.json"
)


# ============================================================
# SAFE FILE-LOADING FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_json_file(path_text: str, default: Any) -> Any:
    path = Path(path_text)

    if not path.exists():
        return default

    try:
        raw_text = path.read_text(encoding="utf-8").strip()

        if not raw_text:
            return default

        return json.loads(raw_text)

    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return default


@st.cache_data(show_spinner=False)
def load_csv_file(path_text: str) -> pd.DataFrame:
    path = Path(path_text)

    if not path.exists():
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path)

        if frame.empty:
            return pd.DataFrame()

        return frame

    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame()


def file_last_modified(path: Path) -> str:
    if not path.exists():
        return "Not available"

    modified_timestamp = path.stat().st_mtime
    modified_datetime = datetime.fromtimestamp(modified_timestamp)

    return modified_datetime.strftime("%Y-%m-%d %I:%M:%S %p")


def clear_dashboard_cache() -> None:
    st.cache_data.clear()


# ============================================================
# FORMATTING HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def money(value: Any) -> str:
    return f"${safe_float(value):,.2f}"


def percent(value: Any) -> str:
    return f"{safe_float(value):,.2f}%"


def number(value: Any, decimals: int = 2) -> str:
    return f"{safe_float(value):,.{decimals}f}"


def readable_boolean(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def get_nested(
    payload: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = payload

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def report_status(path: Path, payload: Any) -> str:
    if not path.exists():
        return "Missing"

    if payload in ({}, [], None):
        return "Empty"

    return "Available"


# ============================================================
# LOAD ALL DASHBOARD DATA
# ============================================================

def load_dashboard_data() -> dict[str, Any]:
    performance_summary = load_json_file(
        str(PERFORMANCE_SUMMARY_PATH),
        {},
    )
    trade_ledger = load_json_file(
        str(TRADE_LEDGER_PATH),
        [],
    )
    position_manager_summary = load_json_file(
        str(POSITION_MANAGER_SUMMARY_PATH),
        {},
    )
    position_valuation_summary = load_json_file(
        str(POSITION_VALUATION_SUMMARY_PATH),
        {},
    )
    market_regime = load_json_file(
        str(MARKET_REGIME_PATH),
        {},
    )
    portfolio_commander = load_json_file(
        str(PORTFOLIO_COMMANDER_PATH),
        {},
    )
    scanner_summary = load_json_file(
        str(SCANNER_SUMMARY_PATH),
        {},
    )
    shadow_controller = load_json_file(
        str(SHADOW_CONTROLLER_PATH),
        {},
    )
    shadow_observation = load_json_file(
        str(SHADOW_OBSERVATION_PATH),
        {},
    )
    observation_state = load_json_file(
        str(OBSERVATION_STATE_PATH),
        {},
    )
    hall_of_fame = load_json_file(
        str(HALL_OF_FAME_PATH),
        {},
    )
    platform_controller = load_json_file(
        str(PLATFORM_CONTROLLER_SUMMARY_PATH),
        {},
    )

    equity_curve = load_csv_file(str(EQUITY_CURVE_PATH))
    position_manager_open_positions = load_csv_file(
        str(POSITION_MANAGER_OPEN_POSITIONS_PATH)
    )
    position_manager_closed_today = load_csv_file(
        str(POSITION_MANAGER_CLOSED_TODAY_PATH)
    )
    portfolio_allocations = load_csv_file(
        str(PORTFOLIO_ALLOCATIONS_PATH)
    )
    top_longs = load_csv_file(str(TOP_LONGS_PATH))
    top_shorts = load_csv_file(str(TOP_SHORTS_PATH))
    shadow_proposals = load_csv_file(str(SHADOW_PROPOSALS_PATH))
    hall_of_fame_table = load_csv_file(str(HALL_OF_FAME_CSV_PATH))

    return {
        "performance_summary": performance_summary,
        "trade_ledger": trade_ledger,
        "position_manager_summary": position_manager_summary,
        "position_valuation_summary": position_valuation_summary,
        "market_regime": market_regime,
        "portfolio_commander": portfolio_commander,
        "scanner_summary": scanner_summary,
        "shadow_controller": shadow_controller,
        "shadow_observation": shadow_observation,
        "observation_state": observation_state,
        "hall_of_fame": hall_of_fame,
        "platform_controller": platform_controller,
        "equity_curve": equity_curve,
        "position_manager_open_positions": position_manager_open_positions,
        "position_manager_closed_today": position_manager_closed_today,
        "portfolio_allocations": portfolio_allocations,
        "top_longs": top_longs,
        "top_shorts": top_shorts,
        "shadow_proposals": shadow_proposals,
        "hall_of_fame_table": hall_of_fame_table,
    }


dashboard_data = load_dashboard_data()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📈 Trading Platform")
    st.caption("Research and shadow-trading dashboard")
    st.divider()

    if st.button("Refresh dashboard", width="stretch"):
        clear_dashboard_cache()
        st.rerun()

    st.subheader("Operating Mode")
    st.success("Shadow research mode")

    st.write(
        "**Trading client created:** "
        + readable_boolean(
            get_nested(
                dashboard_data["position_manager_summary"],
                "trading_client_created",
                default=False,
            )
        )
    )

    st.write(
        "**Order submitted:** "
        + readable_boolean(
            get_nested(
                dashboard_data["position_manager_summary"],
                "order_submitted",
                default=False,
            )
        )
    )

    st.write(
        "**Ledger modified:** "
        + readable_boolean(
            get_nested(
                dashboard_data["position_manager_summary"],
                "ledger_modified",
                default=False,
            )
        )
    )

    st.divider()
    st.subheader("Latest Report Times")

    st.caption(
        "Performance: " + file_last_modified(PERFORMANCE_SUMMARY_PATH)
    )
    st.caption(
        "Position Manager: "
        + file_last_modified(POSITION_MANAGER_SUMMARY_PATH)
    )
    st.caption(
        "Market Regime: " + file_last_modified(MARKET_REGIME_PATH)
    )
    st.caption(
        "Scanner: " + file_last_modified(SCANNER_SUMMARY_PATH)
    )
    st.caption(
        "Shadow Observation: "
        + file_last_modified(SHADOW_OBSERVATION_PATH)
    )

    st.divider()
    st.caption(
        "This dashboard reads local research reports. "
        "It does not place trades."
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title("AI Trading Platform Dashboard v3")
st.caption(
    "Unified research, shadow-position, strategy, scanner, "
    "and performance monitoring"
)
st.info(
    "Dashboard v3 is running in research-only mode. "
    "No broker orders are submitted from this page."
)


# ============================================================
# BASIC SYSTEM STATUS
# ============================================================

st.subheader("System Status")
status_columns = st.columns(5)

with status_columns[0]:
    st.metric(
        "Performance Report",
        report_status(
            PERFORMANCE_SUMMARY_PATH,
            dashboard_data["performance_summary"],
        ),
    )

with status_columns[1]:
    st.metric(
        "Position Manager",
        report_status(
            POSITION_MANAGER_SUMMARY_PATH,
            dashboard_data["position_manager_summary"],
        ),
    )

with status_columns[2]:
    st.metric(
        "Market Regime",
        report_status(
            MARKET_REGIME_PATH,
            dashboard_data["market_regime"],
        ),
    )

with status_columns[3]:
    st.metric(
        "Scanner",
        report_status(
            SCANNER_SUMMARY_PATH,
            dashboard_data["scanner_summary"],
        ),
    )

with status_columns[4]:
    st.metric(
        "Hall of Fame",
        report_status(
            HALL_OF_FAME_PATH,
            dashboard_data["hall_of_fame"],
        ),
    )

st.divider()


# ============================================================
# PORTFOLIO AND PERFORMANCE OVERVIEW
# ============================================================

st.subheader("Portfolio Overview")

performance_summary = dashboard_data["performance_summary"]
position_manager_summary = dashboard_data["position_manager_summary"]
position_valuation_summary = dashboard_data["position_valuation_summary"]
portfolio_commander = dashboard_data["portfolio_commander"]
trade_ledger = dashboard_data["trade_ledger"]

starting_capital = safe_float(
    position_manager_summary.get(
        "starting_capital",
        performance_summary.get("starting_capital", 2000.0),
    ),
    default=2000.0,
)

realized_pnl = safe_float(
    position_manager_summary.get(
        "realized_pnl_dollars",
        performance_summary.get("total_pnl_dollars", 0.0),
    )
)

unrealized_pnl = safe_float(
    position_manager_summary.get(
        "unrealized_pnl_dollars",
        position_valuation_summary.get("unrealized_pnl_dollars", 0.0),
    )
)

estimated_equity = safe_float(
    position_manager_summary.get(
        "estimated_equity",
        position_valuation_summary.get(
            "estimated_equity",
            performance_summary.get("ending_equity", starting_capital),
        ),
    ),
    default=starting_capital,
)

estimated_return_percent = safe_float(
    position_manager_summary.get(
        "estimated_total_return_percent",
        position_valuation_summary.get(
            "estimated_total_return_percent",
            performance_summary.get("total_return_percent", 0.0),
        ),
    )
)

open_position_count = safe_int(
    position_manager_summary.get(
        "open_position_count",
        performance_summary.get("open_positions", 0),
    )
)

closed_position_count = safe_int(
    position_manager_summary.get(
        "closed_position_count",
        performance_summary.get("closed_positions", 0),
    )
)

cash_reserve = safe_float(
    portfolio_commander.get(
        "cash_reserve_dollars",
        max(
            starting_capital
            - safe_float(
                portfolio_commander.get("allocated_percent", 0.0)
            )
            / 100
            * starting_capital,
            0.0,
        ),
    )
)

portfolio_metric_columns = st.columns(5)

with portfolio_metric_columns[0]:
    st.metric(
        label="Estimated Equity",
        value=money(estimated_equity),
        delta=money(estimated_equity - starting_capital),
    )

with portfolio_metric_columns[1]:
    st.metric(
        label="Starting Capital",
        value=money(starting_capital),
    )

with portfolio_metric_columns[2]:
    st.metric(
        label="Unrealized P/L",
        value=money(unrealized_pnl),
        delta=percent(
            (unrealized_pnl / starting_capital * 100)
            if starting_capital > 0
            else 0.0
        ),
    )

with portfolio_metric_columns[3]:
    st.metric(
        label="Realized P/L",
        value=money(realized_pnl),
    )

with portfolio_metric_columns[4]:
    st.metric(
        label="Estimated Return",
        value=percent(estimated_return_percent),
    )

secondary_metric_columns = st.columns(5)

with secondary_metric_columns[0]:
    st.metric(
        label="Open Positions",
        value=open_position_count,
    )

with secondary_metric_columns[1]:
    st.metric(
        label="Closed Positions",
        value=closed_position_count,
    )

with secondary_metric_columns[2]:
    st.metric(
        label="Cash Reserve",
        value=money(cash_reserve),
    )

with secondary_metric_columns[3]:
    st.metric(
        label="Win Rate",
        value=percent(
            performance_summary.get("win_rate_percent", 0.0)
        ),
    )

with secondary_metric_columns[4]:
    st.metric(
        label="Profit Factor",
        value=number(
            performance_summary.get("profit_factor", 0.0),
            decimals=2,
        ),
    )

st.divider()


# ============================================================
# PERFORMANCE DETAILS
# ============================================================

left_performance_column, right_performance_column = st.columns([1, 1])

with left_performance_column:
    st.subheader("Performance Statistics")

    performance_rows = [
        {
            "Metric": "Wins",
            "Value": str(
                safe_int(performance_summary.get("wins", 0))
            ),
        },
        {
            "Metric": "Losses",
            "Value": str(
                safe_int(performance_summary.get("losses", 0))
            ),
        },
        {
            "Metric": "Breakeven Trades",
            "Value": str(
                safe_int(
                    performance_summary.get("breakeven_trades", 0)
                )
            ),
        },
        {
            "Metric": "Average Winner",
            "Value": percent(
                performance_summary.get("average_winner_percent", 0.0)
            ),
        },
        {
            "Metric": "Average Loser",
            "Value": percent(
                performance_summary.get("average_loser_percent", 0.0)
            ),
        },
        {
            "Metric": "Expectancy",
            "Value": percent(
                performance_summary.get("expectancy_percent", 0.0)
            ),
        },
        {
            "Metric": "Largest Winner",
            "Value": percent(
                performance_summary.get("largest_winner_percent", 0.0)
            ),
        },
        {
            "Metric": "Largest Loser",
            "Value": percent(
                performance_summary.get("largest_loser_percent", 0.0)
            ),
        },
        {
            "Metric": "Average Holding Days",
            "Value": number(
                performance_summary.get("average_holding_days", 0.0),
                decimals=1,
            ),
        },
        {
            "Metric": "Maximum Drawdown",
            "Value": percent(
                performance_summary.get("maximum_drawdown_percent", 0.0)
            ),
        },
    ]

    performance_table = pd.DataFrame(performance_rows)
    performance_table["Value"] = performance_table["Value"].astype(str)

    st.dataframe(
        performance_table,
        width="stretch",
        hide_index=True,
    )

with right_performance_column:
    st.subheader("Position Manager Status")

    manager_mode = normalize_text(
        position_manager_summary.get("mode", "Not available")
    )

    if manager_mode == "PREVIEW_ONLY":
        st.info("Position Manager is operating in preview-only mode.")
    elif manager_mode == "APPLY_SHADOW_LEDGER_CHANGES":
        st.warning(
            "Position Manager is allowed to update the shadow ledger."
        )
    else:
        st.warning("Position Manager mode is not available.")

    position_manager_rows = [
        {
            "Item": "Report Date",
            "Value": str(
                position_manager_summary.get(
                    "report_date",
                    "Not available",
                )
            ),
        },
        {
            "Item": "Mode",
            "Value": str(manager_mode),
        },
        {
            "Item": "Created Today",
            "Value": str(
                safe_int(
                    position_manager_summary.get(
                        "created_today_count",
                        0,
                    )
                )
            ),
        },
        {
            "Item": "Closed Today",
            "Value": str(
                safe_int(
                    position_manager_summary.get(
                        "closed_today_count",
                        0,
                    )
                )
            ),
        },
        {
            "Item": "Duplicates Skipped",
            "Value": str(
                safe_int(
                    position_manager_summary.get(
                        "duplicate_proposals_skipped",
                        0,
                    )
                )
            ),
        },
        {
            "Item": "Invalid Proposals Skipped",
            "Value": str(
                safe_int(
                    position_manager_summary.get(
                        "invalid_proposals_skipped",
                        0,
                    )
                )
            ),
        },
        {
            "Item": "Ledger Modified",
            "Value": readable_boolean(
                position_manager_summary.get(
                    "ledger_modified",
                    False,
                )
            ),
        },
        {
            "Item": "Order Submitted",
            "Value": readable_boolean(
                position_manager_summary.get(
                    "order_submitted",
                    False,
                )
            ),
        },
    ]

    position_manager_frame = pd.DataFrame(position_manager_rows)
    position_manager_frame["Value"] = (
        position_manager_frame["Value"].astype(str)
    )

    st.dataframe(
        position_manager_frame,
        width="stretch",
        hide_index=True,
    )

st.divider()


# ============================================================
# OPEN POSITIONS FROM THE TRADE LEDGER
# ============================================================

st.subheader("Open Positions")

open_trade_rows: list[dict[str, Any]] = []

if isinstance(trade_ledger, list):
    for trade in trade_ledger:
        if not isinstance(trade, dict):
            continue

        if normalize_text(trade.get("status")).upper() != "OPEN":
            continue

        entry_price = safe_float(trade.get("entry_price"))
        current_price = safe_float(
            trade.get("current_price", entry_price)
        )
        shares = safe_float(trade.get("shares"))
        side = normalize_text(trade.get("side", "LONG")).upper()

        if side == "SHORT":
            calculated_unrealized_pnl = (
                entry_price - current_price
            ) * shares
        else:
            calculated_unrealized_pnl = (
                current_price - entry_price
            ) * shares

        unrealized_trade_pnl = safe_float(
            trade.get(
                "unrealized_pnl_dollars",
                calculated_unrealized_pnl,
            )
        )

        open_trade_rows.append(
            {
                "Symbol": trade.get("symbol", ""),
                "Strategy": trade.get("strategy_name", ""),
                "Side": side,
                "Entry Date": trade.get("entry_date", ""),
                "Entry Price": round(entry_price, 4),
                "Current Price": round(current_price, 4),
                "Shares": round(shares, 6),
                "Allocation": round(
                    safe_float(trade.get("proposed_dollars")),
                    2,
                ),
                "Unrealized P/L": round(unrealized_trade_pnl, 2),
                "Stop": trade.get("stop_price"),
                "Target": trade.get("target_price"),
                "Holding Days": safe_int(
                    trade.get("holding_days", 0)
                ),
                "Status": trade.get("status", ""),
            }
        )

if open_trade_rows:
    open_positions_frame = pd.DataFrame(open_trade_rows)

    st.dataframe(
        open_positions_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Entry Price": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Current Price": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Allocation": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Unrealized P/L": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Stop": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Target": st.column_config.NumberColumn(
                format="$%.2f"
            ),
        },
    )
else:
    st.info("No open shadow positions were found.")

st.divider()


# ============================================================
# EQUITY CURVE
# ============================================================

st.subheader("Equity Curve")
equity_curve = dashboard_data["equity_curve"]

if (
    isinstance(equity_curve, pd.DataFrame)
    and not equity_curve.empty
    and "equity" in equity_curve.columns
):
    chart_frame = equity_curve.copy()

    chart_frame["equity"] = pd.to_numeric(
        chart_frame["equity"],
        errors="coerce",
    )

    chart_frame = chart_frame.dropna(subset=["equity"])

    if "date" in chart_frame.columns:
        chart_frame["date"] = chart_frame["date"].astype(str)
        chart_frame = chart_frame.set_index("date")

    st.line_chart(
        chart_frame[["equity"]],
        width="stretch",
    )
else:
    st.info(
        "The equity curve will appear after "
        "performance history becomes available."
    )

st.divider()
st.success(
    "Dashboard Parts 1 and 2 loaded successfully. "
    "Portfolio statistics, open positions, and performance "
    "details are active."
)


# ============================================================
# PLATFORM CONTROLLER HEALTH
# ============================================================

st.subheader("AI Trading Platform Controller")

platform_controller = dashboard_data.get(
    "platform_controller",
    {},
)

controller_status = normalize_text(
    platform_controller.get(
        "overall_status",
        "NOT AVAILABLE",
    )
).upper()

controller_columns = st.columns(5)

with controller_columns[0]:
    st.metric(
        "Controller Status",
        controller_status,
    )

with controller_columns[1]:
    st.metric(
        "Required Steps Passed",
        (
            f"{safe_int(platform_controller.get('required_steps_passed', 0))}/"
            f"{safe_int(platform_controller.get('required_step_count', 0))}"
        ),
    )

with controller_columns[2]:
    st.metric(
        "Failed Steps",
        safe_int(
            platform_controller.get(
                "failed_step_count",
                0,
            )
        ),
    )

with controller_columns[3]:
    st.metric(
        "Run Duration",
        (
            f"{safe_float(platform_controller.get('duration_seconds', 0.0)):.1f}s"
        ),
    )

with controller_columns[4]:
    safety_payload = platform_controller.get(
        "safety",
        {},
    )
    st.metric(
        "Orders Submitted",
        readable_boolean(
            safety_payload.get(
                "order_submitted",
                False,
            )
        ),
    )

if controller_status == "PASSED":
    st.success(
        "The latest platform-controller run passed all required checks."
    )
elif controller_status == "COMPLETED_WITH_OPTIONAL_FAILURES":
    st.warning(
        "Required controller steps passed, but at least one optional step failed."
    )
elif controller_status == "FAILED":
    st.error(
        "The latest platform-controller run failed a required step."
    )
else:
    st.info(
        "No platform-controller summary is available yet."
    )

step_results = platform_controller.get(
    "step_results",
    [],
)

if isinstance(step_results, list) and step_results:
    step_rows = []

    for result in step_results:
        if not isinstance(result, dict):
            continue

        step_rows.append(
            {
                "Step": str(result.get("name", "")),
                "Status": str(result.get("status", "")),
                "Required": readable_boolean(
                    result.get("required", False)
                ),
                "Return Code": str(
                    result.get("returncode", "")
                ),
                "Duration (s)": round(
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
        pd.DataFrame(step_rows),
        width="stretch",
        hide_index=True,
    )
else:
    st.info(
        "Controller step-level results will appear after the controller runs."
    )

st.divider()


# ============================================================
# MARKET, STRATEGY, AND OBSERVATION SNAPSHOT
# ============================================================

st.subheader("Platform Snapshot")

snapshot = platform_controller.get(
    "platform_snapshot",
    {},
)

snapshot_columns = st.columns(5)

with snapshot_columns[0]:
    st.metric(
        "Market Regime",
        normalize_text(
            snapshot.get(
                "market_regime",
                dashboard_data["market_regime"].get(
                    "regime",
                    "UNKNOWN",
                ),
            )
        ),
    )

with snapshot_columns[1]:
    st.metric(
        "Hall of Fame",
        safe_int(
            snapshot.get(
                "hall_of_fame_strategy_count",
                dashboard_data["hall_of_fame"].get(
                    "strategy_count",
                    0,
                ),
            )
        ),
    )

with snapshot_columns[2]:
    st.metric(
        "Shadow Proposals",
        safe_int(
            snapshot.get(
                "shadow_proposal_count",
                dashboard_data["shadow_controller"].get(
                    "proposal_count",
                    0,
                ),
            )
        ),
    )

with snapshot_columns[3]:
    st.metric(
        "Open Positions",
        safe_int(
            snapshot.get(
                "open_positions",
                open_position_count,
            )
        ),
    )

with snapshot_columns[4]:
    st.metric(
        "Estimated Equity",
        money(
            snapshot.get(
                "estimated_equity",
                estimated_equity,
            )
        ),
    )

observation_state = snapshot.get(
    "observation_state",
    dashboard_data["observation_state"],
)

if not isinstance(observation_state, dict):
    observation_state = {}

target_days = safe_int(
    observation_state.get(
        "target_days",
        8,
    ),
    default=8,
)

completed_days = safe_int(
    observation_state.get(
        "completed_days",
        0,
    )
)

progress_value = (
    min(max(completed_days / target_days, 0.0), 1.0)
    if target_days > 0
    else 0.0
)

st.progress(
    progress_value,
    text=(
        f"Shadow observation: {completed_days} of "
        f"{target_days} successful days"
    ),
)

st.divider()


# ============================================================
# SCANNER LEADERS
# ============================================================

st.subheader("Scanner Leaders")

scanner = dashboard_data["scanner_summary"]

top_longs = scanner.get(
    "top_longs",
    [],
)

top_shorts = scanner.get(
    "top_shorts",
    [],
)

scanner_left, scanner_right = st.columns(2)

with scanner_left:
    st.write("**Top Long Candidates**")

    if isinstance(top_longs, list) and top_longs:
        long_frame = pd.DataFrame(top_longs)

        columns = [
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
            if column in long_frame.columns
        ]

        st.dataframe(
            long_frame[columns].head(10)
            if columns
            else long_frame.head(10),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No long scanner results are available.")

with scanner_right:
    st.write("**Top Short Candidates**")

    if isinstance(top_shorts, list) and top_shorts:
        short_frame = pd.DataFrame(top_shorts)

        columns = [
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
            if column in short_frame.columns
        ]

        st.dataframe(
            short_frame[columns].head(10)
            if columns
            else short_frame.head(10),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No short scanner results are available.")

st.divider()


# ============================================================
# FINAL SAFETY PANEL
# ============================================================

st.subheader("Safety Panel")

safety = platform_controller.get(
    "safety",
    {},
)

safety_rows = [
    {
        "Check": "Shadow Mode",
        "Status": readable_boolean(
            safety.get("shadow_mode", True)
        ),
    },
    {
        "Check": "Position Manager Apply Used",
        "Status": readable_boolean(
            safety.get(
                "position_manager_apply_used",
                False,
            )
        ),
    },
    {
        "Check": "Trading Client Created",
        "Status": readable_boolean(
            safety.get(
                "trading_client_created",
                False,
            )
        ),
    },
    {
        "Check": "Order Submitted",
        "Status": readable_boolean(
            safety.get(
                "order_submitted",
                False,
            )
        ),
    },
]

safety_frame = pd.DataFrame(safety_rows)
safety_frame["Status"] = safety_frame["Status"].astype(str)

st.dataframe(
    safety_frame,
    width="stretch",
    hide_index=True,
)

if (
    not bool(safety.get("position_manager_apply_used", False))
    and not bool(safety.get("trading_client_created", False))
    and not bool(safety.get("order_submitted", False))
):
    st.success(
        "Safety check passed: preview-only operation, "
        "no trading client, and no submitted orders."
    )
else:
    st.error(
        "Review the latest controller safety fields before continuing."
    )

st.caption(
    "AI Trading Platform Dashboard v3 • Research-only monitoring"
)