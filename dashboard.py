"""
Local performance dashboard for the Gold AI Trading Bot.

Run with:
    python -m streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    BACKTEST_REPORT_PATH,
    DECISION_LOG_PATH,
    MODEL_REPORT_PATH,
    TRADE_LOG_PATH,
)


LEGACY_SIGNAL_PATH = Path("reports/legacy_signals.csv")


st.set_page_config(
    page_title="Gold AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
)


def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Read a CSV file without crashing when the file is missing,
    empty, or temporarily malformed.
    """

    if not path.exists():
        return pd.DataFrame()

    try:
        frame = pd.read_csv(path)

        if "timestamp_utc" in frame.columns:
            frame["timestamp_utc"] = pd.to_datetime(
                frame["timestamp_utc"],
                errors="coerce",
                utc=True,
            )

        return frame

    except Exception as error:
        st.warning(
            f"Could not read {path}: {error}"
        )

        return pd.DataFrame()


def read_json_safely(path: Path) -> dict:
    """
    Read a JSON file safely.
    """

    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception as error:
        st.warning(
            f"Could not read {path}: {error}"
        )

        return {}


def format_percent(
    value,
    digits: int = 2,
) -> str:
    """
    Convert a decimal value such as 0.125 into 12.50%.
    """

    try:
        return f"{float(value):.{digits}%}"

    except (TypeError, ValueError):
        return "N/A"


def format_currency(value) -> str:
    """
    Format a value as US currency.
    """

    try:
        return f"${float(value):,.2f}"

    except (TypeError, ValueError):
        return "N/A"


def latest_value(
    frame: pd.DataFrame,
    column: str,
    default="N/A",
):
    """
    Return the latest non-null value from a dataframe column.
    """

    if frame.empty or column not in frame.columns:
        return default

    values = frame[column].dropna()

    if values.empty:
        return default

    return values.iloc[-1]


def calculate_trade_statistics(
    trades: pd.DataFrame,
) -> dict:
    """
    Calculate basic statistics from the available trade log.

    Realized profit statistics will become more detailed after
    explicit entry/exit pairing is added to the logger.
    """

    if trades.empty:
        return {
            "trade_events": 0,
            "buys": 0,
            "sells": 0,
        }

    actions = (
        trades.get(
            "action",
            pd.Series(dtype=str),
        )
        .astype(str)
        .str.upper()
    )

    return {
        "trade_events": len(trades),
        "buys": int((actions == "BUY").sum()),
        "sells": int((actions == "SELL").sum()),
    }


def show_status_message(
    title: str,
    value: str,
) -> None:
    """
    Display a readable status block.
    """

    st.markdown(f"**{title}**")

    st.code(
        value,
        language=None,
    )


decisions = read_csv_safely(
    Path(DECISION_LOG_PATH)
)

trades = read_csv_safely(
    Path(TRADE_LOG_PATH)
)

legacy_signals = read_csv_safely(
    LEGACY_SIGNAL_PATH
)

model_report = read_json_safely(
    Path(MODEL_REPORT_PATH)
)

backtest_report = read_json_safely(
    Path(BACKTEST_REPORT_PATH)
)

trade_statistics = calculate_trade_statistics(
    trades
)


st.title("Gold AI Trading Dashboard")

st.caption(
    "Paper-trading monitoring and legacy-versus-v3 comparison."
)


# ============================================================
# CURRENT STATUS
# ============================================================

st.header("Current status")

latest_v3_action = str(
    latest_value(
        decisions,
        "action",
    )
)

latest_v3_probability = latest_value(
    decisions,
    "probability_up",
    None,
)

latest_v3_price = latest_value(
    decisions,
    "price",
    None,
)

latest_legacy_action = str(
    latest_value(
        legacy_signals,
        "hypothetical_action",
    )
)

latest_legacy_probability = latest_value(
    legacy_signals,
    "probability_up",
    None,
)

latest_legacy_price = latest_value(
    legacy_signals,
    "price",
    None,
)


column_1, column_2, column_3, column_4 = (
    st.columns(4)
)

with column_1:
    st.metric(
        "V3 latest action",
        latest_v3_action,
    )

with column_2:
    st.metric(
        "V3 probability up",
        format_percent(
            latest_v3_probability,
            1,
        ),
    )

with column_3:
    st.metric(
        "Legacy latest action",
        latest_legacy_action,
    )

with column_4:
    st.metric(
        "Legacy probability up",
        format_percent(
            latest_legacy_probability,
            1,
        ),
    )


price_column_1, price_column_2, event_column = (
    st.columns(3)
)

with price_column_1:
    st.metric(
        "Latest V3 GLD price",
        format_currency(
            latest_v3_price
        ),
    )

with price_column_2:
    st.metric(
        "Latest legacy GLD price",
        format_currency(
            latest_legacy_price
        ),
    )

with event_column:
    st.metric(
        "Recorded trade events",
        trade_statistics["trade_events"],
    )


# ============================================================
# MODEL VALIDATION
# ============================================================

st.header("Current model validation")

candidate_metrics = model_report.get(
    "candidate_metrics",
    {},
)

model_column_1, model_column_2, model_column_3, model_column_4 = (
    st.columns(4)
)

with model_column_1:
    st.metric(
        "Model status",
        str(
            model_report.get(
                "status",
                "N/A",
            )
        ).upper(),
    )

with model_column_2:
    st.metric(
        "Accuracy",
        format_percent(
            candidate_metrics.get(
                "accuracy"
            )
        ),
    )

with model_column_3:
    st.metric(
        "ROC-AUC",
        (
            f"{candidate_metrics.get('roc_auc'):.3f}"
            if isinstance(
                candidate_metrics.get(
                    "roc_auc"
                ),
                (int, float),
            )
            else "N/A"
        ),
    )

with model_column_4:
    st.metric(
        "F1 score",
        format_percent(
            candidate_metrics.get(
                "f1"
            )
        ),
    )


# ============================================================
# BACKTEST RESULTS
# ============================================================

st.header("Latest backtest")

backtest_column_1, backtest_column_2, backtest_column_3, backtest_column_4 = (
    st.columns(4)
)

with backtest_column_1:
    st.metric(
        "Total return",
        format_percent(
            backtest_report.get(
                "total_return"
            )
        ),
    )

with backtest_column_2:
    st.metric(
        "Final value",
        format_currency(
            backtest_report.get(
                "final_value"
            )
        ),
    )

with backtest_column_3:
    st.metric(
        "Win rate",
        format_percent(
            backtest_report.get(
                "win_rate"
            )
        ),
    )

with backtest_column_4:
    st.metric(
        "Maximum drawdown",
        format_percent(
            backtest_report.get(
                "maximum_drawdown"
            )
        ),
    )


# ============================================================
# SIGNAL COMPARISON
# ============================================================

st.header("Signal history")

comparison_frames = []

if not decisions.empty:
    v3_columns = [
        column
        for column in [
            "timestamp_utc",
            "price",
            "probability_up",
            "action",
            "reason",
        ]
        if column in decisions.columns
    ]

    v3_history = decisions[
        v3_columns
    ].copy()

    v3_history["bot"] = "V3"

    comparison_frames.append(
        v3_history
    )

if not legacy_signals.empty:
    legacy_columns = [
        column
        for column in [
            "timestamp_utc",
            "price",
            "probability_up",
            "hypothetical_action",
        ]
        if column in legacy_signals.columns
    ]

    legacy_history = legacy_signals[
        legacy_columns
    ].copy()

    legacy_history = legacy_history.rename(
        columns={
            "hypothetical_action":
                "action",
        }
    )

    legacy_history["reason"] = (
        "Read-only legacy signal"
    )

    legacy_history["bot"] = "Legacy"

    comparison_frames.append(
        legacy_history
    )

if comparison_frames:
    comparison = pd.concat(
        comparison_frames,
        ignore_index=True,
        sort=False,
    )

    if "timestamp_utc" in comparison.columns:
        comparison = comparison.sort_values(
            "timestamp_utc",
            ascending=False,
        )

    display_columns = [
        column
        for column in [
            "timestamp_utc",
            "bot",
            "price",
            "probability_up",
            "action",
            "reason",
        ]
        if column in comparison.columns
    ]

    st.dataframe(
        comparison[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No signal history has been recorded yet."
    )


# ============================================================
# CONFIDENCE CHART
# ============================================================

st.header("Probability-up history")

chart_frames = []

if (
    not decisions.empty
    and "timestamp_utc" in decisions.columns
    and "probability_up" in decisions.columns
):
    chart_frames.append(
        decisions[
            [
                "timestamp_utc",
                "probability_up",
            ]
        ]
        .assign(bot="V3")
    )

if (
    not legacy_signals.empty
    and "timestamp_utc" in legacy_signals.columns
    and "probability_up" in legacy_signals.columns
):
    chart_frames.append(
        legacy_signals[
            [
                "timestamp_utc",
                "probability_up",
            ]
        ]
        .assign(bot="Legacy")
    )

if chart_frames:
    chart_data = pd.concat(
        chart_frames,
        ignore_index=True,
    )

    chart_data = chart_data.dropna(
        subset=[
            "timestamp_utc",
            "probability_up",
        ]
    )

    chart_data["probability_up"] = (
        pd.to_numeric(
            chart_data[
                "probability_up"
            ],
            errors="coerce",
        )
    )

    chart_data = chart_data.dropna(
        subset=[
            "probability_up",
        ]
    )

    pivoted = chart_data.pivot_table(
        index="timestamp_utc",
        columns="bot",
        values="probability_up",
        aggfunc="last",
    )

    st.line_chart(
        pivoted,
        use_container_width=True,
    )

else:
    st.info(
        "More daily signals are needed before "
        "the confidence chart can be displayed."
    )


# ============================================================
# RECENT TRADE EVENTS
# ============================================================

st.header("Recent paper-trade events")

if not trades.empty:
    recent_trades = trades.sort_values(
        "timestamp_utc",
        ascending=False,
    )

    st.dataframe(
        recent_trades,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No v3 paper-trade events have been logged yet."
    )


st.caption(
    "This dashboard is for monitoring and research. "
    "It does not establish that the strategy is profitable."
)