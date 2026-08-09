"""
Strategy Learning Engine v1

Research-only adaptive strategy evaluation engine for the
gold-trading-bot platform.

The engine combines:
- Closed-trade strategy performance
- Strategy Hall of Fame research results
- Profit factor
- Win rate
- Expectancy
- Drawdown
- Consistency
- Trade count / evidence strength

It produces:
- A learning score from 0 to 100
- Strategy status classifications
- Suggested allocation multipliers
- Promote / hold / reduce / retire research recommendations

Important:
- This version does NOT automatically change Portfolio Commander.
- It does NOT modify the trade ledger.
- It does NOT modify production strategies.
- It does NOT create a broker client.
- It does NOT submit orders.

Run:
    python strategy_learning_engine_v1.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

PERFORMANCE_ANALYTICS_DIRECTORY = (
    REPORTS_DIRECTORY
    / "performance_analytics"
)

HALL_OF_FAME_DIRECTORY = (
    REPORTS_DIRECTORY
    / "hall_of_fame"
)

LEARNING_DIRECTORY = (
    REPORTS_DIRECTORY
    / "strategy_learning"
)

LEARNING_HISTORY_DIRECTORY = (
    LEARNING_DIRECTORY
    / "history"
)

PERFORMANCE_SUMMARY_PATH = (
    PERFORMANCE_ANALYTICS_DIRECTORY
    / "latest_summary.json"
)

STRATEGY_ANALYTICS_PATH = (
    PERFORMANCE_ANALYTICS_DIRECTORY
    / "strategy_analytics.csv"
)

HALL_OF_FAME_JSON_PATH = (
    HALL_OF_FAME_DIRECTORY
    / "strategy_hall_of_fame.json"
)

HALL_OF_FAME_CSV_PATH = (
    HALL_OF_FAME_DIRECTORY
    / "strategy_hall_of_fame.csv"
)

LATEST_SUMMARY_PATH = (
    LEARNING_DIRECTORY
    / "latest_summary.json"
)

LATEST_TEXT_PATH = (
    LEARNING_DIRECTORY
    / "latest_summary.txt"
)

STRATEGY_SCORES_PATH = (
    LEARNING_DIRECTORY
    / "strategy_learning_scores.csv"
)

ALLOCATION_GUIDANCE_PATH = (
    LEARNING_DIRECTORY
    / "allocation_guidance.csv"
)

LEARNING_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

LEARNING_HISTORY_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_LIVE_TRADES_FOR_EARLY_LEARNING = 5
MIN_LIVE_TRADES_FOR_CONFIDENT_LEARNING = 20
MIN_LIVE_TRADES_FOR_RETIREMENT = 20

PROMOTE_SCORE = 75.0
HOLD_SCORE = 55.0
REDUCE_SCORE = 40.0

MIN_ALLOCATION_MULTIPLIER = 0.25
MAX_ALLOCATION_MULTIPLIER = 1.50


# ============================================================
# HELPERS
# ============================================================

def load_json(
    path: Path,
    default: Any,
) -> Any:

    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return default


def load_csv(
    path: Path,
) -> pd.DataFrame:

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


def save_json(
    path: Path,
    payload: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        if pd.isna(
            value
        ):
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

        if pd.isna(
            value
        ):
            return default

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def normalize_strategy_name(
    value: Any,
) -> str:

    result = str(
        value or ""
    ).strip()

    return (
        result
        if result
        else "UNKNOWN"
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


# ============================================================
# COLUMN HELPERS
# ============================================================

def first_existing_value(
    row: pd.Series,
    names: list[str],
    default: Any = None,
) -> Any:

    for name in names:

        if name in row.index:

            value = row[
                name
            ]

            if not pd.isna(
                value
            ):
                return value

    return default


def find_strategy_column(
    frame: pd.DataFrame,
) -> str | None:

    candidates = [
        "strategy_name",
        "strategy",
        "name",
        "candidate_name",
    ]

    for candidate in candidates:

        if candidate in frame.columns:
            return candidate

    return None


# ============================================================
# HALL OF FAME NORMALIZATION
# ============================================================

def normalize_hall_of_fame(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    if frame.empty:
        return pd.DataFrame()

    strategy_column = (
        find_strategy_column(
            frame
        )
    )

    if strategy_column is None:
        return pd.DataFrame()

    rows: list[
        dict[str, Any]
    ] = []

    for _, row in frame.iterrows():

        strategy_name = (
            normalize_strategy_name(
                row.get(
                    strategy_column
                )
            )
        )

        score = safe_float(
            first_existing_value(
                row,
                [
                    "score",
                    "composite_score",
                ],
                0.0,
            )
        )

        profit_factor = safe_float(
            first_existing_value(
                row,
                [
                    "profit_factor",
                    "median_profit_factor",
                ],
                0.0,
            )
        )

        drawdown = safe_float(
            first_existing_value(
                row,
                [
                    "drawdown_percent",
                    "max_drawdown_percent",
                    "maximum_drawdown_percent",
                ],
                0.0,
            )
        )

        consistency = safe_float(
            first_existing_value(
                row,
                [
                    "consistency_percent",
                    "consistency",
                ],
                0.0,
            )
        )

        return_percent = safe_float(
            first_existing_value(
                row,
                [
                    "return_percent",
                    "median_return_percent",
                    "median_return",
                ],
                0.0,
            )
        )

        trade_count = safe_int(
            first_existing_value(
                row,
                [
                    "trade_count",
                    "trades",
                ],
                0,
            )
        )

        role = str(
            first_existing_value(
                row,
                [
                    "role",
                ],
                "",
            )
            or ""
        ).upper()

        symbol = str(
            first_existing_value(
                row,
                [
                    "symbol",
                ],
                "",
            )
            or ""
        ).upper()

        status = str(
            first_existing_value(
                row,
                [
                    "status",
                ],
                "",
            )
            or ""
        ).upper()

        rows.append(
            {
                "strategy_name": (
                    strategy_name
                ),

                "hall_score": (
                    score
                ),

                "hall_profit_factor": (
                    profit_factor
                ),

                "hall_drawdown_percent": (
                    drawdown
                ),

                "hall_consistency_percent": (
                    consistency
                ),

                "hall_return_percent": (
                    return_percent
                ),

                "hall_trade_count": (
                    trade_count
                ),

                "hall_role": (
                    role
                ),

                "hall_symbol": (
                    symbol
                ),

                "hall_status": (
                    status
                ),
            }
        )

    normalized = pd.DataFrame(
        rows
    )

    if normalized.empty:
        return normalized

    normalized = (
        normalized.sort_values(
            "hall_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "strategy_name"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return normalized


# ============================================================
# LIVE PERFORMANCE NORMALIZATION
# ============================================================

def normalize_live_performance(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    if frame.empty:
        return pd.DataFrame()

    strategy_column = (
        find_strategy_column(
            frame
        )
    )

    if strategy_column is None:
        return pd.DataFrame()

    rows: list[
        dict[str, Any]
    ] = []

    for _, row in frame.iterrows():

        strategy_name = (
            normalize_strategy_name(
                row.get(
                    strategy_column
                )
            )
        )

        rows.append(
            {
                "strategy_name": (
                    strategy_name
                ),

                "live_trade_count": safe_int(
                    first_existing_value(
                        row,
                        [
                            "trade_count",
                        ],
                        0,
                    )
                ),

                "live_wins": safe_int(
                    first_existing_value(
                        row,
                        [
                            "wins",
                        ],
                        0,
                    )
                ),

                "live_losses": safe_int(
                    first_existing_value(
                        row,
                        [
                            "losses",
                        ],
                        0,
                    )
                ),

                "live_win_rate_percent": safe_float(
                    first_existing_value(
                        row,
                        [
                            "win_rate_percent",
                        ],
                        0.0,
                    )
                ),

                "live_total_pnl_dollars": safe_float(
                    first_existing_value(
                        row,
                        [
                            "total_pnl_dollars",
                        ],
                        0.0,
                    )
                ),

                "live_average_pnl_dollars": safe_float(
                    first_existing_value(
                        row,
                        [
                            "average_pnl_dollars",
                        ],
                        0.0,
                    )
                ),

                "live_average_return_percent": safe_float(
                    first_existing_value(
                        row,
                        [
                            "average_return_percent",
                        ],
                        0.0,
                    )
                ),

                "live_profit_factor": safe_float(
                    first_existing_value(
                        row,
                        [
                            "profit_factor",
                        ],
                        0.0,
                    )
                ),

                "live_expectancy_dollars": safe_float(
                    first_existing_value(
                        row,
                        [
                            "expectancy_dollars",
                        ],
                        0.0,
                    )
                ),

                "live_expectancy_percent": safe_float(
                    first_existing_value(
                        row,
                        [
                            "expectancy_percent",
                        ],
                        0.0,
                    )
                ),

                "live_average_holding_days": safe_float(
                    first_existing_value(
                        row,
                        [
                            "average_holding_days",
                        ],
                        0.0,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SCORE COMPONENTS
# ============================================================

def research_quality_score(
    hall_score: float,
    profit_factor: float,
    consistency_percent: float,
    drawdown_percent: float,
    return_percent: float,
) -> float:

    normalized_hall_score = clamp(
        hall_score / 40.0,
        0.0,
        1.0,
    )

    normalized_pf = clamp(
        profit_factor / 2.0,
        0.0,
        1.0,
    )

    normalized_consistency = clamp(
        consistency_percent / 100.0,
        0.0,
        1.0,
    )

    normalized_return = clamp(
        (
            return_percent
            + 5.0
        )
        / 15.0,
        0.0,
        1.0,
    )

    drawdown_penalty = clamp(
        drawdown_percent
        / 15.0,
        0.0,
        1.0,
    )

    score = (
        normalized_hall_score
        * 30.0
        + normalized_pf
        * 25.0
        + normalized_consistency
        * 25.0
        + normalized_return
        * 20.0
    )

    score *= (
        1.0
        - (
            drawdown_penalty
            * 0.25
        )
    )

    return clamp(
        score,
        0.0,
        100.0,
    )


def live_quality_score(
    trade_count: int,
    win_rate_percent: float,
    profit_factor: float,
    expectancy_percent: float,
    total_pnl_dollars: float,
) -> float:

    if trade_count <= 0:
        return 50.0

    normalized_win_rate = clamp(
        win_rate_percent
        / 70.0,
        0.0,
        1.0,
    )

    normalized_pf = clamp(
        profit_factor
        / 2.0,
        0.0,
        1.0,
    )

    normalized_expectancy = clamp(
        (
            expectancy_percent
            + 2.0
        )
        / 6.0,
        0.0,
        1.0,
    )

    if total_pnl_dollars > 0:

        pnl_component = 1.0

    elif total_pnl_dollars < 0:

        pnl_component = 0.0

    else:

        pnl_component = 0.5

    score = (
        normalized_win_rate
        * 30.0
        + normalized_pf
        * 30.0
        + normalized_expectancy
        * 30.0
        + pnl_component
        * 10.0
    )

    return clamp(
        score,
        0.0,
        100.0,
    )


def evidence_strength(
    live_trade_count: int,
) -> float:

    if live_trade_count <= 0:
        return 0.0

    return clamp(
        live_trade_count
        / MIN_LIVE_TRADES_FOR_CONFIDENT_LEARNING,
        0.0,
        1.0,
    )


# ============================================================
# LEARNING STATUS
# ============================================================

def determine_recommendation(
    learning_score: float,
    live_trade_count: int,
    live_total_pnl_dollars: float,
    live_expectancy_percent: float,
    live_profit_factor: float,
) -> tuple[
    str,
    str,
    float,
    list[str],
]:

    reasons: list[
        str
    ] = []

    if live_trade_count == 0:

        reasons.append(
            "No closed shadow trades exist for this strategy yet."
        )

        reasons.append(
            "Research/backtest evidence is being used as a prior, "
            "but no live shadow-learning adjustment should be made yet."
        )

        return (
            "RESEARCH_PRIOR_ONLY",
            "HOLD",
            1.00,
            reasons,
        )

    if (
        live_trade_count
        < MIN_LIVE_TRADES_FOR_EARLY_LEARNING
    ):

        reasons.append(
            f"Only {live_trade_count} closed trade(s) are available."
        )

        reasons.append(
            "The sample is too small for a meaningful allocation change."
        )

        return (
            "EARLY_LEARNING",
            "HOLD",
            1.00,
            reasons,
        )

    if (
        live_trade_count
        >= MIN_LIVE_TRADES_FOR_RETIREMENT
        and live_total_pnl_dollars < 0
        and live_expectancy_percent < 0
        and live_profit_factor < 0.85
    ):

        reasons.append(
            "The strategy has a sufficiently large live sample "
            "with negative P/L, negative expectancy, and weak profit factor."
        )

        return (
            "RETIREMENT_CANDIDATE",
            "RETIRE",
            MIN_ALLOCATION_MULTIPLIER,
            reasons,
        )

    if learning_score >= PROMOTE_SCORE:

        reasons.append(
            "Combined research and live evidence is strong."
        )

        return (
            "STRONG",
            "PROMOTE",
            1.25,
            reasons,
        )

    if learning_score >= HOLD_SCORE:

        reasons.append(
            "Combined evidence supports maintaining the current allocation."
        )

        return (
            "HEALTHY",
            "HOLD",
            1.00,
            reasons,
        )

    if learning_score >= REDUCE_SCORE:

        reasons.append(
            "Evidence is mixed and supports reduced research allocation."
        )

        return (
            "WATCH",
            "REDUCE",
            0.75,
            reasons,
        )

    reasons.append(
        "Combined evidence is weak."
    )

    return (
        "WEAK",
        "REDUCE",
        0.50,
        reasons,
    )


# ============================================================
# BUILD LEARNING TABLE
# ============================================================

def build_learning_table(
    hall_frame: pd.DataFrame,
    live_frame: pd.DataFrame,
) -> pd.DataFrame:

    if (
        hall_frame.empty
        and live_frame.empty
    ):

        return pd.DataFrame()

    if hall_frame.empty:

        combined = (
            live_frame.copy()
        )

    elif live_frame.empty:

        combined = (
            hall_frame.copy()
        )

    else:

        combined = hall_frame.merge(
            live_frame,
            on="strategy_name",
            how="outer",
        )

    numeric_defaults = {
        "hall_score": 0.0,
        "hall_profit_factor": 0.0,
        "hall_drawdown_percent": 0.0,
        "hall_consistency_percent": 0.0,
        "hall_return_percent": 0.0,
        "hall_trade_count": 0,
        "live_trade_count": 0,
        "live_wins": 0,
        "live_losses": 0,
        "live_win_rate_percent": 0.0,
        "live_total_pnl_dollars": 0.0,
        "live_average_pnl_dollars": 0.0,
        "live_average_return_percent": 0.0,
        "live_profit_factor": 0.0,
        "live_expectancy_dollars": 0.0,
        "live_expectancy_percent": 0.0,
        "live_average_holding_days": 0.0,
    }

    for (
        column,
        default,
    ) in numeric_defaults.items():

        if column not in combined.columns:

            combined[
                column
            ] = default

        combined[
            column
        ] = pd.to_numeric(
            combined[
                column
            ],
            errors="coerce",
        ).fillna(
            default
        )

    for column in [
        "hall_role",
        "hall_symbol",
        "hall_status",
    ]:

        if column not in combined.columns:

            combined[
                column
            ] = ""

        combined[
            column
        ] = (
            combined[
                column
            ]
            .fillna("")
            .astype(str)
        )

    output_rows: list[
        dict[str, Any]
    ] = []

    for _, row in combined.iterrows():

        strategy_name = (
            normalize_strategy_name(
                row.get(
                    "strategy_name"
                )
            )
        )

        hall_score = safe_float(
            row.get(
                "hall_score"
            )
        )

        hall_profit_factor = safe_float(
            row.get(
                "hall_profit_factor"
            )
        )

        hall_drawdown_percent = abs(
            safe_float(
                row.get(
                    "hall_drawdown_percent"
                )
            )
        )

        hall_consistency_percent = safe_float(
            row.get(
                "hall_consistency_percent"
            )
        )

        hall_return_percent = safe_float(
            row.get(
                "hall_return_percent"
            )
        )

        live_trade_count = safe_int(
            row.get(
                "live_trade_count"
            )
        )

        live_win_rate_percent = safe_float(
            row.get(
                "live_win_rate_percent"
            )
        )

        live_total_pnl_dollars = safe_float(
            row.get(
                "live_total_pnl_dollars"
            )
        )

        live_profit_factor = safe_float(
            row.get(
                "live_profit_factor"
            )
        )

        live_expectancy_percent = safe_float(
            row.get(
                "live_expectancy_percent"
            )
        )

        research_score = (
            research_quality_score(
                hall_score=(
                    hall_score
                ),
                profit_factor=(
                    hall_profit_factor
                ),
                consistency_percent=(
                    hall_consistency_percent
                ),
                drawdown_percent=(
                    hall_drawdown_percent
                ),
                return_percent=(
                    hall_return_percent
                ),
            )
        )

        live_score = (
            live_quality_score(
                trade_count=(
                    live_trade_count
                ),
                win_rate_percent=(
                    live_win_rate_percent
                ),
                profit_factor=(
                    live_profit_factor
                ),
                expectancy_percent=(
                    live_expectancy_percent
                ),
                total_pnl_dollars=(
                    live_total_pnl_dollars
                ),
            )
        )

        evidence = (
            evidence_strength(
                live_trade_count
            )
        )

        live_weight = (
            0.70
            * evidence
        )

        research_weight = (
            1.0
            - live_weight
        )

        learning_score = (
            research_score
            * research_weight
            + live_score
            * live_weight
        )

        (
            learning_status,
            recommendation,
            allocation_multiplier,
            reasons,
        ) = determine_recommendation(
            learning_score=(
                learning_score
            ),
            live_trade_count=(
                live_trade_count
            ),
            live_total_pnl_dollars=(
                live_total_pnl_dollars
            ),
            live_expectancy_percent=(
                live_expectancy_percent
            ),
            live_profit_factor=(
                live_profit_factor
            ),
        )

        allocation_multiplier = clamp(
            allocation_multiplier,
            MIN_ALLOCATION_MULTIPLIER,
            MAX_ALLOCATION_MULTIPLIER,
        )

        if live_trade_count == 0:

            evidence_label = (
                "NO_LIVE_EVIDENCE"
            )

        elif (
            live_trade_count
            < MIN_LIVE_TRADES_FOR_EARLY_LEARNING
        ):

            evidence_label = (
                "VERY_LOW"
            )

        elif (
            live_trade_count
            < MIN_LIVE_TRADES_FOR_CONFIDENT_LEARNING
        ):

            evidence_label = (
                "DEVELOPING"
            )

        else:

            evidence_label = (
                "CONFIDENT"
            )

        output_rows.append(
            {
                "strategy_name": (
                    strategy_name
                ),

                "role": str(
                    row.get(
                        "hall_role",
                        "",
                    )
                ),

                "symbol": str(
                    row.get(
                        "hall_symbol",
                        "",
                    )
                ),

                "hall_status": str(
                    row.get(
                        "hall_status",
                        "",
                    )
                ),

                "research_score": round(
                    research_score,
                    4,
                ),

                "live_score": round(
                    live_score,
                    4,
                ),

                "learning_score": round(
                    learning_score,
                    4,
                ),

                "evidence_strength": round(
                    evidence,
                    4,
                ),

                "evidence_label": (
                    evidence_label
                ),

                "research_weight": round(
                    research_weight,
                    4,
                ),

                "live_weight": round(
                    live_weight,
                    4,
                ),

                "live_trade_count": (
                    live_trade_count
                ),

                "live_wins": safe_int(
                    row.get(
                        "live_wins"
                    )
                ),

                "live_losses": safe_int(
                    row.get(
                        "live_losses"
                    )
                ),

                "live_win_rate_percent": round(
                    live_win_rate_percent,
                    4,
                ),

                "live_total_pnl_dollars": round(
                    live_total_pnl_dollars,
                    2,
                ),

                "live_profit_factor": round(
                    live_profit_factor,
                    4,
                ),

                "live_expectancy_percent": round(
                    live_expectancy_percent,
                    4,
                ),

                "hall_score": round(
                    hall_score,
                    4,
                ),

                "hall_profit_factor": round(
                    hall_profit_factor,
                    4,
                ),

                "hall_drawdown_percent": round(
                    hall_drawdown_percent,
                    4,
                ),

                "hall_consistency_percent": round(
                    hall_consistency_percent,
                    4,
                ),

                "hall_return_percent": round(
                    hall_return_percent,
                    4,
                ),

                "learning_status": (
                    learning_status
                ),

                "recommendation": (
                    recommendation
                ),

                "suggested_allocation_multiplier": round(
                    allocation_multiplier,
                    4,
                ),

                "reasons": (
                    " | ".join(
                        reasons
                    )
                ),
            }
        )

    result = pd.DataFrame(
        output_rows
    )

    if result.empty:
        return result

    return (
        result.sort_values(
            [
                "learning_score",
                "research_score",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SYSTEM-LEVEL STATUS
# ============================================================

def determine_system_status(
    learning_table: pd.DataFrame,
    closed_trade_count: int,
) -> tuple[
    str,
    str,
]:

    if learning_table.empty:

        return (
            "NO_STRATEGIES_AVAILABLE",
            (
                "No strategy research or live strategy "
                "performance data was available."
            ),
        )

    if closed_trade_count <= 0:

        return (
            "WAITING_FOR_CLOSED_TRADES",
            (
                "No shadow trades have closed yet. "
                "Strategy rankings remain research priors "
                "and allocation multipliers remain neutral."
            ),
        )

    if (
        closed_trade_count
        < MIN_LIVE_TRADES_FOR_CONFIDENT_LEARNING
    ):

        return (
            "EARLY_LEARNING",
            (
                "Closed-trade evidence exists, but the "
                "sample is still developing."
            ),
        )

    return (
        "ACTIVE_LEARNING",
        (
            "Sufficient closed-trade evidence exists "
            "for adaptive research recommendations."
        ),
    )


# ============================================================
# TEXT REPORT
# ============================================================

def write_text_report(
    payload: dict[str, Any],
) -> None:

    lines = [
        "STRATEGY LEARNING ENGINE V1",
        "=" * 38,

        (
            "Generated at: "
            f"{payload['generated_at']}"
        ),

        (
            "System status: "
            f"{payload['learning_system_status']}"
        ),

        (
            "Closed trades available: "
            f"{payload['closed_trade_count']}"
        ),

        (
            "Strategies evaluated: "
            f"{payload['strategy_count']}"
        ),

        (
            "Promote recommendations: "
            f"{payload['promote_count']}"
        ),

        (
            "Hold recommendations: "
            f"{payload['hold_count']}"
        ),

        (
            "Reduce recommendations: "
            f"{payload['reduce_count']}"
        ),

        (
            "Retirement candidates: "
            f"{payload['retire_count']}"
        ),

        "",

        "SYSTEM MESSAGE",
        "-" * 24,

        payload[
            "learning_system_message"
        ],

        "",

        "TOP STRATEGIES",
        "-" * 24,
    ]

    strategies = payload.get(
        "strategies",
        [],
    )

    if strategies:

        for strategy in strategies[
            :10
        ]:

            lines.append(
                (
                    f"{strategy['strategy_name']} | "
                    f"score={strategy['learning_score']:.2f} | "
                    f"{strategy['recommendation']} | "
                    f"multiplier="
                    f"{strategy['suggested_allocation_multiplier']:.2f} | "
                    f"live trades={strategy['live_trade_count']}"
                )
            )

    else:

        lines.append(
            "No strategies available."
        )

    lines.extend(
        [
            "",

            "SAFETY",
            "-" * 24,

            "Research learning only.",

            "Portfolio Commander modified: False",

            "Production strategy changed: False",

            "Trade ledger modified: False",

            "Trading client created: False",

            "Market request made: False",

            "Order submitted: False",
        ]
    )

    LATEST_TEXT_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    performance_summary = load_json(
        PERFORMANCE_SUMMARY_PATH,
        {},
    )

    live_raw = load_csv(
        STRATEGY_ANALYTICS_PATH
    )

    hall_raw = load_csv(
        HALL_OF_FAME_CSV_PATH
    )

    hall_json = load_json(
        HALL_OF_FAME_JSON_PATH,
        {},
    )

    live_frame = (
        normalize_live_performance(
            live_raw
        )
    )

    hall_frame = (
        normalize_hall_of_fame(
            hall_raw
        )
    )

    learning_table = (
        build_learning_table(
            hall_frame=hall_frame,
            live_frame=live_frame,
        )
    )

    closed_trade_count = safe_int(
        performance_summary.get(
            "closed_position_count",
            0,
        )
    )

    (
        system_status,
        system_message,
    ) = determine_system_status(
        learning_table=learning_table,
        closed_trade_count=(
            closed_trade_count
        ),
    )

    strategies: list[
        dict[str, Any]
    ] = []

    if not learning_table.empty:

        strategies = (
            learning_table.to_dict(
                orient="records"
            )
        )

    promote_count = sum(
        strategy.get(
            "recommendation"
        ) == "PROMOTE"
        for strategy
        in strategies
    )

    hold_count = sum(
        strategy.get(
            "recommendation"
        ) == "HOLD"
        for strategy
        in strategies
    )

    reduce_count = sum(
        strategy.get(
            "recommendation"
        ) == "REDUCE"
        for strategy
        in strategies
    )

    retire_count = sum(
        strategy.get(
            "recommendation"
        ) == "RETIRE"
        for strategy
        in strategies
    )

    if strategies:

        top_strategy = (
            strategies[
                0
            ].get(
                "strategy_name"
            )
        )

        top_learning_score = (
            safe_float(
                strategies[
                    0
                ].get(
                    "learning_score"
                )
            )
        )

    else:

        top_strategy = None
        top_learning_score = 0.0

    payload = {

        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "engine": (
            "Strategy Learning Engine v1"
        ),

        "learning_system_status": (
            system_status
        ),

        "learning_system_message": (
            system_message
        ),

        "closed_trade_count": (
            closed_trade_count
        ),

        "strategy_count": len(
            strategies
        ),

        "promote_count": (
            promote_count
        ),

        "hold_count": (
            hold_count
        ),

        "reduce_count": (
            reduce_count
        ),

        "retire_count": (
            retire_count
        ),

        "top_strategy": (
            top_strategy
        ),

        "top_learning_score": round(
            top_learning_score,
            4,
        ),

        "configuration": {

            "minimum_live_trades_for_early_learning": (
                MIN_LIVE_TRADES_FOR_EARLY_LEARNING
            ),

            "minimum_live_trades_for_confident_learning": (
                MIN_LIVE_TRADES_FOR_CONFIDENT_LEARNING
            ),

            "minimum_live_trades_for_retirement": (
                MIN_LIVE_TRADES_FOR_RETIREMENT
            ),

            "promote_score": (
                PROMOTE_SCORE
            ),

            "hold_score": (
                HOLD_SCORE
            ),

            "reduce_score": (
                REDUCE_SCORE
            ),

            "minimum_allocation_multiplier": (
                MIN_ALLOCATION_MULTIPLIER
            ),

            "maximum_allocation_multiplier": (
                MAX_ALLOCATION_MULTIPLIER
            ),
        },

        "hall_of_fame_strategy_count": safe_int(
            hall_json.get(
                "strategy_count",
                len(
                    hall_frame
                ),
            )
        ),

        "strategies": (
            strategies
        ),

        "research_only": True,

        "portfolio_commander_modified": False,

        "production_strategy_changed": False,

        "trade_ledger_modified": False,

        "trading_client_created": False,

        "market_request_made": False,

        "order_submitted": False,
    }

    save_json(
        LATEST_SUMMARY_PATH,
        payload,
    )

    history_path = (
        LEARNING_HISTORY_DIRECTORY
        / (
            datetime.now()
            .astimezone()
            .strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            + ".json"
        )
    )

    save_json(
        history_path,
        payload,
    )

    if learning_table.empty:

        pd.DataFrame(
            columns=[
                "strategy_name",
                "learning_score",
                "learning_status",
                "recommendation",
                "suggested_allocation_multiplier",
            ]
        ).to_csv(
            STRATEGY_SCORES_PATH,
            index=False,
        )

        pd.DataFrame(
            columns=[
                "strategy_name",
                "recommendation",
                "suggested_allocation_multiplier",
                "evidence_label",
                "live_trade_count",
            ]
        ).to_csv(
            ALLOCATION_GUIDANCE_PATH,
            index=False,
        )

    else:

        learning_table.to_csv(
            STRATEGY_SCORES_PATH,
            index=False,
        )

        guidance_columns = [
            "strategy_name",
            "role",
            "symbol",
            "learning_score",
            "learning_status",
            "recommendation",
            "suggested_allocation_multiplier",
            "evidence_label",
            "live_trade_count",
            "live_win_rate_percent",
            "live_profit_factor",
            "live_expectancy_percent",
            "hall_score",
            "hall_profit_factor",
            "hall_consistency_percent",
            "hall_drawdown_percent",
            "reasons",
        ]

        available_columns = [
            column
            for column
            in guidance_columns
            if column
            in learning_table.columns
        ]

        learning_table[
            available_columns
        ].to_csv(
            ALLOCATION_GUIDANCE_PATH,
            index=False,
        )

    write_text_report(
        payload
    )

    print(
        "Strategy Learning Engine v1"
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print()

    print(
        f"JSON report: "
        f"{LATEST_SUMMARY_PATH}"
    )

    print(
        f"Text report: "
        f"{LATEST_TEXT_PATH}"
    )

    print(
        f"Strategy scores: "
        f"{STRATEGY_SCORES_PATH}"
    )

    print(
        f"Allocation guidance: "
        f"{ALLOCATION_GUIDANCE_PATH}"
    )

    print(
        "Research learning only."
    )

    print(
        "Portfolio Commander was not modified."
    )

    print(
        "Production strategy was not changed."
    )

    print(
        "The trade ledger was not modified."
    )

    print(
        "No trading client was created."
    )

    print(
        "No market request was made."
    )

    print(
        "No order was submitted."
    )


if __name__ == "__main__":
    main()