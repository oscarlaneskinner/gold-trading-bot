"""Trade-intelligence enrichment for the GLD research engine."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any

from trade_memory import DATABASE_PATH, connect, initialize_database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_intelligence_schema() -> None:
    initialize_database()

    with closing(connect()) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS decision_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER NOT NULL UNIQUE,
                market_regime TEXT NOT NULL,
                trend_state TEXT NOT NULL,
                volatility_state TEXT NOT NULL,
                momentum_state TEXT NOT NULL,
                rsi_14 REAL,
                rsi_7 REAL,
                atr_pct REAL,
                volatility_20d REAL,
                price_vs_ema200 REAL,
                ema9_vs_ema21 REAL,
                ema21_vs_ema50 REAL,
                volume_ma_ratio REAL,
                volume_change REAL,
                confidence_bucket TEXT NOT NULL,
                research_tags_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(decision_id)
                    REFERENCES decisions(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_decision_intelligence_regime
            ON decision_intelligence(market_regime);

            CREATE INDEX IF NOT EXISTS idx_decision_intelligence_confidence
            ON decision_intelligence(confidence_bucket);

            CREATE TABLE IF NOT EXISTS trade_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL UNIQUE,
                entry_decision_id INTEGER,
                market_regime_at_entry TEXT,
                trend_state_at_entry TEXT,
                volatility_state_at_entry TEXT,
                momentum_state_at_entry TEXT,
                confidence_bucket_at_entry TEXT,
                entry_research_tags_json TEXT,
                result_classification TEXT,
                grade TEXT,
                holding_days REAL,
                realized_return REAL,
                realized_profit_loss REAL,
                updated_at_utc TEXT NOT NULL,
                FOREIGN KEY(trade_id)
                    REFERENCES trades(id)
                    ON DELETE CASCADE,
                FOREIGN KEY(entry_decision_id)
                    REFERENCES decisions(id)
                    ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_trade_intelligence_result
            ON trade_intelligence(result_classification);

            CREATE INDEX IF NOT EXISTS idx_trade_intelligence_regime
            ON trade_intelligence(market_regime_at_entry);
            """
        )
        connection.commit()


def confidence_bucket(probability_up: float) -> str:
    if probability_up >= 0.90:
        return "90_to_100"
    if probability_up >= 0.85:
        return "85_to_90"
    if probability_up >= 0.80:
        return "80_to_85"
    if probability_up >= 0.70:
        return "70_to_80"
    if probability_up >= 0.60:
        return "60_to_70"
    return "50_to_60"


def classify_trend(features: dict[str, float]) -> str:
    price_vs_ema200 = float(features.get("price_vs_ema200", 0.0))
    ema9_vs_ema21 = float(features.get("ema9_vs_ema21", 0.0))
    ema21_vs_ema50 = float(features.get("ema21_vs_ema50", 0.0))

    bullish_votes = sum(
        value > 0
        for value in (
            price_vs_ema200,
            ema9_vs_ema21,
            ema21_vs_ema50,
        )
    )

    bearish_votes = sum(
        value < 0
        for value in (
            price_vs_ema200,
            ema9_vs_ema21,
            ema21_vs_ema50,
        )
    )

    if bullish_votes == 3:
        return "strong_uptrend"
    if bullish_votes >= 2:
        return "weak_uptrend"
    if bearish_votes == 3:
        return "strong_downtrend"
    if bearish_votes >= 2:
        return "weak_downtrend"
    return "sideways"


def classify_volatility(features: dict[str, float]) -> str:
    atr_pct = float(features.get("atr_pct", 0.0))
    volatility_20d = float(features.get("volatility_20d", 0.0))

    combined = max(atr_pct, volatility_20d)

    if combined >= 0.035:
        return "high"
    if combined >= 0.020:
        return "moderate"
    return "low"


def classify_momentum(features: dict[str, float]) -> str:
    rsi_14 = float(features.get("rsi_14", 50.0))
    return_5d = float(features.get("return_5d", 0.0))
    return_20d = float(features.get("return_20d", 0.0))

    if rsi_14 >= 70:
        return "overbought"
    if rsi_14 <= 30:
        return "oversold"
    if return_5d > 0 and return_20d > 0:
        return "positive"
    if return_5d < 0 and return_20d < 0:
        return "negative"
    return "mixed"


def classify_market_regime(
    trend_state: str,
    volatility_state: str,
) -> str:
    if trend_state in {"strong_uptrend", "weak_uptrend"}:
        return (
            "bull_trend_high_volatility"
            if volatility_state == "high"
            else "bull_trend"
        )

    if trend_state in {"strong_downtrend", "weak_downtrend"}:
        return (
            "bear_trend_high_volatility"
            if volatility_state == "high"
            else "bear_trend"
        )

    return (
        "sideways_high_volatility"
        if volatility_state == "high"
        else "sideways"
    )


def build_research_tags(
    *,
    features: dict[str, float],
    probability_up: float,
    trend_state: str,
    volatility_state: str,
    momentum_state: str,
) -> list[str]:
    tags = [
        trend_state,
        f"volatility_{volatility_state}",
        f"momentum_{momentum_state}",
        f"confidence_{confidence_bucket(probability_up)}",
    ]

    if float(features.get("volume_ma_ratio", 1.0)) >= 1.20:
        tags.append("high_relative_volume")

    if abs(float(features.get("price_vs_ema200", 0.0))) >= 0.10:
        tags.append("far_from_ema200")

    if float(features.get("atr_pct", 0.0)) >= 0.03:
        tags.append("elevated_atr")

    return tags


def record_decision_intelligence(
    *,
    decision_id: int,
    probability_up: float,
    features: dict[str, float],
) -> dict[str, Any]:
    initialize_intelligence_schema()

    trend_state = classify_trend(features)
    volatility_state = classify_volatility(features)
    momentum_state = classify_momentum(features)
    market_regime = classify_market_regime(
        trend_state,
        volatility_state,
    )

    tags = build_research_tags(
        features=features,
        probability_up=probability_up,
        trend_state=trend_state,
        volatility_state=volatility_state,
        momentum_state=momentum_state,
    )

    payload = {
        "market_regime": market_regime,
        "trend_state": trend_state,
        "volatility_state": volatility_state,
        "momentum_state": momentum_state,
        "confidence_bucket": confidence_bucket(probability_up),
        "research_tags": tags,
    }

    with closing(connect()) as connection:
        connection.execute(
            """
            INSERT INTO decision_intelligence (
                decision_id,
                market_regime,
                trend_state,
                volatility_state,
                momentum_state,
                rsi_14,
                rsi_7,
                atr_pct,
                volatility_20d,
                price_vs_ema200,
                ema9_vs_ema21,
                ema21_vs_ema50,
                volume_ma_ratio,
                volume_change,
                confidence_bucket,
                research_tags_json,
                created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(decision_id) DO UPDATE SET
                market_regime = excluded.market_regime,
                trend_state = excluded.trend_state,
                volatility_state = excluded.volatility_state,
                momentum_state = excluded.momentum_state,
                rsi_14 = excluded.rsi_14,
                rsi_7 = excluded.rsi_7,
                atr_pct = excluded.atr_pct,
                volatility_20d = excluded.volatility_20d,
                price_vs_ema200 = excluded.price_vs_ema200,
                ema9_vs_ema21 = excluded.ema9_vs_ema21,
                ema21_vs_ema50 = excluded.ema21_vs_ema50,
                volume_ma_ratio = excluded.volume_ma_ratio,
                volume_change = excluded.volume_change,
                confidence_bucket = excluded.confidence_bucket,
                research_tags_json = excluded.research_tags_json,
                created_at_utc = excluded.created_at_utc
            """,
            (
                int(decision_id),
                market_regime,
                trend_state,
                volatility_state,
                momentum_state,
                features.get("rsi_14"),
                features.get("rsi_7"),
                features.get("atr_pct"),
                features.get("volatility_20d"),
                features.get("price_vs_ema200"),
                features.get("ema9_vs_ema21"),
                features.get("ema21_vs_ema50"),
                features.get("volume_ma_ratio"),
                features.get("volume_change"),
                payload["confidence_bucket"],
                json.dumps(tags, sort_keys=True),
                utc_now(),
            ),
        )
        connection.commit()

    return payload


def intelligence_summary() -> dict[str, Any]:
    initialize_intelligence_schema()

    with closing(connect()) as connection:
        decision_rows = connection.execute(
            """
            SELECT market_regime, COUNT(*) AS count
            FROM decision_intelligence
            GROUP BY market_regime
            ORDER BY count DESC
            """
        ).fetchall()

        confidence_rows = connection.execute(
            """
            SELECT confidence_bucket, COUNT(*) AS count
            FROM decision_intelligence
            GROUP BY confidence_bucket
            ORDER BY confidence_bucket
            """
        ).fetchall()

        total = connection.execute(
            "SELECT COUNT(*) FROM decision_intelligence"
        ).fetchone()[0]

    return {
        "intelligence_record_count": int(total),
        "market_regime_counts": {
            row["market_regime"]: int(row["count"])
            for row in decision_rows
        },
        "confidence_bucket_counts": {
            row["confidence_bucket"]: int(row["count"])
            for row in confidence_rows
        },
    }
