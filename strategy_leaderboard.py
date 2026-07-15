"""Rank completed Strategy Laboratory experiments."""

from __future__ import annotations
from typing import Any

WEIGHTS = {
    "average_return": 0.25,
    "profit_factor": 0.20,
    "maximum_drawdown": 0.20,
    "win_rate": 0.15,
    "risk_adjusted_return": 0.15,
    "trade_count": 0.05,
}

def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

def score_metrics(metrics: dict[str, Any]) -> float | None:
    required = {
        "average_return", "profit_factor", "maximum_drawdown",
        "win_rate", "trade_count",
    }
    if not required.issubset(metrics):
        return None

    average_return = float(metrics["average_return"])
    profit_factor = float(metrics["profit_factor"])
    drawdown = abs(float(metrics["maximum_drawdown"]))
    win_rate = float(metrics["win_rate"])
    trade_count = int(metrics["trade_count"])
    risk_adjusted = average_return / drawdown if drawdown else 0.0

    parts = {
        "average_return": clamp((average_return + 0.10) / 0.30),
        "profit_factor": clamp(profit_factor / 3.0),
        "maximum_drawdown": clamp(1.0 - drawdown / 0.20),
        "win_rate": clamp(win_rate),
        "risk_adjusted_return": clamp((risk_adjusted + 1.0) / 4.0),
        "trade_count": clamp(trade_count / 100.0),
    }
    return round(sum(parts[k] * WEIGHTS[k] for k in WEIGHTS) * 100, 3)

def build_leaderboard(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for exp in experiments:
        score = score_metrics(exp.get("metrics") or {})
        if score is not None:
            rows.append({
                "experiment_id": exp["experiment_id"],
                "name": exp["name"],
                "status": exp["status"],
                "score": score,
                "metrics": exp["metrics"],
            })
    rows.sort(key=lambda r: (r["score"], r["metrics"].get("average_return", 0)), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows
