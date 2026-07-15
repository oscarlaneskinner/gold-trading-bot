"""Compare candidate and baseline metrics."""

from __future__ import annotations
from typing import Any

DIRECTIONS = {
    "average_return": "higher",
    "median_return": "higher",
    "win_rate": "higher",
    "profit_factor": "higher",
    "maximum_drawdown": "lower_absolute",
    "trade_count": "higher",
}

def compare_candidate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for metric, direction in DIRECTIONS.items():
        if metric not in baseline or metric not in candidate:
            continue
        b = float(baseline[metric])
        c = float(candidate[metric])
        if direction == "lower_absolute":
            improvement = abs(b) - abs(c)
            better = abs(c) < abs(b)
        else:
            improvement = c - b
            better = c > b
        results[metric] = {
            "baseline": b,
            "candidate": c,
            "absolute_improvement": improvement,
            "relative_improvement": improvement / abs(b) if b else None,
            "candidate_better": better,
        }
    return {
        "comparisons": results,
        "candidate_wins": sum(v["candidate_better"] for v in results.values()),
        "baseline_wins_or_ties": sum(not v["candidate_better"] for v in results.values()),
    }
