"""Conservative statistical screening for candidate strategies."""

from __future__ import annotations
import math
import statistics
from typing import Any

def mean_interval(values: list[float]) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.96 * se
    return {"mean": mean, "lower": mean - margin, "upper": mean + margin}

def validate_candidate(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    candidate_folds = candidate_metrics.get("fold_returns") or []
    baseline_folds = baseline_metrics.get("fold_returns") or []
    candidate_ci = mean_interval(candidate_folds)
    baseline_ci = mean_interval(baseline_folds)

    rules = {
        "trade_count_at_least_30": int(candidate_metrics.get("trade_count", 0)) >= 30,
        "at_least_6_folds": len(candidate_folds) >= 6,
        "at_least_4_positive_folds": sum(x > 0 for x in candidate_folds) >= 4,
        "average_return_not_worse": float(candidate_metrics.get("average_return", 0)) >= float(baseline_metrics.get("average_return", 0)),
        "maximum_drawdown_not_materially_worse": abs(float(candidate_metrics.get("maximum_drawdown", 1))) <= 1.25 * abs(float(baseline_metrics.get("maximum_drawdown", 1))),
        "profit_factor_not_worse": float(candidate_metrics.get("profit_factor", 0)) >= 0.95 * float(baseline_metrics.get("profit_factor", 0)),
        "confidence_interval_available": candidate_ci is not None and baseline_ci is not None,
    }
    return {
        "passed": all(rules.values()),
        "rules": rules,
        "candidate_mean_confidence_interval": candidate_ci,
        "baseline_mean_confidence_interval": baseline_ci,
        "note": "Screening only; this does not prove future profitability.",
    }
