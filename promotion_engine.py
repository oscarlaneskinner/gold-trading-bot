"""Advisory-only promotion recommendations."""

from __future__ import annotations
from typing import Any
from baseline_comparison import compare_candidate
from statistical_validation import validate_candidate
from strategy_leaderboard import score_metrics

def recommend(candidate: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics") or {}
    candidate_score = score_metrics(metrics)
    baseline_score = score_metrics(baseline_metrics)

    if candidate_score is None or baseline_score is None:
        recommendation = "COLLECT_MORE_DATA"
        reason = "Candidate or baseline metrics are incomplete."
        comparison = {}
        validation = {}
    else:
        comparison = compare_candidate(baseline_metrics, metrics)
        validation = validate_candidate(metrics, baseline_metrics)
        if int(metrics.get("trade_count", 0)) < 30:
            recommendation = "COLLECT_MORE_DATA"
            reason = "Fewer than 30 candidate trades are available."
        elif validation["passed"] and candidate_score > baseline_score and comparison["candidate_wins"] > comparison["baseline_wins_or_ties"]:
            recommendation = "PROMOTE_TO_EXTENDED_PAPER_TEST"
            reason = "Candidate passed screening and scored above baseline."
        elif candidate_score <= baseline_score:
            recommendation = "REJECT"
            reason = "Candidate did not outperform the baseline score."
        else:
            recommendation = "READY_FOR_HUMAN_REVIEW"
            reason = "Mixed results require deliberate human review."

    return {
        "recommendation": recommendation,
        "reason": reason,
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "comparison": comparison,
        "statistical_validation": validation,
        "production_strategy_changed": False,
        "order_submitted": False,
    }
