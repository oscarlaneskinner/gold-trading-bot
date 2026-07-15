"""Build the complete advisory Phase 7 Strategy Laboratory report."""

from __future__ import annotations
import json
from pathlib import Path
from experiment_registry import load_registry
from logger import write_json
from promotion_engine import recommend
from strategy_leaderboard import build_leaderboard

BASELINE_PATH = Path("reports/strategy_baseline_metrics.json")
REPORT_PATH = Path("reports/phase7_strategy_lab.json")

def run():
    registry = load_registry()
    baseline_metrics = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else None
    experiments = registry["experiments"]
    recommendations = []
    if baseline_metrics:
        for exp in experiments:
            if exp.get("metrics"):
                recommendations.append({
                    "experiment_id": exp["experiment_id"],
                    "result": recommend(exp, baseline_metrics),
                })
    report = {
        "baseline": registry["baseline"],
        "baseline_metrics_available": baseline_metrics is not None,
        "experiment_count": len(experiments),
        "leaderboard": build_leaderboard(experiments),
        "promotion_recommendations": recommendations,
        "overall_status": "READY" if baseline_metrics else "BASELINE_METRICS_REQUIRED",
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }
    write_json(REPORT_PATH, report)
    print("GLD complete Phase 7 Strategy Laboratory")
    print(json.dumps(report, indent=2))
    print(f"JSON report: {REPORT_PATH}")
    print("No market request was made.")
    print("No order was submitted.")

if __name__ == "__main__":
    run()
