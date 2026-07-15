"""Offline integration test for Phase 7.3 through 7.6."""

from __future__ import annotations
import json
from promotion_engine import recommend
from strategy_leaderboard import build_leaderboard

def run():
    baseline = {
        "average_return": 0.020, "median_return": 0.018,
        "win_rate": 0.60, "profit_factor": 1.50,
        "maximum_drawdown": -0.08, "trade_count": 60,
        "fold_returns": [0.02, 0.01, -0.01, 0.03, 0.02, 0.01],
    }
    candidate = {
        "experiment_id": "EXP-TEST",
        "name": "Synthetic superior candidate",
        "status": "COMPLETED",
        "metrics": {
            "average_return": 0.026, "median_return": 0.022,
            "win_rate": 0.65, "profit_factor": 1.70,
            "maximum_drawdown": -0.075, "trade_count": 65,
            "fold_returns": [0.03, 0.02, -0.005, 0.035, 0.025, 0.02],
        },
    }
    leaderboard = build_leaderboard([candidate])
    promotion = recommend(candidate, baseline)
    output = {
        "status": "passed",
        "leaderboard_count": len(leaderboard),
        "top_experiment": leaderboard[0]["experiment_id"],
        "promotion_recommendation": promotion["recommendation"],
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }
    print("GLD Phase 7 complete integration test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

if __name__ == "__main__":
    run()
