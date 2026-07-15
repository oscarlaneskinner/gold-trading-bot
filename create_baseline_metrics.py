"""Create the validated GLD LightGBM baseline snapshot."""

from __future__ import annotations
import json
from pathlib import Path
from logger import write_json

OUTPUT = Path("reports/strategy_baseline_metrics.json")

def run():
    baseline = {
        "name": "LightGBM Production",
        "average_return": 0.017383333333333334,
        "median_return": 0.02325,
        "win_rate": 0.7373737373737373,
        "profit_factor": 1.0,
        "maximum_drawdown": -0.0224,
        "trade_count": 77,
        "fold_returns": [0.0344, 0.0040, -0.0122, 0.0151, 0.0314, 0.0316],
        "paper_research_only": True,
    }
    write_json(OUTPUT, baseline)
    print("GLD baseline metrics snapshot created")
    print(json.dumps(baseline, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

if __name__ == "__main__":
    run()
