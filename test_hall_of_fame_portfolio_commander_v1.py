from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


sample = {
    "top_finalists": [
        {
            "candidate_id": "GLD-CHAMPION",
            "symbol": "GLD",
            "score": 55.0,
            "median_test_return_percent": 8.2,
            "median_drawdown_percent": 4.1,
            "median_profit_factor": 1.6,
            "consistency_percent": 75.0,
            "median_trade_count": 40,
            "status": "WALK_FORWARD_FINALIST",
        },
        {
            "candidate_id": "LONG-MOMENTUM",
            "symbol": "MULTI",
            "filter_set": "Long Momentum",
            "score": 48.0,
            "median_test_return_percent": 6.4,
            "median_drawdown_percent": 5.0,
            "median_profit_factor": 1.45,
            "consistency_percent": 70.0,
            "median_trade_count": 35,
            "status": "WALK_FORWARD_FINALIST",
        },
        {
            "candidate_id": "SHORT-BREAKDOWN",
            "symbol": "MULTI",
            "filter_set": "Short Breakdown",
            "score": 44.0,
            "median_test_return_percent": 5.8,
            "median_drawdown_percent": 6.0,
            "median_profit_factor": 1.35,
            "consistency_percent": 65.0,
            "median_trade_count": 30,
            "status": "WALK_FORWARD_FINALIST",
        },
    ]
}

with tempfile.TemporaryDirectory() as temporary:
    report = Path(temporary) / "sample.json"
    report.write_text(json.dumps(sample), encoding="utf-8")

    first = subprocess.run(
        [sys.executable, "strategy_hall_of_fame_v1.py", "--reports", str(report)],
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [sys.executable, "portfolio_commander_v1.py"],
        capture_output=True,
        text=True,
    )

    hall = Path("reports/hall_of_fame/strategy_hall_of_fame.json")
    portfolio = Path("reports/portfolio/portfolio_commander_v1.json")

    output = {
        "status": "passed" if first.returncode == 0 and second.returncode == 0 else "failed",
        "hall_of_fame_created": hall.exists(),
        "portfolio_plan_created": portfolio.exists(),
        "database_created": Path("data/strategy_hall_of_fame.sqlite3").exists(),
        "market_request_made": False,
        "order_submitted": False,
    }

    print("Hall of Fame and Portfolio Commander test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if output["status"] != "passed":
        print(first.stdout, first.stderr)
        print(second.stdout, second.stderr)
        raise SystemExit("Test failed.")
