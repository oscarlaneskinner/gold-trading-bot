from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


Path("reports/market_regime").mkdir(parents=True, exist_ok=True)
Path("reports/portfolio").mkdir(parents=True, exist_ok=True)
Path("reports/hall_of_fame").mkdir(parents=True, exist_ok=True)
Path("reports/scanner").mkdir(parents=True, exist_ok=True)

Path("reports/market_regime/market_regime_lab_v1.json").write_text(
    json.dumps({
        "regime": "BULL",
        "permissions": {
            "allow_gld_bot": True,
            "allow_long_bot": True,
            "allow_short_bot": False,
            "reduce_position_size": False,
        },
    }),
    encoding="utf-8",
)

Path("reports/portfolio/portfolio_commander_v1.json").write_text(
    json.dumps({
        "allocations": [
            {
                "strategy_name": "ARENA-0081",
                "role": "GLD",
                "symbol": "GLD",
                "allocation_dollars": 700.0,
            },
            {
                "strategy_name": "MM-0013",
                "role": "LONG",
                "symbol": "MULTI",
                "allocation_dollars": 620.0,
            },
        ]
    }),
    encoding="utf-8",
)

Path("reports/hall_of_fame/strategy_hall_of_fame.json").write_text(
    json.dumps({
        "strategy_count": 2
    }),
    encoding="utf-8",
)

Path("reports/scanner/championship_scanner_v1.json").write_text(
    json.dumps({
        "top_longs": [
            {
                "symbol": "NVDA",
                "score": 88.5,
                "suggested_stop": 150.0,
                "suggested_target": 180.0,
            }
        ]
    }),
    encoding="utf-8",
)

completed = subprocess.run(
    [
        sys.executable,
        "two_bot_shadow_controller_v1.py",
    ],
    capture_output=True,
    text=True,
)

report_path = Path(
    "reports/shadow/two_bot_shadow_controller_v1.json"
)

report = (
    json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )
    if report_path.exists()
    else {}
)

output = {
    "status": (
        "passed"
        if completed.returncode == 0
        else "failed"
    ),
    "proposal_count":
        report.get("proposal_count", 0),
    "json_report_created":
        report_path.exists(),
    "csv_report_created":
        Path(
            "reports/shadow/"
            "two_bot_shadow_proposals.csv"
        ).exists(),
    "summary_created":
        Path(
            "reports/shadow/"
            "two_bot_shadow_controller_v1_summary.txt"
        ).exists(),
    "market_request_made": False,
    "order_submitted": False,
}

print("Two-Bot Shadow Controller v1 test")
print(json.dumps(output, indent=2))
print("No market request was made.")
print("No order was submitted.")

if (
    output["status"] != "passed"
    or output["proposal_count"] != 2
):
    print(completed.stdout)
    print(completed.stderr)
    raise SystemExit(
        "Two-Bot Shadow Controller test failed."
    )
