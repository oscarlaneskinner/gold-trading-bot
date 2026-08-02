from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

with tempfile.TemporaryDirectory() as temporary:
    csv_path = Path(temporary) / "sample.csv"
    rows = 1200
    x = np.arange(rows)
    close = 100 + 0.03*x + 6*np.sin(x/18) + 2*np.sin(x/5)
    pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
        "open": close - 0.3,
        "high": close + 1.2,
        "low": close - 1.2,
        "close": close,
        "volume": 1_000_000 + 200_000*np.sin(x/9),
    }).to_csv(csv_path, index=False)

    completed = subprocess.run(
        [sys.executable, "strategy_research_arena.py", "--csv", str(csv_path)],
        capture_output=True,
        text=True,
    )

    report = Path("reports/strategy_research_arena.json")
    data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}

    output = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "candidate_count": data.get("candidate_count", 0),
        "json_report_created": report.exists(),
        "csv_report_created": Path("reports/strategy_research_arena.csv").exists(),
        "market_request_made": False,
        "order_submitted": False,
    }

    print("UT Strategy Research Arena test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if output["status"] != "passed" or output["candidate_count"] < 100:
        print(completed.stdout)
        print(completed.stderr)
        raise SystemExit("Strategy arena test failed.")
