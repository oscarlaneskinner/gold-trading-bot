"""Offline test for automated UT tournament."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "sample.csv"
    rows = 500
    x = np.arange(rows)
    close = 100 + 0.04 * x + 5 * np.sin(x / 15)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000 + 100_000 * np.sin(x / 7),
        }
    )
    frame.to_csv(path, index=False)

    completed = subprocess.run(
        [sys.executable, "automated_ut_tournament.py", "--csv", str(path)],
        capture_output=True,
        text=True,
    )

    output = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "json_report_created": Path("reports/automated_ut_tournament.json").exists(),
        "csv_report_created": Path("reports/automated_ut_tournament.csv").exists(),
        "market_request_made": False,
        "order_submitted": False,
    }

    print("Automated UT tournament test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if output["status"] != "passed":
        print(completed.stdout)
        print(completed.stderr)
        raise SystemExit("Automated tournament test failed.")
