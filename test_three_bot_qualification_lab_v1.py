from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


with tempfile.TemporaryDirectory() as temporary:
    data_dir = Path(temporary) / "scanner"
    data_dir.mkdir()

    symbols = ["SPY", "QQQ", "IWM", "AMZN", "NVDA", "TSLA", "SLV"]

    for offset, symbol in enumerate(symbols):
        rows = 900
        x = np.arange(rows)
        close = 150 - 0.03 * x + 5 * np.sin(x / (15 + offset))

        frame = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=rows, freq="D"),
            "open": close + 0.2,
            "high": close + 1.2,
            "low": close - 1.2,
            "close": close,
            "volume": 1_500_000 + 200_000 * np.sin(x / 8),
        })

        frame.to_csv(data_dir / f"{symbol}_1D.csv", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "three_bot_qualification_lab_v1.py",
            "--data-dir",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
    )

    report_path = Path("reports/qualification/three_bot_qualification_lab_v1.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    output = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "qualified_role_count": report.get("qualified_role_count", 0),
        "qualification_report_created": report_path.exists(),
        "role_csv_created": Path("reports/qualification/qualified_three_bot_roles.csv").exists(),
        "short_leaderboard_created": Path("reports/qualification/short_arena_leaderboard.csv").exists(),
        "market_request_made": False,
        "order_submitted": False,
    }

    print("Three-Bot Qualification Lab v1 test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if output["status"] != "passed":
        print(completed.stdout)
        print(completed.stderr)
        raise SystemExit("Qualification Lab test failed.")
