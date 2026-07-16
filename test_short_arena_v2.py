from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


with tempfile.TemporaryDirectory() as temporary:
    data_dir = Path(temporary) / "data"
    data_dir.mkdir()

    symbols = ["SPY", "QQQ", "IWM", "AMZN", "NVDA", "TSLA", "SLV"]

    for offset, symbol in enumerate(symbols):
        rows = 900
        x = np.arange(rows)
        close = 180 - 0.035 * x + 6 * np.sin(x / (13 + offset))

        frame = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=rows, freq="D"),
                "open": close + 0.25,
                "high": close + 1.4,
                "low": close - 1.4,
                "close": close,
                "volume": 1_500_000 + 250_000 * np.sin(x / 7),
            }
        )

        frame.to_csv(data_dir / f"{symbol}_1D.csv", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "short_arena_v2.py",
            "--data-dir",
            str(data_dir),
            "--limit",
            "120",
        ],
        capture_output=True,
        text=True,
    )

    result_path = Path("reports/short_arena_v2/short_arena_v2_results.json")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}

    output = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "candidate_count": result.get("candidate_count", 0),
        "results_created": result_path.exists(),
        "leaderboard_created": Path("reports/short_arena_v2/short_arena_v2_leaderboard.csv").exists(),
        "top_100_created": Path("reports/short_arena_v2/short_arena_v2_top_100.csv").exists(),
        "market_request_made": False,
        "order_submitted": False,
    }

    print("Short Arena v2 Championship test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if output["status"] != "passed" or output["candidate_count"] != 120:
        print(completed.stdout)
        print(completed.stderr)
        raise SystemExit("Short Arena v2 test failed.")
