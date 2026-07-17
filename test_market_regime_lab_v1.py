from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def create_market(
    path: Path,
    direction: str,
) -> None:
    rows = 320
    x = np.arange(rows)

    if direction == "bull":
        close = 100 + 0.18 * x
    elif direction == "bear":
        close = 180 - 0.18 * x
    elif direction == "volatile":
        close = 120 + 12 * np.sin(x / 2)
    else:
        close = 120 + 1.5 * np.sin(x / 20)

    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="D",
            ),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        }
    )

    frame.to_csv(
        path,
        index=False,
    )


with tempfile.TemporaryDirectory() as temporary:
    data_directory = Path(temporary)

    create_market(
        data_directory / "SPY_1D.csv",
        "bull",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "market_regime_lab_v1.py",
            "--data-dir",
            str(data_directory),
        ],
        capture_output=True,
        text=True,
    )

    report_path = Path(
        "reports/market_regime/"
        "market_regime_lab_v1.json"
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
        "regime": report.get("regime"),
        "json_report_created":
            report_path.exists(),
        "summary_created":
            Path(
                "reports/market_regime/"
                "market_regime_lab_v1_summary.txt"
            ).exists(),
        "market_request_made": False,
        "order_submitted": False,
    }

    print("Market Regime Lab v1 test")
    print(
        json.dumps(
            output,
            indent=2,
        )
    )
    print("No market request was made.")
    print("No order was submitted.")

    if (
        output["status"] != "passed"
        or output["regime"] != "BULL"
    ):
        print(completed.stdout)
        print(completed.stderr)
        raise SystemExit(
            "Market Regime Lab test failed."
        )
