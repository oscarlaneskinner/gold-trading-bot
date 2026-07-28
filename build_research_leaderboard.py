"""Build a filtered research tournament leaderboard."""

from __future__ import annotations

import argparse
import json

from research_tournament_db import (
    build_leaderboard,
    database_summary,
)


def run() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--asset")
    parser.add_argument("--timeframe")
    parser.add_argument("--source-platform")
    parser.add_argument(
        "--minimum-trades",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    report = build_leaderboard(
        asset=args.asset,
        timeframe=args.timeframe,
        source_platform=args.source_platform,
        minimum_trades=args.minimum_trades,
    )

    print("Research tournament leaderboard")
    print(
        json.dumps(
            {
                "summary": database_summary(),
                "report": report,
            },
            indent=2,
        )
    )
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
