"""Inspect the research tournament database."""

from __future__ import annotations

import json

from research_tournament_db import (
    build_leaderboard,
    database_summary,
)


def run() -> None:
    print("Research tournament summary")
    print(
        json.dumps(
            {
                "summary": database_summary(),
                "overall_leaderboard":
                    build_leaderboard(),
                "market_request_made": False,
                "order_submitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
