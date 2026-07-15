"""Register one research result from TradingView or Python."""

from __future__ import annotations

import argparse
import json

from research_tournament_db import (
    database_summary,
    register_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--asset", required=True)
    parser.add_argument("--venue", default="")
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--strategy-variant", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--starting-capital", type=float, required=True)
    parser.add_argument("--commission-percent", type=float, required=True)
    parser.add_argument("--slippage-units", type=float, required=True)
    parser.add_argument("--position-percent", type=float, required=True)
    parser.add_argument("--net-profit-amount", type=float)
    parser.add_argument("--net-profit-percent", type=float, required=True)
    parser.add_argument("--max-drawdown-amount", type=float)
    parser.add_argument("--max-drawdown-percent", type=float, required=True)
    parser.add_argument("--profitable-trades-percent", type=float, required=True)
    parser.add_argument("--profit-factor", type=float, required=True)
    parser.add_argument("--closed-trades", type=int, required=True)
    parser.add_argument("--market-regime", default="")
    parser.add_argument("--source-platform", required=True)
    parser.add_argument("--strategy-version", default="")
    parser.add_argument("--notes", default="")

    return parser.parse_args()


def run() -> None:
    args = parse_args()

    record = register_experiment(
        {
            "asset": args.asset,
            "venue": args.venue,
            "strategy_name": args.strategy_name,
            "strategy_variant": args.strategy_variant,
            "timeframe": args.timeframe,
            "date_start": args.date_start,
            "date_end": args.date_end,
            "starting_capital": args.starting_capital,
            "commission_percent": args.commission_percent,
            "slippage_units": args.slippage_units,
            "position_percent": args.position_percent,
            "net_profit_amount": args.net_profit_amount,
            "net_profit_percent": args.net_profit_percent,
            "max_drawdown_amount": args.max_drawdown_amount,
            "max_drawdown_percent": args.max_drawdown_percent,
            "profitable_trades_percent":
                args.profitable_trades_percent,
            "profit_factor": args.profit_factor,
            "closed_trades": args.closed_trades,
            "market_regime": args.market_regime,
            "source_platform": args.source_platform,
            "strategy_version": args.strategy_version,
            "notes": args.notes,
        }
    )

    print("Research result registered")
    print(
        json.dumps(
            {
                "record": record,
                "database_summary": database_summary(),
                "market_request_made": False,
                "order_submitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
