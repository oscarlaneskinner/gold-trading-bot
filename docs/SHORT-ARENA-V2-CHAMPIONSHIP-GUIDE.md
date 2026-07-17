# Short Arena v2 Championship Edition

This package searches multiple short-strategy families and promotes only a
qualifying short candidate into the Strategy Hall of Fame.

## Strategy families

- Breakdown
- Failed Rally
- EMA Rejection
- Relative Weakness
- Lower High
- Donchian Breakdown
- ATR Breakdown
- Volume Reversal
- RSI Failure
- MACD Bear Cross

## Install

```cmd
pip install -r requirements_short_arena_v2.txt
```

## Offline test

```cmd
python test_short_arena_v2.py
```

## Safe trial run

```cmd
python short_arena_v2.py --data-dir data --limit 1000
```

## Larger run

```cmd
python short_arena_v2.py --data-dir data --limit 10000
```

## Full configured run

```cmd
python short_arena_v2.py --data-dir data
```

The full configuration is very large. Use trial limits first.

## Reports

```text
reports\short_arena_v2\short_arena_v2_results.json
reports\short_arena_v2\short_arena_v2_leaderboard.csv
reports\short_arena_v2\short_arena_v2_top_100.csv
```

## Promotion rule

The best candidate is inserted into:

```text
data\strategy_hall_of_fame.sqlite3
```

only when it passes the configured cross-market qualification thresholds.

## Safety

- no market requests,
- no trading client,
- no order submission,
- no production strategy changes.
