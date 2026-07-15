# UT Multi-Market Strategy Arena v2

## Install
Copy this bundle into the repository on the `v2-tradingview` branch.

## Download historical data
```cmd
python download_multi_market_research_data.py
```

## Run the offline package test
```cmd
python test_multi_market_strategy_arena_v2.py
```

## Run the real arena
```cmd
python multi_market_strategy_arena_v2.py --data-dir data
```

## Reports
- `reports\multi_market_strategy_arena_v2.json`
- `reports\multi_market_strategy_arena_v2.csv`
- `reports\multi_market_strategy_arena_v2_summary.txt`

The ranking rewards positive median and mean out-of-sample return, profit factor,
cross-symbol consistency, and controlled drawdown. Top candidates still require
untouched holdout testing and paper trading.

Research only. Historical data requests may be made; no order client is created,
no order is submitted, and no production strategy is changed.
