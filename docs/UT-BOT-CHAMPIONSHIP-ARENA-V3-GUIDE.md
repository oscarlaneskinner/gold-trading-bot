# UT Bot Championship Arena v3

Includes a large configurable parameter grid, multi-core execution, walk-forward validation, cross-market scoring, Monte Carlo robustness analysis, TradingView finalist export, heatmap-ready CSVs, and automatic rejection of weak candidates.

## Install
```cmd
pip install -r requirements_arena_v3.txt
```

## Test
```cmd
python test_arena_v3.py
```

## Trial runs
```cmd
python arena_v3.py --data-dir data --limit 1000
python arena_v3.py --data-dir data --limit 10000
```

## Full run
```cmd
python arena_v3.py --data-dir data
```

The full default grid is about 1.08 million combinations and may take many hours or longer. Increase gradually and monitor temperature, memory, and runtime.

## Outputs
- reports\arena_v3_results.json
- reports\arena_v3_leaderboard.csv
- reports\arena_v3_top_100.csv
- reports\arena_v3_tradingview_finalists.csv
- reports\heatmaps\heatmap_score.csv
- reports\heatmaps\heatmap_median_test_return_percent.csv
- reports\heatmaps\heatmap_median_profit_factor.csv

Large searches can increase overfitting risk. Finalists still require untouched holdout testing, TradingView verification, and paper trading. No trading client is created, no order is submitted, and production remains unchanged.
