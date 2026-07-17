# Research Database and Tournament Manager

## Purpose

Store comparable strategy results in one SQLite research database and rank
them only when their asset, timeframe, and assumptions are compatible.

## Required fields

Each record includes:

- asset and venue,
- strategy and variant,
- timeframe,
- date range,
- starting capital,
- commission and slippage,
- position size,
- net profit,
- maximum drawdown,
- win rate,
- profit factor,
- closed trades,
- source platform,
- version and notes.

## Register the current GLD UT Original result

```powershell
python register_research_result.py `
  --asset GLD `
  --venue "NYSE Arca" `
  --strategy-name "UT Competition Lab" `
  --strategy-variant "UT Original" `
  --timeframe "1D" `
  --date-start "2004-11-18" `
  --date-end "2026-07-15" `
  --starting-capital 100000 `
  --commission-percent 0.01 `
  --slippage-units 1 `
  --position-percent 15 `
  --net-profit-amount 26042.87 `
  --net-profit-percent 26.04 `
  --max-drawdown-amount 6047.02 `
  --max-drawdown-percent 5.17 `
  --profitable-trades-percent 40.79 `
  --profit-factor 1.431 `
  --closed-trades 407 `
  --source-platform TradingView `
  --strategy-version 2.1
```

## Build a fair GLD daily leaderboard

```powershell
python build_research_leaderboard.py `
  --asset GLD `
  --timeframe 1D `
  --minimum-trades 30
```

## Safety

This package:

- does not connect to TradingView,
- does not connect to Alpaca,
- does not place orders,
- does not alter Version 1.0,
- stores research results only.
