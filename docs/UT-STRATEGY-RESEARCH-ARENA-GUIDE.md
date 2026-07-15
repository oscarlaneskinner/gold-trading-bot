# UT Strategy Research Arena

This bundle screens more than 1,000 controlled combinations using:

- 16 filter sets,
- 3 UT sensitivities,
- 3 ATR periods,
- 3 holding periods,
- 3 risk profiles.

It uses a chronological 70/30 train/test split and ranks candidates by
out-of-sample return, drawdown, profit factor, trade count, and robustness.

## Data choices

### Use existing local CSV

```powershell
python strategy_research_arena.py --csv data\GLD_1D.csv
```

### Download from Alpaca

Set the existing Alpaca data credentials, then run:

```powershell
python download_alpaca_research_data.py
python strategy_research_arena.py --csv data\GLD_1D.csv
```

The Alpaca downloader requests historical market data only. It never submits
orders.

## Outputs

```text
reports\strategy_research_arena.json
reports\strategy_research_arena.csv
```

Only the top finalists should be re-tested in TradingView. The arena is a
screening system, not proof of future profitability and not a replacement for
walk-forward paper trading.

## Safety

- no automatic strategy promotion,
- no broker-order client,
- no order submission,
- production strategy remains unchanged.
