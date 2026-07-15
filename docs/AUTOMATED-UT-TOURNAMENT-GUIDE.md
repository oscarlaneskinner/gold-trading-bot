# Automated UT Tournament

## What this solves

TradingView's Strategy Report runs the strategy currently loaded on the chart.
This bundle runs many UT variants in one local Python batch from a single
TradingView OHLCV CSV export.

## Important accuracy note

The Python engine approximates the Pine strategy and execution assumptions.
TradingView's broker emulator can differ in bar-order fills, slippage handling,
and other details. Use the automated batch to screen competitors, then validate
the top finalists in TradingView before any paper-trading promotion.

## Export data from TradingView

On the GLD daily chart:

1. Open the chart menu.
2. Choose **Download chart data**.
3. Export all available GLD daily OHLCV history.
4. Save it as:

```text
data/GLD_1D.csv
```

TradingView may use different column capitalization. The importer accepts
common names such as Time, Open, High, Low, Close, and Volume.

## Install dependencies

```powershell
pip install numpy pandas
```

## Run the entire tournament

```powershell
python automated_ut_tournament.py --csv data/GLD_1D.csv
```

It creates:

```text
reports/automated_ut_tournament.json
reports/automated_ut_tournament.csv
```

## Recommended workflow

1. Batch-test every competitor locally.
2. Select the top 3 to 5 by score and risk.
3. Validate only those finalists in TradingView Strategy Report.
4. Register validated results in the permanent research database.
5. Promote nothing automatically.

## Safety

The automation:

- reads a local CSV,
- makes no market request,
- uses no broker credentials,
- submits no order,
- does not alter production strategy.
