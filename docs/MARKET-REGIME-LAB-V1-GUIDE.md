# Market Regime Lab v1

This complete version reads local SPY daily data and classifies the current
market environment.

## Install

```cmd
pip install -r requirements_market_regime_lab_v1.txt
```

## Test

```cmd
python test_market_regime_lab_v1.py
```

## Run with your existing data folder

```cmd
python market_regime_lab_v1.py --data-dir data
```

## Reports

```text
reports\market_regime\market_regime_lab_v1.json
reports\market_regime\market_regime_lab_v1_summary.txt
```

## Bot permissions

The report includes:

- whether the GLD bot may run,
- whether the long bot may run,
- whether the short bot may run,
- whether position size should be reduced.

This is research-only and does not submit orders.
