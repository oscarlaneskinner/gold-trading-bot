# Two-Bot Shadow Controller v1

This package creates a daily research-only plan for the qualified GLD and long
strategies.

## Inputs

It reads:

```text
reports\market_regime\market_regime_lab_v1.json
reports\portfolio\portfolio_commander_v1.json
reports\hall_of_fame\strategy_hall_of_fame.json
reports\scanner\championship_scanner_v1.json
```

## Install

```cmd
pip install -r requirements_two_bot_shadow_controller_v1.txt
```

## Test

```cmd
python test_two_bot_shadow_controller_v1.py
```

## Run

First refresh the upstream reports, then run:

```cmd
python two_bot_shadow_controller_v1.py
```

## Reports

```text
reports\shadow\two_bot_shadow_controller_v1.json
reports\shadow\two_bot_shadow_proposals.csv
reports\shadow\two_bot_shadow_controller_v1_summary.txt
```

This version never contacts Alpaca and never submits orders.
