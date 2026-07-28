# TradingView Research Setup

The Python GLD bot uses LightGBM. Pine Script cannot load the Python model directly, so this script is an independent, interpretable competitor.

1. Open TradingView and chart `AMEX:GLD`.
2. Open Pine Editor.
3. Create a new strategy.
4. Paste `gld_research_strategy.pine`.
5. Save and add it to the chart.
6. Open Strategy Tester.
7. Start with 15% position size, 10% stop, 20% target, 5% trailing stop, and 20 bars maximum hold.
8. Create notification-only alerts first.
9. Do not enter Alpaca credentials in TradingView.
10. Do not add a webhook URL in Package 1.

TradingView alert triggers must still be created manually in the Create Alert dialog.
