# UT Multi-Filter Championship

This package adds 20 controlled UT-style multi-filter competitors.

## First competitors to test

1. UT + EMA200 + RSI
2. UT + EMA200 + MACD
3. UT + EMA200 + Supertrend
4. UT + RSI + MACD
5. UT + EMA200 + RSI + MACD
6. UT + EMA200 + RSI + Relative Volume
7. UT + EMA200 + RSI + MACD + Relative Volume

## Fair comparison rules

Keep these identical:

- GLD
- 1D timeframe
- Nov. 18, 2004 through Jul. 15, 2026
- $100,000 starting capital
- 0.01% commission
- one unit of slippage
- 15% position size
- UT sensitivity 1.0
- ATR period 10
- hard stop 3%
- take profit 8%
- maximum hold 20 bars

Change only the Competitor field.

## Guided registration

Run:

```powershell
python multi_filter_research_wizard.py
```

The wizard registers the experiment and refreshes the leaderboard, championship summary, and command-center reports.

## Championship report

Run:

```powershell
python ut_multi_filter_manager.py
```

Output:

```text
reports/ut_multi_filter_championship.json
```

## Safety

Research only. No market request, broker connection, order submission, or automatic promotion.
