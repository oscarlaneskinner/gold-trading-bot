# UT Alternative Indicators Championship

This single bundle includes research competitors based on:

- an original B-Xtrender-style momentum approximation,
- linear-regression trend filters,
- smoothed Heiken Ashi filters,
- Average Daily Range volatility filters and targets,
- combinations of all four.

## Important licensing note

The included B-Xtrender-style implementation is an original approximation
using common RSI, EMA, and momentum concepts. It does not copy or republish
QuantTherapy's Pine source code.

## Suggested testing order

1. UT + B-Xtrender Style
2. UT + Linear Regression Trend
3. UT + Smoothed Heiken Ashi
4. UT + ADR Volatility
5. UT + EMA200 + B-Xtrender Style
6. UT + B-Xtrender Style + Linear Regression
7. UT + B-Xtrender Style + Smoothed Heiken Ashi
8. UT + All Alternative Filters

## Fair comparison rules

Keep GLD, 1D, dates, starting capital, commission, slippage, position size,
UT sensitivity, ATR period, stop, target, and maximum hold unchanged.
Change only the Competitor field.

Run:

```powershell
python alternative_indicator_research_wizard.py
```

to register results and refresh the leaderboard and reports.

Research only. No broker connection, market request, order submission, or
automatic promotion.
