# GLD AI Trading Research Platform v1.0.0

Version 1.0.0 is the first stable release of the GLD AI Trading Research Platform.

## What this release contains

The platform combines four layers:

1. **Trading**
   - LightGBM model
   - Alpaca paper execution
   - Configurable position sizing
   - Risk and exit management

2. **Memory**
   - SQLite decision history
   - Filled-order synchronization
   - Open and closed trade tracking

3. **Research**
   - Market-regime classification
   - Pattern discovery
   - Weekly reports
   - Trade review and grading

4. **Strategy Laboratory**
   - Experiment registry
   - Baseline metrics
   - Leaderboard
   - Statistical screening
   - Advisory promotion recommendations

## Release status

- Production baseline: LightGBM `lgbm_d`
- Symbol: GLD
- Trading mode: Paper only
- Phase 7 status: Ready
- Automatic promotion: Disabled
- Human review: Required

## Important notice

This software is a research and paper-trading system. Backtests and paper
results do not guarantee future performance. Real-money deployment should not
be considered until a meaningful paper-trading history, execution review,
operational monitoring, and risk limits have been completed.
