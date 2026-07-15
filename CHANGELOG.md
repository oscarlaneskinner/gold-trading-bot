# Changelog

All notable changes to the GLD AI Trading Research Platform are documented here.

## [1.0.0] - 2026-07-15

### Added

- LightGBM-based GLD paper-trading engine.
- Configurable 10% production and 15% paper-research position sizing.
- Alpaca paper-account execution and position tracking.
- Duplicate-order and duplicate-position protection.
- Stop-loss, take-profit, model-exit, and holding-period controls.
- Persistent SQLite trade-memory database.
- Decision and trade synchronization with filled Alpaca orders.
- Trade grading and performance summaries.
- Read-only HTML and JSON performance dashboards.
- Market-regime, trend, volatility, momentum, and confidence classifications.
- Pattern-discovery reports with minimum sample-size safeguards.
- Weekly research reports.
- Research-data integrity auditing.
- Dependency auditing and core regression testing.
- Strategy Laboratory experiment registry.
- Strategy leaderboard, baseline comparison, statistical screening,
  and advisory promotion recommendations.

### Safety

- Paper trading remains mandatory.
- Research workflows cannot place orders.
- Strategy promotions are advisory only.
- Human approval remains mandatory before production changes.

### Known limitations

- The current live paper history contains too few completed trades for
  evidence-based strategy promotion.
- The first synchronized trade predates full feature-memory integration,
  so some entry-context fields are unavailable.
- The platform currently focuses on GLD and daily/swing decisions.
- Intraday and higher-frequency strategies require separate data,
  execution, and infrastructure validation.
