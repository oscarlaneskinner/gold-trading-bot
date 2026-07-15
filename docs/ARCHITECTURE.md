# Architecture

```text
Market Data
    |
    v
Feature Engineering
    |
    v
LightGBM Prediction
    |
    v
Trading Decision Engine
    |
    +--> Alpaca Paper Orders
    |
    +--> CSV Logs
    |
    +--> SQLite Trade Memory
              |
              +--> Trade Synchronization
              +--> Trade Review
              +--> Market Intelligence
              +--> Pattern Discovery
              +--> Weekly Research Reports
              +--> Data Integrity Audit
              +--> Performance Dashboard
              |
              v
        Strategy Laboratory
              |
              +--> Experiment Registry
              +--> Baseline Comparison
              +--> Leaderboard
              +--> Statistical Screening
              +--> Promotion Recommendation
```

## Core principle

The production trading engine and the research platform are separated.
Research may recommend changes, but it cannot deploy them automatically.
