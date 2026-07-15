# Safety Checklist

Before running a trading workflow:

- [ ] Alpaca credentials are paper-account credentials.
- [ ] `PAPER_TRADING` is `True`.
- [ ] The intended position size is displayed.
- [ ] No unintended GLD position exists.
- [ ] No unintended GLD buy order is open.
- [ ] The selected workflow is correct.
- [ ] The workflow is run only once.

Before promoting a strategy:

- [ ] At least 30 candidate trades are available.
- [ ] At least six test folds are available.
- [ ] At least four folds are positive.
- [ ] Drawdown is within the allowed limit.
- [ ] Profit factor is not materially worse.
- [ ] Baseline comparison is complete.
- [ ] Statistical screening passed.
- [ ] Human review is complete.
- [ ] Extended paper testing is planned.

Before real-money consideration:

- [ ] A meaningful paper-trading history exists.
- [ ] Execution slippage has been measured.
- [ ] Outage and recovery procedures are tested.
- [ ] Maximum daily and portfolio losses are enforced.
- [ ] Legal, tax, and broker requirements have been reviewed.
