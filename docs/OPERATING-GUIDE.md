# Operating Guide

## Normal paper-trading workflows

### Daily AI Trading Bot v4

- Uses the 10% production research allocation.
- Reads the current model and market data.
- May place one paper order when all entry safeguards pass.

### GLD 15 Percent Paper Test

- Uses the 15% research allocation.
- Manual only.
- Intended for controlled paper comparison.

## Research workflows

These do not place orders:

- Inspect GLD Trade Memory
- Synchronize GLD Trade Memory
- Review GLD Trade Memory
- Build GLD Performance Dashboard
- Discover GLD Trading Patterns
- Build GLD Weekly Research Report
- Audit GLD Research Data
- Inspect GLD Trade Intelligence
- Run GLD Phase 7 Master Workflow

## Daily operating routine

1. Check the Alpaca paper account.
2. Run only the intended trading workflow.
3. Do not rerun a trading workflow unnecessarily.
4. Confirm the action, order ID, and duplicate-order checks.
5. Run the dashboard or research reports as needed.

## Promotion policy

No model, threshold, sizing rule, or strategy should be promoted because of
one favorable result. A candidate must pass:

- minimum trade-count rules,
- fold stability,
- drawdown constraints,
- baseline comparison,
- statistical screening,
- human review.
