# Backup and Restore

## Critical files

Back up:

```text
models/
data/trade_memory.sqlite3
reports/
logs/
config.py
daily_bot.py
risk_manager.py
strategy.py
.github/workflows/
```

## Recommended backup method

1. Confirm all changes are committed.
2. Push `main` to GitHub.
3. Create the `v1.0.0` tag.
4. Download a ZIP archive of the release.
5. Save a separate copy of:
   - `data/trade_memory.sqlite3`
   - `models/`
   - `reports/`

## Restore from GitHub

```powershell
cd C:\GitHub
git clone <YOUR_REPOSITORY_URL> gold-trading-bot-restored
cd gold-trading-bot-restored
git checkout v1.0.0
```

Then recreate the virtual environment and restore paper secrets in GitHub.

## Database verification after restore

Run:

1. Test GLD Research Data Audit
2. Inspect GLD Trade Memory
3. Inspect GLD Trade Intelligence
4. Run GLD Core Regression Tests
