# Eight-Day Automated Shadow Observation Suite v1

This packet runs the entire two-bot shadow pipeline once per weekday and
preserves eight successful daily observations.

## Included

- daily data refresh using the existing Alpaca historical-data downloader,
- Market Regime Lab,
- Strategy Hall of Fame,
- Portfolio Commander,
- Championship Market Scanner,
- Two-Bot Shadow Controller,
- dated report archives,
- observation-day counter,
- optional SMTP email,
- GitHub Actions weekday schedule,
- safety inspection that blocks order-related code.

## Install

Extract everything into:

```text
C:\GitHub\gold-trading-bot
```

Install:

```cmd
pip install -r requirements_eight_day_shadow_observation_v1.txt
```

Test:

```cmd
python test_eight_day_shadow_observation_v1.py
```

## Run locally

```cmd
python eight_day_shadow_observation_v1.py
```

To test without refreshing market data:

```cmd
python eight_day_shadow_observation_v1.py --skip-data-refresh
```

To restart the eight-day counter:

```cmd
python eight_day_shadow_observation_v1.py --reset
```

## GitHub schedule

The included workflow runs at 21:30 UTC Monday through Friday. In July, that is
5:30 PM Eastern Time.

The workflow commits dated observation reports back to the current branch.

## Required GitHub secrets

Already used by your project:

```text
ALPACA_API_KEY
ALPACA_SECRET_KEY
```

Optional email secrets:

```text
SHADOW_REPORT_EMAIL_TO
SHADOW_REPORT_EMAIL_FROM
SHADOW_REPORT_SMTP_HOST
SHADOW_REPORT_SMTP_PORT
SHADOW_REPORT_SMTP_USERNAME
SHADOW_REPORT_SMTP_PASSWORD
```

Without SMTP secrets, the workflow still runs and creates an email preview file.

## Reports

```text
reports\shadow_observation\observation_state.json
reports\shadow_observation\latest_summary.json
reports\shadow_observation\latest_summary.txt
reports\shadow_observation\latest_email_preview.txt
reports\shadow_observation\archive\YYYY-MM-DD\
```

## Safety

This suite is shadow-only:

- no Alpaca trading client,
- no paper order,
- no live order,
- no production strategy changes.

Eight days tests operational reliability. It does not prove profitability.
