# Daily Competition Email Setup

## What it sends

The email includes:

- current tournament champion,
- overall leaderboard,
- return, drawdown, profit factor, and trade count,
- recent experiments,
- database totals.

The recipient is:

```text
oscarlaneskinner@yahoo.com
```

## Yahoo security requirement

Do not place your normal Yahoo password in GitHub.

Create a Yahoo third-party app password and store it as a GitHub secret.

## GitHub secrets

Repository:

```text
Settings > Secrets and variables > Actions
```

Create:

```text
YAHOO_EMAIL_USERNAME
```

Value:

```text
oscarlaneskinner@yahoo.com
```

Create:

```text
YAHOO_EMAIL_APP_PASSWORD
```

Value:

```text
the Yahoo-generated app password
```

Do not commit the app password into any file.

## Schedule

The included workflow uses:

```text
30 22 * * 1-5
```

That is 22:30 UTC on weekdays, approximately:

- 6:30 PM Eastern during daylight-saving time,
- 5:30 PM Eastern during standard time.

GitHub Actions schedules use UTC and may start several minutes late.

## First test

Because the workflow currently lives on the `v2-tradingview` research branch,
run these local tests first:

```powershell
python test_research_command_center.py
python research_command_center.py
```

After the email workflow is available on the default branch or a dedicated
deployment repository, use **Run workflow** to send a test email.

## Safety

This workflow:

- reads the tournament database,
- creates reports,
- sends email,
- does not request market data,
- does not submit orders.
