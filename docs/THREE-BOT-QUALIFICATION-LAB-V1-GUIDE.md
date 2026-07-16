# Three-Bot Qualification Lab v1

This package qualifies one real research strategy for each planned role:

- GLD specialist
- long opportunity strategy
- short opportunity strategy

## What it does

1. Registers the TradingView-verified GLD finalist `ARENA-0081`.
2. Selects the strongest eligible real long strategy already stored in the Hall of Fame.
3. Runs a dedicated short-side arena using local scanner CSV data.
4. Registers the best qualifying short candidate.
5. Creates a three-role qualification report.

## Install

```cmd
pip install -r requirements_three_bot_qualification_lab_v1.txt
```

## Offline test

```cmd
python test_three_bot_qualification_lab_v1.py
```

## Run with real scanner data

```cmd
python three_bot_qualification_lab_v1.py --data-dir data\scanner
```

## Reports

```text
reports\qualification\three_bot_qualification_lab_v1.json
reports\qualification\qualified_three_bot_roles.csv
reports\qualification\short_arena_leaderboard.csv
```

## Decision rule

The report will show either:

```text
THREE_BOT_SHADOW_CONTROLLER
```

or:

```text
CONTINUE_QUALIFICATION
```

Only when all three roles qualify should the next shadow-mode controller be built.

## Safety

- research only,
- no Alpaca trading client,
- no order submission,
- no production-strategy changes.
