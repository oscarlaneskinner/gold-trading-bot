# Strategy Hall of Fame and Portfolio Commander v1

This package adds two connected systems.

## Strategy Hall of Fame

It reads existing Arena, scanner, and research reports and stores strategy
results in:

```text
data\strategy_hall_of_fame.sqlite3
```

It creates:

```text
reports\hall_of_fame\strategy_hall_of_fame.json
reports\hall_of_fame\strategy_hall_of_fame.csv
```

Run:

```cmd
python strategy_hall_of_fame_v1.py
```

## Portfolio Commander

It filters Hall of Fame strategies using score, profit factor, drawdown, and
consistency rules. It then creates a research-only allocation plan for up to
three roles:

- GLD specialist
- long opportunity strategy
- short opportunity strategy

Run:

```cmd
python portfolio_commander_v1.py
```

Reports:

```text
reports\portfolio\portfolio_commander_v1.json
reports\portfolio\portfolio_allocations.csv
```

## One-command workflow

```cmd
python run_championship_command_center_v1.py
```

## Important limitation

This version creates a suggested research allocation only. It does not check
live positions, short availability, margin requirements, buying power, or
current market conditions. It creates no trading client and submits no orders.
