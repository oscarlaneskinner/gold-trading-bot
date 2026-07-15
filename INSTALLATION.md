# Installation

## Requirements

- Windows 10 or 11
- Python 3.13
- Git
- GitHub Desktop
- Visual Studio Code
- Alpaca paper-trading account

## Repository setup

```powershell
cd C:\GitHub
git clone <YOUR_REPOSITORY_URL>
cd gold-trading-bot
```

## Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_models.txt
```

## GitHub secrets

The repository requires paper-account credentials:

```text
ALPACA_API_KEY
ALPACA_SECRET_KEY
```

Use only the keys shown in Alpaca's paper-trading section.

## Initial validation

Run these GitHub workflows:

1. Audit GLD Python Dependencies
2. Run GLD Core Regression Tests
3. Test GLD Research Data Audit
4. Test GLD Phase 7 Complete Bundle
5. Validate GLD Version 1.0 Release
