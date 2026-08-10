# AI Trading Bot (GLD and beyond) — Read This First

## ⚠️ Important disclaimer — please actually read this

This software is a **rules-based trading strategy experiment**, not proven investment
advice. In extensive testing (documented in the `reports/` folder), most configurations
of this strategy either **failed to beat, or only roughly matched, simply buying and
holding the underlying asset.** A "PASSED" result in any report means a specific
configuration beat buy-and-hold in a specific historical backtest — it does **not**
mean it will do so in the future.

- Past performance does not guarantee future results.
- Markets involve real risk of loss. Only trade with money you can afford to lose.
- This is not financial advice, and the author is not a licensed financial advisor.
- **Start with paper trading** (fake money, real market data) until you understand
  exactly what the bot does and have reviewed its historical performance yourself.

Treat this as a learning tool and a starting point for your own experimentation —
not a proven source of income.

---

## What this is

An automated trading system that:
1. Pulls historical price data for a stock/ETF
2. Trains a machine learning model to predict short-term price direction
3. Backtests the strategy honestly — **every result is automatically compared
   against simply buying and holding the same asset**, so you always see whether
   the added complexity is actually worth it
4. Optionally places paper (simulated) trades through Alpaca based on the model's
   predictions

## What you'll need

This was built using several outside tools and services. You'll need accounts
with some of these to use the software fully:

| Tool | What it's for | Required? | Link |
|---|---|---|---|
| **Python 3.11+** | Runs the actual bot code | Yes | https://www.python.org/downloads/ |
| **Alpaca** | Brokerage — provides market data and (paper) trade execution | Yes | https://alpaca.markets |
| **Git** | Downloads/updates the code | Recommended | https://git-scm.com |
| **GitHub account** | Optional — only needed if you want to run this on a schedule in the cloud instead of your own computer | Optional | https://github.com |
| **TradingView** | Optional — useful for charting and visually reviewing price action alongside the bot's decisions | Optional | https://www.tradingview.com |

You do **not** need to pay for any of these to get started — Alpaca's paper trading
and free-tier market data, Python, and Git are all free.

---

## Setup — step by step

### 1. Install Python

Download and install from https://www.python.org/downloads/ (Windows/Mac/Linux
all supported). During install on Windows, check the box that says
**"Add Python to PATH."**

### 2. Get the code

If you received this as a folder/zip, extract it somewhere easy to find (like your
Desktop). If you're using Git:
```
git clone <repository-url>
cd gold-trading-bot
```

### 3. Create a virtual environment

Open a terminal (Command Prompt on Windows) in the project folder:
```
python -m venv venv
```

Activate it:
- **Windows**: `venv\Scripts\activate.bat`
- **Mac/Linux**: `source venv/bin/activate`

You should see `(venv)` appear at the start of your terminal prompt.

### 4. Install dependencies

```
pip install -r requirements.txt
```

### 5. Get your Alpaca API keys

1. Sign up at https://alpaca.markets
2. Make sure you're viewing your **Paper Trading** account (there's a toggle in
   the dashboard — start here, not Live Trading)
3. Find the **API Keys** section and generate a new key pair
4. Copy both the **API Key ID** and **Secret Key** somewhere safe — the secret is
   only shown once

### 6. Configure your keys

Copy `.env.example` to a new file named `.env` in the same folder:
```
copy .env.example .env        (Windows)
cp .env.example .env          (Mac/Linux)
```

Open `.env` in a text editor and replace the placeholder values with your real
Alpaca keys:
```
ALPACA_API_KEY=your_actual_key_here
ALPACA_SECRET_KEY=your_actual_secret_here
```

**Never share this file or paste its contents anywhere public** — anyone with
these values could access your Alpaca account.

### 7. Train the model

```
python train_model.py
```

This trains a model on historical data and saves it to `models/model.pkl`.

### 8. Test the strategy honestly before trusting it

```
python walk_forward_backtest.py
```

This shows you real, out-of-sample results — including the buy-and-hold
comparison — for the current configuration. **Read this output before running
anything live.**

### 9. Run the bot

```
python daily_bot.py
```

This checks current market conditions, gets the model's prediction, and (if
conditions are met) places a paper trade through Alpaca.

### 10. Automate it (optional)

To run this automatically every trading day, you have two options:
- **Your own computer**: use Windows Task Scheduler (or `cron` on Mac/Linux) to
  run `daily_bot.py` once each morning
- **The cloud**: if you have a GitHub account, this repo includes GitHub Actions
  workflows (`.github/workflows/`) that can run it automatically — see the
  comments in those files for setup

---

## Testing a new stock or configuration

Want to test this strategy on a different stock, or with different settings?

```
python create_bot.py
```

Edit the settings at the top of the script (symbol, stop-loss %, take-profit %,
hold days, model type), or set them as environment variables. This runs the same
honest, buy-and-hold-comparing backtest on whatever you choose — **always check
the result before trusting a new configuration.**

---

## Understanding the reports

Every backtest run saves a report to the `reports/` folder as a `.json` file.
Look for:
- `"passed": true/false` — did this configuration actually beat buy-and-hold?
- `"mean_outperformance_pct"` — by how much (positive = beat it, negative = lost to it)
- `"verdict"` — a plain-English summary

If you only remember one thing: **a positive average return does not mean the
strategy is good.** Always check it against the buy-and-hold number in the same
report.

---

## Getting help

- Alpaca's documentation: https://docs.alpaca.markets
- Python installation help: https://docs.python.org/3/using/index.html
- If something breaks, check the error message carefully — it usually tells you
  exactly which line and file the problem is in.

## A final note

This project was built iteratively, with a strong emphasis on honest testing
over impressive-looking results. Every strategy included here has been checked
against simply holding the asset, and most did not clearly win. That's not a
flaw in the software — it's the actual, honest finding, and a big part of why
this tool is more useful as a learning platform than as a promise of profit.
