"""Research-only GLD Feature Engineering v2 comparison. Cannot place trades."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from data import get_market_data

SYMBOL = "GLD"
LOOKBACK_DAYS = 5000
HORIZON = 20
FOLDS = 6
MIN_TRAIN = 750
INITIAL_CAPITAL = 10_000.0
POSITION_PERCENT = 0.10
THRESHOLD = 0.50
SLIPPAGE = 0.0005
STOP = 0.10
TARGET = 0.20
MAX_HOLD = 20
RANDOM_STATE = 42

REPORT_DIR = Path("reports")
JSON_PATH = REPORT_DIR / "feature_v2_model_comparison.json"
CSV_PATH = REPORT_DIR / "feature_v2_model_comparison_folds.csv"

FEATURES = [
    "return_1d","return_5d","return_10d","return_20d","return_60d",
    "price_vs_ema200","ema9_vs_ema21","ema21_vs_ema50",
    "rsi_14","rsi_7","atr_pct","volatility_20d","volatility_ratio",
    "downside_volatility_20d","volume_change","volume_ma_ratio",
    "volume_zscore_20d","macd_pct","macd_signal_pct","macd_hist_pct",
    "bollinger_position","bollinger_width","distance_52w_high",
    "distance_52w_low","ema50_slope_20d","ema200_slope_20d","adx_14",
]


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - frame["close"].shift()).abs(),
        (frame["low"] - frame["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()


def adx(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    current_atr = atr(frame, length)
    plus_di = 100 * plus_dm.ewm(alpha=1/length, min_periods=length, adjust=False).mean() / current_atr
    minus_di = 100 * minus_dm.ewm(alpha=1/length, min_periods=length, adjust=False).mean() / current_atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/length, min_periods=length, adjust=False).mean()


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    f = raw.copy()
    c, v = f["close"], f["volume"]

    for days in [1, 5, 10, 20, 60]:
        f[f"return_{days}d"] = c.pct_change(days)

    for span in [9, 21, 50, 200]:
        f[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()

    f["price_vs_ema200"] = (c - f["ema_200"]) / f["ema_200"]
    f["ema9_vs_ema21"] = (f["ema_9"] - f["ema_21"]) / f["ema_21"]
    f["ema21_vs_ema50"] = (f["ema_21"] - f["ema_50"]) / f["ema_50"]
    f["rsi_14"], f["rsi_7"] = rsi(c, 14), rsi(c, 7)
    f["atr_pct"] = atr(f, 14) / c
    f["volatility_20d"] = f["return_1d"].rolling(20).std()
    vol60 = f["return_1d"].rolling(60).std()
    f["volatility_ratio"] = f["volatility_20d"] / vol60
    f["downside_volatility_20d"] = f["return_1d"].where(f["return_1d"] < 0, 0).rolling(20).std()

    vol_mean, vol_std = v.rolling(20).mean(), v.rolling(20).std()
    f["volume_change"] = v.pct_change()
    f["volume_ma_ratio"] = v / vol_mean
    f["volume_zscore_20d"] = (v - vol_mean) / vol_std

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd_pct"] = macd / c
    f["macd_signal_pct"] = signal / c
    f["macd_hist_pct"] = (macd - signal) / c

    middle, std = c.rolling(20).mean(), c.rolling(20).std()
    upper, lower = middle + 2*std, middle - 2*std
    width = (upper - lower).replace(0, np.nan)
    f["bollinger_position"] = (c - lower) / width
    f["bollinger_width"] = width / middle

    high252, low252 = c.rolling(252).max(), c.rolling(252).min()
    f["distance_52w_high"] = c / high252 - 1
    f["distance_52w_low"] = c / low252 - 1
    f["ema50_slope_20d"] = f["ema_50"] / f["ema_50"].shift(20) - 1
    f["ema200_slope_20d"] = f["ema_200"] / f["ema_200"].shift(20) - 1
    f["adx_14"] = adx(f, 14)
    return f


def prepare() -> pd.DataFrame:
    f = add_features(get_market_data(symbol=SYMBOL, lookback_days=LOOKBACK_DAYS))
    f["future_close"] = f["close"].shift(-HORIZON)
    f = f[f["future_close"].notna()].copy()
    f["target"] = (f["future_close"] > f["close"]).astype(int)
    f[FEATURES] = f[FEATURES].replace([np.inf, -np.inf], np.nan)
    return f.dropna(subset=FEATURES + ["timestamp","open","high","low","close","target"]).sort_values("timestamp").reset_index(drop=True)


def folds(frame: pd.DataFrame):
    size = (len(frame) - MIN_TRAIN) // FOLDS
    result = []
    for n in range(FOLDS):
        start = MIN_TRAIN + n * size
        end = len(frame) if n == FOLDS - 1 else start + size
        train, test = frame.iloc[:start].copy(), frame.iloc[start:end].copy()
        if train["target"].nunique() == 2 and test["target"].nunique() == 2:
            result.append((train, test))
    return result


def models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,max_depth=6,min_samples_leaf=20,
            class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=350,max_depth=3,learning_rate=0.025,
            subsample=0.8,colsample_bytree=0.8,min_child_weight=10,
            reg_alpha=0.1,reg_lambda=2.0,objective="binary:logistic",
            eval_metric="logloss",random_state=RANDOM_STATE,n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,max_depth=5,learning_rate=0.02,
            num_leaves=31,min_child_samples=40,subsample=0.8,
            colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=2.0,
            class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1,verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=350,depth=5,learning_rate=0.025,l2_leaf_reg=6.0,
            loss_function="Logloss",eval_metric="AUC",auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,verbose=False,thread_count=-1,
        ),
    }


def probability_up(model, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    return np.asarray(model.predict_proba(X)[:, classes.index(1)], dtype=float)


def simulate(model, test: pd.DataFrame) -> dict[str, Any]:
    X = test[FEATURES]
    pred = np.asarray(model.predict(X), dtype=int)
    prob = probability_up(model, X)

    cash, shares = INITIAL_CAPITAL, 0.0
    entry_price = entry_value = entry_index = None
    trades, curve, invested = [], [], 0

    for i, ((_, row), p, pr) in enumerate(zip(test.iterrows(), pred, prob)):
        close, low, high = float(row["close"]), float(row["low"]), float(row["high"])

        if shares == 0:
            if p == 1 and pr >= THRESHOLD:
                allocation = cash * POSITION_PERCENT
                execution = close * (1 + SLIPPAGE)
                shares = allocation / execution
                cash -= allocation
                entry_price, entry_value, entry_index = execution, allocation, i
        else:
            invested += 1
            held = i - int(entry_index)
            stop_price = float(entry_price) * (1 - STOP)
            target_price = float(entry_price) * (1 + TARGET)
            exit_price = stop_price if low <= stop_price else target_price if high >= target_price else close if held >= MAX_HOLD else None

            if exit_price is not None:
                proceeds = shares * exit_price * (1 - SLIPPAGE)
                cash += proceeds
                trades.append((proceeds - float(entry_value)) / float(entry_value))
                shares, entry_price, entry_value, entry_index = 0.0, None, None, None

        curve.append(cash + shares * close)

    if shares > 0:
        proceeds = shares * float(test.iloc[-1]["close"]) * (1 - SLIPPAGE)
        cash += proceeds
        trades.append((proceeds - float(entry_value)) / float(entry_value))
        curve[-1] = cash

    arr = np.asarray(curve, dtype=float)
    max_dd = float((arr / np.maximum.accumulate(arr) - 1).min()) if arr.size else 0.0
    wins, losses = [x for x in trades if x > 0], [x for x in trades if x < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)

    return {
        "total_return": float(cash / INITIAL_CAPITAL - 1),
        "maximum_drawdown": max_dd,
        "trades": len(trades),
        "win_rate": float(np.mean(np.asarray(trades) > 0)) if trades else 0.0,
        "profit_factor": float(pf),
        "exposure": float(invested / max(1, len(test))),
    }


def summarize(rows):
    returns = np.asarray([r["total_return"] for r in rows], dtype=float)
    drawdowns = np.asarray([r["maximum_drawdown"] for r in rows], dtype=float)
    finite_pf = [min(float(r["profit_factor"]), 10.0) for r in rows if math.isfinite(float(r["profit_factor"]))]
    return {
        "folds": len(rows),
        "positive_folds": int((returns > 0).sum()),
        "median_fold_return": float(np.median(returns)),
        "mean_fold_return": float(np.mean(returns)),
        "worst_fold_return": float(np.min(returns)),
        "best_fold_return": float(np.max(returns)),
        "total_trades": int(sum(r["trades"] for r in rows)),
        "mean_win_rate": float(np.mean([r["win_rate"] for r in rows])),
        "mean_profit_factor": float(np.mean(finite_pf)) if finite_pf else 0.0,
        "mean_maximum_drawdown": float(np.mean(drawdowns)),
        "worst_maximum_drawdown": float(np.min(drawdowns)),
        "mean_model_accuracy": float(np.mean([r["model_accuracy"] for r in rows])),
        "mean_model_roc_auc": float(np.mean([r["model_roc_auc"] for r in rows])),
    }


def score(s):
    return float(
        4*s["median_fold_return"] + 2*s["mean_fold_return"]
        + 0.015*s["positive_folds"]
        + 0.010*min(3.0, s["mean_profit_factor"])
        + 0.10*(s["mean_model_roc_auc"] - 0.50)
        + 0.015*min(1.0, s["total_trades"]/50.0)
        + 1.5*s["worst_maximum_drawdown"]
    )


def run():
    frame = prepare()
    rows = []

    for fold_number, (train, test) in enumerate(folds(frame), start=1):
        print(f"\nFold {fold_number}: {test.iloc[0]['timestamp']} through {test.iloc[-1]['timestamp']}")
        for name, model in models().items():
            print(f"  Training {name}...")
            model.fit(train[FEATURES], train["target"])
            pred = np.asarray(model.predict(test[FEATURES]), dtype=int)
            prob = probability_up(model, test[FEATURES])
            result = simulate(model, test)
            row = {
                "fold": fold_number,
                "model": name,
                "model_accuracy": float(accuracy_score(test["target"], pred)),
                "model_roc_auc": float(roc_auc_score(test["target"], prob)),
                **result,
            }
            rows.append(row)
            print(f"    return={result['total_return']:.2%}, trades={result['trades']}, auc={row['model_roc_auc']:.3f}, max_dd={result['maximum_drawdown']:.2%}")

    summaries = {}
    for name in models():
        summary = summarize([r for r in rows if r["model"] == name])
        summary["research_score"] = score(summary)
        summaries[name] = summary

    ranking = sorted(summaries, key=lambda name: summaries[name]["research_score"], reverse=True)
    report = {
        "features": FEATURES,
        "summaries": summaries,
        "recommendation": {
            "research_winner": ranking[0],
            "ranking": ranking,
            "winner_details": summaries[ranking[0]],
            "note": "Research only. Compare against the prior 13-feature LightGBM result before deployment.",
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    print("\n=== FEATURE V2 RESULT ===")
    print(json.dumps(report["recommendation"], indent=2))
    print(f"\nJSON report: {JSON_PATH}")
    print(f"CSV report: {CSV_PATH}")


if __name__ == "__main__":
    run()
