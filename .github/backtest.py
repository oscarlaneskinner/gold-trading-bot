"""
Backtesting engine for Gold AI Trading Bot

Simulates historical trades without using real money.
"""

import pickle

import pandas as pd


from config import (
    SYMBOL,
    MODEL_PATH
)


from data import get_market_data

from features import (
    add_features,
    MODEL_FEATURES
)


from strategy import evaluate_trade



INITIAL_CAPITAL = 10000



def load_model():

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(file)



def run_backtest():

    print(
        "Starting backtest..."
    )


    df = get_market_data(
        SYMBOL,
        lookback_days=2500
    )


    df = add_features(df)


    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )


    model = load_model()


    cash = INITIAL_CAPITAL

    shares = 0

    trades = []


    for i in range(len(df)):


        row = df.iloc[i]


        X = (
            df[MODEL_FEATURES]
            .iloc[[i]]
        )


        prediction = int(
            model.predict(X)[0]
        )


        confidence = float(
            model.predict_proba(X)[0][1]
        )


        decision = evaluate_trade(

            prediction,

            confidence,

            row

        )


        action = decision["action"]



        # BUY

        if action == "BUY" and shares == 0:


            shares = (
                cash /
                row["close"]
            )


            cash = 0


            trades.append(

                {
                    "date": row["timestamp"],
                    "action": "BUY",
                    "price": row["close"]
                }

            )



        # SELL

        elif action == "SELL" and shares > 0:


            cash = (
                shares *
                row["close"]
            )


            shares = 0


            trades.append(

                {
                    "date": row["timestamp"],
                    "action": "SELL",
                    "price": row["close"]
                }

            )



    # Final value

    if shares > 0:

        final_value = (
            shares *
            df.iloc[-1]["close"]
        )

    else:

        final_value = cash



    return_percent = (

        final_value -
        INITIAL_CAPITAL

    ) / INITIAL_CAPITAL



    print(
        "\n========== RESULTS =========="
    )

    print(
        f"Starting Capital: ${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Value: ${final_value:,.2f}"
    )

    print(
        f"Return: {return_percent:.2%}"
    )

    print(
        f"Trades: {len(trades)}"
    )


    print(
        "============================="
    )



    return trades



if __name__ == "__main__":

    run_backtest()
