"""
Gold AI Trading Bot v2

Main execution engine.

Flow:

1. Download market data
2. Generate indicators
3. Load AI model
4. Predict direction
5. Apply strategy rules
6. Calculate risk
7. Place trade
8. Log everything
"""


import pickle
from datetime import datetime


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

from broker import (
    get_position,
    get_equity,
    buy_stock,
    sell_stock
)

from risk_manager import (
    calculate_position_size
)

from logger import log_trade



def load_model():

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(file)



def run_bot():


    print(
        f"""
===========================
AI Trading Bot Run
{datetime.now()}
===========================
"""
    )


    # -------------------------
    # Get market data
    # -------------------------

    df = get_market_data()

    print(
        f"Downloaded {len(df)} days"
    )


    # -------------------------
    # Build indicators
    # -------------------------

    df = add_features(df)


    latest = df.iloc[-1]


    X = (
        df[MODEL_FEATURES]
        .iloc[[-1]]
    )


    # -------------------------
    # AI prediction
    # -------------------------

    model = load_model()


    prediction = int(
        model.predict(X)[0]
    )


    probability = float(
        model.predict_proba(X)[0][1]
    )


    print(
        f"""
AI Prediction:
{'UP' if prediction == 1 else 'DOWN'}

Confidence:
{probability:.2%}
"""
    )



    # -------------------------
    # Strategy decision
    # -------------------------

    decision = evaluate_trade(

        prediction,

        probability,

        latest

    )


    action = decision["action"]

    reason = decision["reason"]


    print(
        action,
        reason
    )



    # -------------------------
    # Position check
    # -------------------------

    position = get_position()



    order_id = None



    # -------------------------
    # Execute BUY
    # -------------------------

    if action == "BUY" and position is None:


        equity = get_equity()


        shares = calculate_position_size(

            equity,

            latest["close"],

            probability

        )


        dollar_amount = (
            shares *
            latest["close"]
        )


        print(
            f"""
BUYING

Shares:
{shares}

Amount:
${dollar_amount:.2f}
"""
        )


        order = buy_stock(

            SYMBOL,

            dollar_amount

        )


        order_id = str(
            order.id
        )


    # -------------------------
    # Execute SELL
    # -------------------------

    elif action == "SELL" and position:


        order = sell_stock(
            SYMBOL
        )


        order_id = str(
            order.id
        )



    else:

        print(
            "No trade executed"
        )



    # -------------------------
    # Log result
    # -------------------------

    log_trade(

        SYMBOL,

        prediction,

        probability,

        action,

        reason,

        latest["close"],

        order_id

    )


    print(
        "Bot complete"
    )




if __name__ == "__main__":

    run_bot()
