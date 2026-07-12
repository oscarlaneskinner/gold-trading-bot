"""
AI model training pipeline for Gold AI Trading Bot

Creates a machine learning model that predicts
whether GLD will rise over the next HOLD_DAYS.
"""


import os
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


from config import (
    SYMBOL,
    HOLD_DAYS,
    MODEL_PATH
)


from data import get_market_data

from features import (
    add_features,
    MODEL_FEATURES
)



def create_target(df):

    """
    Target:

    1 = price higher in HOLD_DAYS
    0 = price lower/equal
    """

    future_price = (
        df["close"]
        .shift(-HOLD_DAYS)
    )


    df["target"] = (
        future_price >
        df["close"]
    ).astype(int)


    return df



def train():

    print(
        "Downloading market data..."
    )


    df = get_market_data(
        SYMBOL,
        lookback_days=2500
    )


    print(
        f"Downloaded {len(df)} rows"
    )


    print(
        "Creating features..."
    )


    df = add_features(df)


    df = create_target(df)


    # Remove incomplete rows

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )


    X = df[MODEL_FEATURES]

    y = df["target"]



    # Time based split
    # (avoid future data leakage)

    split = int(
        len(df) * 0.80
    )


    X_train = X[:split]

    X_test = X[split:]


    y_train = y[:split]

    y_test = y[split:]



    print(
        "Training AI model..."
    )


    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=6,

        random_state=42,

        class_weight="balanced"

    )


    model.fit(
        X_train,
        y_train
    )



    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print(
        f"Accuracy: {accuracy:.2%}"
    )


    print(
        classification_report(
            y_test,
            predictions
        )
    )



    # Save model

    folder = os.path.dirname(
        MODEL_PATH
    )


    if folder:

        os.makedirs(
            folder,
            exist_ok=True
        )


    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )


    print(
        "Model saved:"
    )

    print(
        MODEL_PATH
    )



if __name__ == "__main__":

    train()
