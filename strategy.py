"""
Trading strategy logic for Gold AI Trading Bot

Combines:
- AI prediction
- Confidence threshold
- Trend filters
- RSI filters
"""

from config import (
    CONFIDENCE_THRESHOLD,
    USE_TREND_FILTER,
    USE_RSI_FILTER
)



def evaluate_trade(
    prediction,
    confidence,
    latest_row
):
    """
    Returns:
        BUY
        SELL
        HOLD
    """


    # ==========================
    # AI CONFIDENCE CHECK
    # ==========================

    if prediction == 1:

        if confidence < CONFIDENCE_THRESHOLD:

            return {
                "action": "HOLD",
                "reason":
                f"AI confidence too low: {confidence:.1%}"
            }


    # ==========================
    # TREND FILTER
    # ==========================

    if USE_TREND_FILTER:

        bullish_trend = (

            latest_row["close"]
            >
            latest_row["ema_200"]

            and

            latest_row["ema_9"]
            >
            latest_row["ema_21"]

            and

            latest_row["ema_21"]
            >
            latest_row["ema_50"]

        )


        if not bullish_trend:

            return {
                "action": "HOLD",
                "reason":
                "Trend filter failed"
            }



    # ==========================
    # RSI FILTER
    # ==========================

    if USE_RSI_FILTER:

        rsi_value = latest_row["rsi_14"]


        if rsi_value >= 70:

            return {
                "action": "HOLD",
                "reason":
                f"RSI overbought: {rsi_value:.1f}"
            }



    # ==========================
    # BUY SIGNAL
    # ==========================

    if prediction == 1:

        return {
            "action": "BUY",
            "reason":
            f"AI bullish confidence {confidence:.1%}"
        }



    # ==========================
    # SELL SIGNAL
    # ==========================

    return {
        "action": "SELL",
        "reason":
        "AI bearish prediction"
    }
