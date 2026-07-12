"""
Risk management module for Gold AI Trading Bot

Controls:
- Position sizing
- Stop loss
- Take profit
- Risk limits
"""

from config import (
    RISK_PER_TRADE,
    MAX_POSITION_PERCENT,
    ATR_STOP_MULTIPLIER,
    ATR_TARGET_MULTIPLIER
)


def calculate_position_size(
    account_equity,
    price,
    confidence
):
    """
    Calculates trade size based on:
    - account value
    - risk percentage
    - AI confidence
    """


    risk_amount = (
        account_equity *
        RISK_PER_TRADE
    )


    # Increase allocation slightly
    # when AI confidence is stronger

    confidence_multiplier = min(
        confidence,
        1.0
    )


    position_value = (
        risk_amount *
        confidence_multiplier *
        10
    )


    # Maximum account exposure

    max_position = (
        account_equity *
        MAX_POSITION_PERCENT
    )


    position_value = min(
        position_value,
        max_position
    )


    shares = (
        position_value /
        price
    )


    return round(
        shares,
        4
    )



def calculate_stop_loss(
    entry_price,
    atr
):
    """
    ATR based stop loss
    """

    return round(
        entry_price -
        (
            atr *
            ATR_STOP_MULTIPLIER
        ),
        2
    )



def calculate_take_profit(
    entry_price,
    atr
):
    """
    ATR based profit target
    """

    return round(
        entry_price +
        (
            atr *
            ATR_TARGET_MULTIPLIER
        ),
        2
    )



def check_risk_limits(
    account_equity,
    position_value
):
    """
    Emergency check before order.
    """

    max_allowed = (
        account_equity *
        MAX_POSITION_PERCENT
    )


    if position_value > max_allowed:

        return False


    return True
