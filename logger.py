"""
Trade logging module for Gold AI Trading Bot

Stores all decisions and trades in CSV format.
"""

import os
import csv
from datetime import datetime

from config import LOG_FILE



def initialize_log():

    directory = os.path.dirname(
        LOG_FILE
    )


    if directory and not os.path.exists(directory):

        os.makedirs(directory)



    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "timestamp",
                    "symbol",
                    "prediction",
                    "confidence",
                    "action",
                    "reason",
                    "price",
                    "order_id"
                ]
            )



def log_trade(
    symbol,
    prediction,
    confidence,
    action,
    reason,
    price=None,
    order_id=None
):

    initialize_log()


    with open(
        LOG_FILE,
        "a",
        newline=""
    ) as file:


        writer = csv.writer(file)


        writer.writerow(
            [
                datetime.now(),

                symbol,

                prediction,

                round(
                    confidence,
                    4
                ),

                action,

                reason,

                price,

                order_id
            ]
        )
