from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

app = Flask(__name__)

# --- Config loaded from .env ---
ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
PAPER_TRADING = True   # keep True until you're ready to trade real money

SYMBOL = "GLD"
DOLLAR_AMOUNT = 100    # $ amount per trade

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER_TRADING)


def get_position_qty(symbol):
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except Exception:
        return 0.0  # no open position


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json

    # --- Security check ---
    if data.get("secret") != WEBHOOK_SECRET:
        return jsonify({"error": "unauthorized"}), 401

    action = data.get("action")

    try:
        if action == "buy":
            order_request = MarketOrderRequest(
                symbol=SYMBOL,
                notional=DOLLAR_AMOUNT,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = trading_client.submit_order(order_request)
            return jsonify({"status": "buy submitted", "order_id": str(order.id)}), 200

        elif action == "sell":
            qty = get_position_qty(SYMBOL)
            if qty <= 0:
                return jsonify({"status": "no position to sell"}), 200

            order_request = MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = trading_client.submit_order(order_request)
            return jsonify({"status": "sell submitted", "order_id": str(order.id)}), 200

        else:
            return jsonify({"error": "invalid action"}), 400

    except Exception as e:
        print(f"Order error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)