from __future__ import annotations
import argparse, csv, json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/tradingview_signals.csv")
FIELDS = ["received_at_utc","source","strategy","event","ticker","exchange","timeframe","price","order_id","position_size","timestamp"]

def append_signal(payload):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({"received_at_utc": datetime.now(timezone.utc).isoformat(), **{field: payload.get(field) for field in FIELDS[1:]}})

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-json", required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload_json)
    if payload.get("source") != "tradingview":
        raise ValueError("Signal source must be tradingview.")
    append_signal(payload)
    print(json.dumps({"status":"saved","log_path":str(LOG_PATH),"payload":payload,"market_request_made":False,"order_submitted":False}, indent=2))

if __name__ == "__main__":
    run()
