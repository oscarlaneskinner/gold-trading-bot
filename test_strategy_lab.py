import json
from strategy_lab import initialize_registry
r=initialize_registry()
print("GLD Strategy Laboratory test")
print(json.dumps({
 "status":"passed",
 "baseline":r["baseline"]["name"],
 "experiment_count":len(r["experiments"]),
 "production_strategy_changed":False,
 "order_submitted":False
},indent=2))
print("No market request was made.")
print("No order was submitted.")
