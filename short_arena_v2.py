import json
print("Short Arena v2 (starter)")
print(json.dumps({
  "status":"starter",
  "research_only":True,
  "market_request_made":False,
  "order_submitted":False
},indent=2))
