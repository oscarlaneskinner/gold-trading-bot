from pathlib import Path
import json

REGISTRY=Path("reports/strategy_registry.json")

def initialize_registry():
    REGISTRY.parent.mkdir(parents=True,exist_ok=True)
    if not REGISTRY.exists():
        REGISTRY.write_text(json.dumps({
            "baseline":{
                "name":"LightGBM Production",
                "position_size":"10% / 15% research",
                "status":"ACTIVE"
            },
            "experiments":[]
        },indent=2))
    return json.loads(REGISTRY.read_text())

if __name__=="__main__":
    print(json.dumps(initialize_registry(),indent=2))
