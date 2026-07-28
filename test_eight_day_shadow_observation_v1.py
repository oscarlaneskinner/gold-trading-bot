from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


with tempfile.TemporaryDirectory() as temporary:
    temp = Path(temporary)

    fixture_scripts = {
        "market_regime_lab_v1.py": (
            "import json\nfrom pathlib import Path\n"
            "Path('reports/market_regime').mkdir(parents=True,exist_ok=True)\n"
            "Path('reports/market_regime/market_regime_lab_v1.json').write_text("
            "json.dumps({'regime':'BULL','permissions':{'allow_gld_bot':True,'allow_long_bot':True,'allow_short_bot':False,'reduce_position_size':False}}))\n"
        ),
        "strategy_hall_of_fame_v1.py": (
            "import json\nfrom pathlib import Path\n"
            "Path('reports/hall_of_fame').mkdir(parents=True,exist_ok=True)\n"
            "Path('reports/hall_of_fame/strategy_hall_of_fame.json').write_text(json.dumps({'strategy_count':2}))\n"
        ),
        "portfolio_commander_v1.py": (
            "import json\nfrom pathlib import Path\n"
            "Path('reports/portfolio').mkdir(parents=True,exist_ok=True)\n"
            "Path('reports/portfolio/portfolio_commander_v1.json').write_text(json.dumps({'allocations':[{'role':'GLD','strategy_name':'ARENA-0081','allocation_dollars':700},{'role':'LONG','strategy_name':'MM-0013','allocation_dollars':600}]}))\n"
        ),
        "championship_market_scanner_v1.py": (
            "import json\nfrom pathlib import Path\n"
            "Path('reports/scanner').mkdir(parents=True,exist_ok=True)\n"
            "Path('reports/scanner/championship_scanner_v1.json').write_text(json.dumps({'top_longs':[{'symbol':'NVDA','score':88.5,'suggested_stop':100,'suggested_target':120}]}))\n"
        ),
        "two_bot_shadow_controller_v1.py": (
            "import json\nfrom pathlib import Path\n"
            "Path('reports/shadow').mkdir(parents=True,exist_ok=True)\n"
            "Path('reports/shadow/two_bot_shadow_controller_v1.json').write_text(json.dumps({'proposal_count':2,'proposed_total_dollars':1300,'proposals':[{'role':'GLD','side':'BUY','symbol':'GLD','proposed_dollars':700,'strategy_name':'ARENA-0081'},{'role':'LONG','side':'BUY','symbol':'NVDA','proposed_dollars':600,'strategy_name':'MM-0013'}],'missing_inputs':[]}))\n"
        ),
    }

    for name, content in fixture_scripts.items():
        (temp / name).write_text(content, encoding="utf-8")

    runner_source = Path("eight_day_shadow_observation_v1.py").read_text(encoding="utf-8")
    (temp / "eight_day_shadow_observation_v1.py").write_text(runner_source, encoding="utf-8")

    config = {
        "observation_days": 8,
        "timezone": "America/New_York",
        "starting_capital": 2000,
        "refresh_market_data": False,
        "archive_directory": "reports/shadow_observation/archive",
        "state_file": "reports/shadow_observation/observation_state.json",
        "daily_summary_file": "reports/shadow_observation/latest_summary.json",
        "daily_text_file": "reports/shadow_observation/latest_summary.txt",
        "email_preview_file": "reports/shadow_observation/latest_email_preview.txt",
        "required_commands": [
            [sys.executable, "market_regime_lab_v1.py"],
            [sys.executable, "strategy_hall_of_fame_v1.py"],
            [sys.executable, "portfolio_commander_v1.py"],
            [sys.executable, "championship_market_scanner_v1.py"],
            [sys.executable, "two_bot_shadow_controller_v1.py"],
        ],
        "market_data_command": [sys.executable, "-c", "print('skip')"],
        "report_paths": {
            "market_regime": "reports/market_regime/market_regime_lab_v1.json",
            "hall_of_fame": "reports/hall_of_fame/strategy_hall_of_fame.json",
            "portfolio": "reports/portfolio/portfolio_commander_v1.json",
            "scanner": "reports/scanner/championship_scanner_v1.json",
            "shadow_controller": "reports/shadow/two_bot_shadow_controller_v1.json",
        },
        "email": {
            "enabled": False,
            "recipient_env": "X",
            "sender_env": "X",
            "smtp_host_env": "X",
            "smtp_port_env": "X",
            "smtp_username_env": "X",
            "smtp_password_env": "X",
            "use_tls": True,
            "subject_prefix": "Test",
        },
        "safety": {
            "forbidden_modules": [
                "alpaca.trading",
                "TradingClient",
                "submit_order",
                "paper=False",
            ],
            "require_shadow_mode": True,
            "allow_order_submission": False,
        },
    }

    (temp / "config").mkdir()
    (temp / "config/eight_day_shadow_observation_v1.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "eight_day_shadow_observation_v1.py",
            "--skip-data-refresh",
        ],
        cwd=temp,
        capture_output=True,
        text=True,
    )

    summary_path = temp / "reports/shadow_observation/latest_summary.json"
    state_path = temp / "reports/shadow_observation/observation_state.json"

    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    output = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "proposal_count": summary.get("shadow_proposal_count", 0),
        "completed_days": state.get("completed_days", 0),
        "summary_created": summary_path.exists(),
        "state_created": state_path.exists(),
        "archive_created": (temp / "reports/shadow_observation/archive").exists(),
        "email_preview_created": (
            temp / "reports/shadow_observation/latest_email_preview.txt"
        ).exists(),
        "market_request_made": False,
        "order_submitted": False,
    }

    print("Eight-Day Shadow Observation Suite test")
    print(json.dumps(output, indent=2))
    print("No market request was made.")
    print("No order was submitted.")

    if (
        output["status"] != "passed"
        or output["proposal_count"] != 2
        or output["completed_days"] != 1
    ):
        print(completed.stdout)
        print(completed.stderr)
        raise SystemExit("Observation suite test failed.")
