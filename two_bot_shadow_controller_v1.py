"""Two-Bot Shadow Controller v1.

Combines:
- Market Regime Lab permissions,
- Portfolio Commander allocations,
- Hall of Fame strategy records,
- Championship Scanner candidates.

Creates a daily proposed trade plan for:
- GLD specialist,
- long opportunity bot.

Safety:
- no Alpaca client,
- no market request,
- no order submission,
- no production strategy changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def find_allocation(
    portfolio: dict[str, Any],
    role: str,
) -> dict[str, Any] | None:
    for item in portfolio.get("allocations", []):
        if str(item.get("role", "")).upper() == role:
            return item
    return None


def top_long_candidate(
    scanner: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = scanner.get("top_longs", [])
    if not candidates:
        return None
    return candidates[0]


def build_gld_proposal(
    allocation: dict[str, Any],
    regime: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    permissions = regime.get("permissions", {})
    if not permissions.get("allow_gld_bot", False):
        return None

    dollars = min(
        safe_float(allocation.get("allocation_dollars")),
        float(config["starting_capital"])
        * float(config["maximum_single_position_percent"])
        / 100.0,
    )

    if dollars < float(config["minimum_trade_dollars"]):
        return None

    return {
        "role": "GLD",
        "symbol": "GLD",
        "side": "BUY",
        "strategy_name": allocation.get("strategy_name", "GLD strategy"),
        "proposed_dollars": round(dollars, 2),
        "reason": "Qualified GLD strategy and current regime permits GLD exposure.",
        "status": "SHADOW_ONLY",
    }


def build_long_proposal(
    allocation: dict[str, Any],
    scanner: dict[str, Any],
    regime: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    permissions = regime.get("permissions", {})
    if not permissions.get("allow_long_bot", False):
        return None

    candidate = top_long_candidate(scanner)
    if not candidate:
        return None

    dollars = min(
        safe_float(allocation.get("allocation_dollars")),
        float(config["starting_capital"])
        * float(config["maximum_single_position_percent"])
        / 100.0,
    )

    if permissions.get("reduce_position_size", False):
        dollars *= 0.5

    if dollars < float(config["minimum_trade_dollars"]):
        return None

    return {
        "role": "LONG",
        "symbol": str(candidate.get("symbol", "")),
        "side": "BUY",
        "strategy_name": allocation.get("strategy_name", "Long strategy"),
        "scanner_score": safe_float(candidate.get("score")),
        "proposed_dollars": round(dollars, 2),
        "suggested_stop": candidate.get("suggested_stop"),
        "suggested_target": candidate.get("suggested_target"),
        "reason": "Qualified long strategy, current regime permits long exposure, and scanner ranked this symbol first.",
        "status": "SHADOW_ONLY",
    }


def remove_duplicates(
    proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []

    for proposal in proposals:
        symbol = str(proposal.get("symbol", "")).upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        output.append(proposal)

    return output


def enforce_exposure_limit(
    proposals: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    maximum_dollars = (
        float(config["starting_capital"])
        * float(config["maximum_total_exposure_percent"])
        / 100.0
    )

    total = sum(safe_float(item.get("proposed_dollars")) for item in proposals)

    if total <= maximum_dollars or total <= 0:
        return proposals

    scale = maximum_dollars / total

    adjusted = []
    for item in proposals:
        copy = dict(item)
        copy["proposed_dollars"] = round(
            safe_float(item.get("proposed_dollars")) * scale,
            2,
        )
        adjusted.append(copy)

    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/two_bot_shadow_controller_v1.json",
    )
    args = parser.parse_args()

    config = load_json(Path(args.config))
    paths = config["report_paths"]

    regime = load_json(Path(paths["market_regime"]))
    portfolio = load_json(Path(paths["portfolio"]))
    hall = load_json(Path(paths["hall_of_fame"]))
    scanner = load_json(Path(paths["scanner"]))

    missing_inputs = [
        name
        for name, path in paths.items()
        if not Path(path).exists()
    ]

    proposals: list[dict[str, Any]] = []

    gld_allocation = find_allocation(portfolio, "GLD")
    long_allocation = find_allocation(portfolio, "LONG")

    if gld_allocation is not None:
        proposal = build_gld_proposal(
            gld_allocation,
            regime,
            config,
        )
        if proposal is not None:
            proposals.append(proposal)

    if long_allocation is not None:
        proposal = build_long_proposal(
            long_allocation,
            scanner,
            regime,
            config,
        )
        if proposal is not None:
            proposals.append(proposal)

    proposals = remove_duplicates(proposals)
    proposals = enforce_exposure_limit(proposals, config)
    proposals = proposals[: int(config["maximum_daily_proposals"])]

    proposed_total = round(
        sum(safe_float(item.get("proposed_dollars")) for item in proposals),
        2,
    )

    output = {
        "controller": "Two-Bot Shadow Controller v1",
        "market_regime": regime.get("regime", "UNKNOWN"),
        "permissions": regime.get("permissions", {}),
        "missing_inputs": missing_inputs,
        "proposal_count": len(proposals),
        "proposed_total_dollars": proposed_total,
        "cash_remaining_dollars": round(
            float(config["starting_capital"]) - proposed_total,
            2,
        ),
        "proposals": proposals,
        "hall_of_fame_strategy_count": hall.get("strategy_count", 0),
        "shadow_mode": True,
        "production_strategy_changed": False,
        "market_request_made": False,
        "order_submitted": False,
    }

    report_dir = Path("reports/shadow")
    report_dir.mkdir(parents=True, exist_ok=True)

    (
        report_dir
        / "two_bot_shadow_controller_v1.json"
    ).write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(proposals).to_csv(
        report_dir
        / "two_bot_shadow_proposals.csv",
        index=False,
    )

    summary_lines = [
        "TWO-BOT SHADOW CONTROLLER V1",
        "=" * 30,
        f"Regime: {output['market_regime']}",
        f"Proposal count: {output['proposal_count']}",
        f"Proposed total: ${output['proposed_total_dollars']:.2f}",
        f"Cash remaining: ${output['cash_remaining_dollars']:.2f}",
        "",
    ]

    for index, proposal in enumerate(proposals, start=1):
        summary_lines.append(
            f"{index}. {proposal['role']} | {proposal['side']} "
            f"{proposal['symbol']} | ${proposal['proposed_dollars']:.2f}"
        )

    summary_lines.extend(
        [
            "",
            "Shadow mode only.",
            "No market request was made.",
            "No order was submitted.",
        ]
    )

    (
        report_dir
        / "two_bot_shadow_controller_v1_summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("Two-Bot Shadow Controller v1")
    print(json.dumps(output, indent=2))
    print("Shadow mode only.")
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    main()
