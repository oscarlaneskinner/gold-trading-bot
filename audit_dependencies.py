"""Verify that required dependency files exist and contain core packages."""

from __future__ import annotations

import json
from pathlib import Path


def normalized_lines(path: Path) -> set[str]:
    return {
        line.strip().lower()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.strip().startswith("#")
    }


def contains_package(
    lines: set[str],
    package_name: str,
) -> bool:
    normalized = package_name.lower()

    return any(
        line == normalized
        or line.startswith(normalized + "==")
        or line.startswith(normalized + ">=")
        or line.startswith(normalized + "<=")
        or line.startswith(normalized + "~=")
        for line in lines
    )


def run() -> None:
    requirements = Path("requirements.txt")
    model_requirements = Path(
        "requirements_models.txt"
    )

    issues = []

    if not requirements.exists():
        issues.append(
            "requirements.txt is missing."
        )

    if not model_requirements.exists():
        issues.append(
            "requirements_models.txt is missing."
        )

    all_lines: set[str] = set()

    if requirements.exists():
        all_lines |= normalized_lines(
            requirements
        )

    if model_requirements.exists():
        all_lines |= normalized_lines(
            model_requirements
        )

    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "alpaca-py",
    ]

    package_status = {
        package: contains_package(
            all_lines,
            package,
        )
        for package in required_packages
    }

    for package, present in (
        package_status.items()
    ):
        if not present:
            issues.append(
                f"{package} is not listed in the "
                "dependency files."
            )

    output = {
        "status": (
            "passed"
            if not issues
            else "failed"
        ),
        "package_status": package_status,
        "issues": issues,
        "production_strategy_changed": False,
        "order_submitted": False,
    }

    print("GLD dependency audit")
    print(json.dumps(output, indent=2))
    print("No order was submitted.")

    if issues:
        raise SystemExit(
            "Dependency audit failed."
        )


if __name__ == "__main__":
    run()
