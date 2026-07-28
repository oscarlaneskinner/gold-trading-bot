"""Email the daily research competition report through Yahoo SMTP."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


TEXT_REPORT_PATH = Path(
    "reports/research_command_center_daily.txt"
)
HTML_REPORT_PATH = Path(
    "reports/research_command_center_daily.html"
)


def required_environment(
    name: str,
) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Required environment variable "
            f"{name} is missing."
        )

    return value


def run() -> None:
    username = required_environment(
        "YAHOO_EMAIL_USERNAME"
    )

    app_password = required_environment(
        "YAHOO_EMAIL_APP_PASSWORD"
    )

    recipient = os.getenv(
        "COMPETITION_EMAIL_TO",
        "oscarlaneskinner@yahoo.com",
    ).strip()

    if not TEXT_REPORT_PATH.exists():
        raise RuntimeError(
            "Daily text report does not exist. "
            "Run research_command_center.py first."
        )

    text_body = TEXT_REPORT_PATH.read_text(
        encoding="utf-8"
    )

    html_body = (
        HTML_REPORT_PATH.read_text(
            encoding="utf-8"
        )
        if HTML_REPORT_PATH.exists()
        else None
    )

    message = EmailMessage()
    message["Subject"] = (
        "Daily Trading Bot Competition Report"
    )
    message["From"] = username
    message["To"] = recipient
    message.set_content(text_body)

    if html_body:
        message.add_alternative(
            html_body,
            subtype="html",
        )

    with smtplib.SMTP_SSL(
        "smtp.mail.yahoo.com",
        465,
        timeout=30,
    ) as server:
        server.login(
            username,
            app_password,
        )
        server.send_message(message)

    print(
        "Daily competition email sent to "
        f"{recipient}."
    )
    print("No market request was made.")
    print("No order was submitted.")


if __name__ == "__main__":
    run()
