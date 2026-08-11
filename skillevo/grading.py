"""Deterministic checks over a triage outcome.

These are the single source of truth for two different consumers:

* the simulated customer in :mod:`skillevo.scenarios`, which pushes back when a
  check fails (producing the extra conversation turns the loop later mines), and
* the LangSmith evaluators in :mod:`skillevo.evaluate`.

Sharing them keeps in-session friction and offline scoring measuring the same thing.
Everything here is plain string/sequence logic -- no model calls -- so the headline
metrics of the test case are reproducible rather than judge-dependent.
"""

from __future__ import annotations

import re

from skillevo.domain import MINIMAL_TOOL_CALLS

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}

_POLICY_CODE = re.compile(r"\bP-\d{3}\b")


def cites_transaction_id(message: str, transaction_id: str) -> bool:
    """True when the reply identifies the disputed posting by reference."""
    return transaction_id.upper() in (message or "").upper()


def cites_currency(message: str, currency: str) -> bool:
    """True when the reply makes the currency of the amount unambiguous."""
    message = message or ""
    if currency.upper() in message.upper():
        return True
    symbol = CURRENCY_SYMBOLS.get(currency.upper())
    return bool(symbol and symbol in message)


def cites_policy_code(message: str, policy_code: str) -> bool:
    """True when the reply cites the governing policy code, and the right one."""
    found = set(_POLICY_CODE.findall(message or ""))
    return policy_code in found


def has_fraud_check(tool_sequence: list[str]) -> bool:
    return "check_fraud_flags" in tool_sequence


def has_duplicate_check(tool_sequence: list[str]) -> bool:
    return "list_account_transactions" in tool_sequence


def fraud_check_precedes_remediation(tool_sequence: list[str], resolution: str) -> bool:
    """The compliance gate.

    Any resolution other than ``needs_more_info`` is a remediation recommendation,
    and must not be made without first checking the account for fraud holds. The
    memo is produced after the tool calls, so presence implies precedence.
    """
    if resolution == "needs_more_info":
        return True
    return has_fraud_check(tool_sequence)


def tool_efficiency(tool_sequence: list[str]) -> float:
    """1.0 when the agent worked at or below the minimal call count."""
    if not tool_sequence:
        return 0.0
    return min(1.0, MINIMAL_TOOL_CALLS / len(tool_sequence))


def redundant_tool_calls(tool_sequence: list[str]) -> list[str]:
    """Tool names called more than once with no new information to gain."""
    seen: set[str] = set()
    repeats: list[str] = []
    for name in tool_sequence:
        if name in seen:
            repeats.append(name)
        seen.add(name)
    return repeats


def score_memo(
    resolution: str,
    customer_message: str,
    tool_sequence: list[str],
    reference: dict,
    turns_used: int = 1,
    transcript: str | None = None,
) -> dict[str, float]:
    """All deterministic metrics for one completed session.

    Citation checks run against ``transcript`` (every agent reply in the session),
    so the agent is credited for information the customer eventually received.
    Whether it arrived without the customer having to re-ask is measured separately
    by ``first_contact_resolution`` -- keeping completeness and efficiency from
    being conflated into one number.
    """
    cited_in = transcript if transcript is not None else customer_message
    scores = {
        "resolution_correct": float(resolution == reference["resolution"]),
        "compliance_fraud_check": float(
            fraud_check_precedes_remediation(tool_sequence, resolution)
        ),
        "duplicate_check": float(has_duplicate_check(tool_sequence)),
        "cites_transaction_id": float(
            cites_transaction_id(cited_in, reference["transaction_id"])
        ),
        "cites_currency": float(cites_currency(cited_in, reference["currency"])),
        "cites_policy_code": float(cites_policy_code(cited_in, reference["policy_code"])),
        "tool_efficiency": tool_efficiency(tool_sequence),
        # 1.0 when resolved on the first reply; decays as the customer has to re-ask.
        "first_contact_resolution": float(turns_used <= 1),
    }
    graded = [
        scores["resolution_correct"],
        scores["compliance_fraud_check"],
        scores["cites_transaction_id"],
        scores["cites_currency"],
        scores["cites_policy_code"],
        scores["first_contact_resolution"],
    ]
    scores["composite"] = sum(graded) / len(graded)
    return scores
