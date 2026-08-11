"""Synthetic payments back office: fixtures plus the tools the agent can call.

All data here is invented test fixture data. Nothing touches a real system, a
network, or a database. The point is that the *correct procedure* for handling a
dispute is discoverable only by using these tools and observing outcomes -- it is
deliberately absent from the starting SKILL.md.
"""

from __future__ import annotations

from typing import Literal

from langchain.tools import tool

Resolution = Literal[
    "reverse_duplicate",
    "retry_payment",
    "provisional_credit",
    "route_to_fraud_team",
    "explain_authorization_hold",
    "explain_nsf",
    "needs_more_info",
]

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

TRANSACTIONS: dict[str, dict] = {
    # -- training accounts ------------------------------------------------- #
    "TXN-D0": {"account_id": "ACC-1001", "merchant": "Verdant Coffee Roasters", "amount": 18.40, "currency": "USD", "status": "posted", "posted_at": "2026-07-14T09:02:00Z", "mcc": "5814"},
    "TXN-D1": {"account_id": "ACC-1001", "merchant": "Verdant Coffee Roasters", "amount": 18.40, "currency": "USD", "status": "posted", "posted_at": "2026-07-14T10:47:00Z", "mcc": "5814"},
    "TXN-F1": {"account_id": "ACC-1002", "merchant": "Wexford Property Mgmt", "amount": 1240.00, "currency": "USD", "status": "declined", "posted_at": "2026-07-15T14:20:00Z", "decline_code": "do_not_honor", "mcc": "6513"},
    "TXN-U1": {"account_id": "ACC-1003", "merchant": "GadgetHaus Online", "amount": 899.00, "currency": "USD", "status": "posted", "posted_at": "2026-07-16T03:11:00Z", "mcc": "5732"},
    "TXN-D2": {"account_id": "ACC-1004", "merchant": "Shell 4471", "amount": 61.75, "currency": "USD", "status": "posted", "posted_at": "2026-07-17T18:05:00Z", "mcc": "5541"},
    "TXN-D3": {"account_id": "ACC-1004", "merchant": "Shell 4471", "amount": 61.75, "currency": "USD", "status": "posted", "posted_at": "2026-07-17T18:09:00Z", "mcc": "5541"},
    "TXN-N1": {"account_id": "ACC-1005", "merchant": "Cascade Fitness", "amount": 75.00, "currency": "USD", "status": "declined", "posted_at": "2026-07-18T07:30:00Z", "decline_code": "insufficient_funds", "mcc": "7997"},
    "TXN-A1": {"account_id": "ACC-1006", "merchant": "Hotel Meridiana", "amount": 250.00, "currency": "EUR", "status": "pending_authorization", "posted_at": "2026-07-19T21:40:00Z", "mcc": "7011"},
    # -- holdout accounts -------------------------------------------------- #
    "TXN-H1A": {"account_id": "ACC-2001", "merchant": "Sunrise Grocers", "amount": 132.09, "currency": "USD", "status": "posted", "posted_at": "2026-07-21T11:15:00Z", "mcc": "5411"},
    "TXN-H1B": {"account_id": "ACC-2001", "merchant": "Sunrise Grocers", "amount": 132.09, "currency": "USD", "status": "posted", "posted_at": "2026-07-21T11:58:00Z", "mcc": "5411"},
    "TXN-H2": {"account_id": "ACC-2002", "merchant": "Northwind Digital", "amount": 419.99, "currency": "USD", "status": "posted", "posted_at": "2026-07-22T02:04:00Z", "mcc": "5815"},
    "TXN-H3A": {"account_id": "ACC-2003", "merchant": "Peak Outfitters", "amount": 288.50, "currency": "USD", "status": "posted", "posted_at": "2026-07-23T16:22:00Z", "mcc": "5941"},
    "TXN-H3B": {"account_id": "ACC-2003", "merchant": "Peak Outfitters", "amount": 288.50, "currency": "USD", "status": "posted", "posted_at": "2026-07-23T16:31:00Z", "mcc": "5941"},
    "TXN-H4": {"account_id": "ACC-2004", "merchant": "Camden Utilities", "amount": 96.40, "currency": "GBP", "status": "declined", "posted_at": "2026-07-24T08:45:00Z", "decline_code": "do_not_honor", "mcc": "4900"},
    "TXN-H5": {"account_id": "ACC-2005", "merchant": "Harbor Car Rental", "amount": 400.00, "currency": "USD", "status": "pending_authorization", "posted_at": "2026-07-25T13:02:00Z", "mcc": "7512"},
}

ACCOUNTS: dict[str, dict] = {
    "ACC-1001": {"holder": "R. Alvarez", "tier": "standard", "opened": "2019-03-02", "currencies": ["USD"]},
    "ACC-1002": {"holder": "T. Nakamura", "tier": "premier", "opened": "2015-11-19", "currencies": ["USD"]},
    "ACC-1003": {"holder": "P. Okonkwo", "tier": "standard", "opened": "2021-06-30", "currencies": ["USD"]},
    "ACC-1004": {"holder": "M. Lindqvist", "tier": "standard", "opened": "2018-01-08", "currencies": ["USD"]},
    "ACC-1005": {"holder": "J. Whitfield", "tier": "basic", "opened": "2023-09-12", "currencies": ["USD"]},
    "ACC-1006": {"holder": "S. Bertoli", "tier": "premier", "opened": "2016-04-25", "currencies": ["USD", "EUR"]},
    "ACC-2001": {"holder": "D. Ferreira", "tier": "standard", "opened": "2020-02-14", "currencies": ["USD"]},
    "ACC-2002": {"holder": "K. Sandoval", "tier": "premier", "opened": "2017-08-03", "currencies": ["USD"]},
    "ACC-2003": {"holder": "A. Broussard", "tier": "standard", "opened": "2022-05-27", "currencies": ["USD"]},
    "ACC-2004": {"holder": "H. Ellery", "tier": "standard", "opened": "2019-10-11", "currencies": ["GBP", "USD"]},
    "ACC-2005": {"holder": "N. Petrov", "tier": "basic", "opened": "2024-01-16", "currencies": ["USD"]},
}

# Accounts with an open fraud hold cannot be remediated through self-service.
FRAUD_STATE: dict[str, dict] = {
    "ACC-1003": {"open_holds": [{"hold_id": "FH-7781", "opened": "2026-07-16T05:00:00Z", "reason": "card_not_present_velocity"}], "recent_reports": 1},
    "ACC-2003": {"open_holds": [{"hold_id": "FH-8120", "opened": "2026-07-23T19:00:00Z", "reason": "device_mismatch"}], "recent_reports": 2},
}

POLICY_TOPICS = [
    "duplicate_charge",
    "failed_payment",
    "unauthorized_transaction",
    "authorization_hold",
    "insufficient_funds",
]

POLICIES: dict[str, dict] = {
    "duplicate_charge": {
        "code": "P-101",
        "title": "Duplicate card presentment",
        "text": (
            "Two postings from the same merchant for the same amount within 72 hours are "
            "treated as a duplicate presentment, not a failed payment. Reverse the later "
            "posting within 3 business days. No provisional credit is required. "
            "If the account carries an open fraud hold, self-service reversal is frozen and "
            "the case must be routed to the fraud team instead (see P-311)."
        ),
    },
    "failed_payment": {
        "code": "P-204",
        "title": "Declined payment",
        "text": (
            "A declined authorization never captured funds, so there is nothing to refund. "
            "Advise the customer to retry the payment. Escalate only if an authorization "
            "hold is still visible after 5 business days."
        ),
    },
    "unauthorized_transaction": {
        "code": "P-311",
        "title": "Unauthorized transaction",
        "text": (
            "Issue provisional credit within 10 business days; raising the credit also opens "
            "the fraud review, so no prior review is needed. Exception: if the account "
            "already carries an OPEN fraud hold it is under active investigation -- do not "
            "issue credit and do not reverse any posting on that account, route the case to "
            "the fraud team instead. An open fraud hold takes precedence over every other "
            "remediation, including duplicate reversal."
        ),
    },
    "authorization_hold": {
        "code": "P-402",
        "title": "Pending authorization hold",
        "text": (
            "A pending authorization is not a charge. Merchants in lodging, fuel and vehicle "
            "rental routinely hold an estimated amount that releases within 7 days. No "
            "refund or reversal applies."
        ),
    },
    "insufficient_funds": {
        "code": "P-508",
        "title": "Insufficient funds decline",
        "text": (
            "No funds moved, so no refund applies. Explain the NSF decline. Basic-tier "
            "accounts may be assessed a returned-item fee; premier-tier accounts are exempt."
        ),
    },
}

POLICY_CODES = {topic: policy["code"] for topic, policy in POLICIES.items()}


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@tool
def lookup_transaction(transaction_id: str) -> dict:
    """Look up a single transaction by its identifier (e.g. "TXN-D1")."""
    txn = TRANSACTIONS.get(transaction_id.strip().upper())
    if txn is None:
        return {"error": f"no transaction found with id {transaction_id!r}"}
    return {"transaction_id": transaction_id.strip().upper(), **txn}


@tool
def list_account_transactions(account_id: str, days: int = 7) -> dict:
    """List recent transactions on an account, newest first.

    Use this to see the surrounding activity on an account.
    """
    account_id = account_id.strip().upper()
    if account_id not in ACCOUNTS:
        return {"error": f"no account found with id {account_id!r}"}
    rows = [
        {"transaction_id": txn_id, **txn}
        for txn_id, txn in TRANSACTIONS.items()
        if txn["account_id"] == account_id
    ]
    rows.sort(key=lambda r: r["posted_at"], reverse=True)
    return {"account_id": account_id, "window_days": days, "transactions": rows}


@tool
def check_fraud_flags(account_id: str) -> dict:
    """Check whether an account has open fraud holds or recent fraud reports."""
    account_id = account_id.strip().upper()
    if account_id not in ACCOUNTS:
        return {"error": f"no account found with id {account_id!r}"}
    state = FRAUD_STATE.get(account_id, {"open_holds": [], "recent_reports": 0})
    return {
        "account_id": account_id,
        "open_holds": state["open_holds"],
        "recent_reports": state["recent_reports"],
        "has_open_hold": bool(state["open_holds"]),
    }


@tool
def get_policy(topic: str) -> dict:
    """Retrieve the bank policy for a dispute topic.

    Valid topics: duplicate_charge, failed_payment, unauthorized_transaction,
    authorization_hold, insufficient_funds.
    """
    policy = POLICIES.get(topic.strip().lower())
    if policy is None:
        return {"error": f"unknown topic {topic!r}", "valid_topics": POLICY_TOPICS}
    return {"topic": topic.strip().lower(), **policy}


@tool
def lookup_account(account_id: str) -> dict:
    """Look up account holder details and product tier."""
    account_id = account_id.strip().upper()
    account = ACCOUNTS.get(account_id)
    if account is None:
        return {"error": f"no account found with id {account_id!r}"}
    return {"account_id": account_id, **account}


TOOLS = [
    lookup_transaction,
    list_account_transactions,
    check_fraud_flags,
    get_policy,
    lookup_account,
]

TOOL_NAMES = [t.name for t in TOOLS]

# The shortest tool sequence that can correctly resolve any scenario in this set:
# identify the transaction, inspect surrounding activity, clear fraud, cite policy.
MINIMAL_TOOL_CALLS = 4
