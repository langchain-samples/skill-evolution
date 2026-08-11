"""The multi-session corpus, plus a deterministic simulated customer.

Design constraints that make this a *test case* rather than a demo:

1. **No single session teaches the rule.** Each training session exposes one or two
   failure modes. A rule only becomes learnable once the same failure recurs across
   sessions, which is what forces the optimizer to read cross-session history
   instead of over-fitting to one anecdote.

2. **Train and holdout are disjoint.** Holdout scenarios use different accounts,
   merchants, amounts and phrasings. ``H3`` is a composition never seen in
   training (a duplicate charge on an account that also has an open fraud hold), so
   the loop is measured on generalization, not memorization.

3. **The customer is code, not a model.** Follow-up turns fire from the same
   deterministic checks the evaluators use, so the friction signal is reproducible
   and free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skillevo import grading


@dataclass(frozen=True)
class Scenario:
    id: str
    split: str
    account_id: str
    transaction_id: str
    opening_message: str
    reference: dict
    # Failure modes this session is designed to expose, for documentation only --
    # never shown to the agent or the optimizer.
    exposes: tuple[str, ...] = field(default=())


def _ref(resolution: str, policy_code: str, transaction_id: str, currency: str) -> dict:
    return {
        "resolution": resolution,
        "policy_code": policy_code,
        "transaction_id": transaction_id,
        "currency": currency,
    }


TRAIN: list[Scenario] = [
    Scenario(
        id="S1",
        split="train",
        account_id="ACC-1001",
        transaction_id="TXN-D1",
        opening_message=(
            "Hi - I think I got billed twice at the coffee place this morning. "
            "My statement shows reference TXN-D1. Can you sort this out?"
        ),
        reference=_ref("reverse_duplicate", "P-101", "TXN-D1", "USD"),
        exposes=("misses_duplicate_check", "incomplete_memo"),
    ),
    Scenario(
        id="S2",
        split="train",
        account_id="ACC-1002",
        transaction_id="TXN-F1",
        opening_message=(
            "My rent payment did not go through and now I am late. Reference TXN-F1. "
            "I want the money back that you took."
        ),
        reference=_ref("retry_payment", "P-204", "TXN-F1", "USD"),
        exposes=("no_policy_citation", "refund_pressure"),
    ),
    Scenario(
        id="S3",
        split="train",
        account_id="ACC-1003",
        transaction_id="TXN-U1",
        opening_message=(
            "There is an $899 charge on my card from some electronics site I have never "
            "heard of - TXN-U1. I did not authorize this. Please refund it right now."
        ),
        reference=_ref("route_to_fraud_team", "P-311", "TXN-U1", "USD"),
        exposes=("skips_fraud_check", "refund_pressure"),
    ),
    Scenario(
        id="S4",
        split="train",
        account_id="ACC-1004",
        transaction_id="TXN-D2",
        opening_message=(
            "I filled up once but there are two identical gas charges on my account. "
            "The second one is TXN-D2. This is the second time this has happened to me."
        ),
        reference=_ref("reverse_duplicate", "P-101", "TXN-D2", "USD"),
        exposes=("misses_duplicate_check",),
    ),
    Scenario(
        id="S5",
        split="train",
        account_id="ACC-1005",
        transaction_id="TXN-N1",
        opening_message=(
            "My gym membership payment failed, reference TXN-N1, and I am being charged "
            "a fee on top of it. What is going on?"
        ),
        reference=_ref("explain_nsf", "P-508", "TXN-N1", "USD"),
        exposes=("no_policy_citation", "incomplete_memo"),
    ),
    Scenario(
        id="S6",
        split="train",
        account_id="ACC-1006",
        transaction_id="TXN-A1",
        opening_message=(
            "The hotel in Rome has taken 250 off my card but I already paid at checkout. "
            "Reference is TXN-A1. I want it reversed."
        ),
        reference=_ref("explain_authorization_hold", "P-402", "TXN-A1", "EUR"),
        exposes=("incomplete_memo", "currency_ambiguity"),
    ),
]

HOLDOUT: list[Scenario] = [
    Scenario(
        id="H1",
        split="holdout",
        account_id="ACC-2001",
        transaction_id="TXN-H1B",
        opening_message=(
            "The supermarket charged me twice for the same shop. The duplicate is "
            "TXN-H1B. Can you fix it?"
        ),
        reference=_ref("reverse_duplicate", "P-101", "TXN-H1B", "USD"),
    ),
    Scenario(
        id="H2",
        split="holdout",
        account_id="ACC-2002",
        transaction_id="TXN-H2",
        opening_message=(
            "Someone used my card at 2am for a subscription I never signed up for - "
            "TXN-H2. I want my money back today."
        ),
        reference=_ref("provisional_credit", "P-311", "TXN-H2", "USD"),
    ),
    Scenario(
        # Novel composition: looks exactly like the duplicate cases seen in training,
        # but the account is under an active fraud hold, which takes precedence.
        id="H3",
        split="holdout",
        account_id="ACC-2003",
        transaction_id="TXN-H3B",
        opening_message=(
            "Two charges from the same outdoor shop for the same amount, ten minutes "
            "apart. The second is TXN-H3B. Please reverse the extra one."
        ),
        reference=_ref("route_to_fraud_team", "P-311", "TXN-H3B", "USD"),
    ),
    Scenario(
        id="H4",
        split="holdout",
        account_id="ACC-2004",
        transaction_id="TXN-H4",
        opening_message=(
            "My utilities direct debit was declined - TXN-H4. Has the money left my "
            "account or not?"
        ),
        reference=_ref("retry_payment", "P-204", "TXN-H4", "GBP"),
    ),
    Scenario(
        id="H5",
        split="holdout",
        account_id="ACC-2005",
        transaction_id="TXN-H5",
        opening_message=(
            "The car rental company has 400 pending on my card and I returned the car "
            "days ago. Reference TXN-H5. Take it off please."
        ),
        reference=_ref("explain_authorization_hold", "P-402", "TXN-H5", "USD"),
    ),
]

ALL = TRAIN + HOLDOUT
BY_ID = {s.id: s for s in ALL}


# --------------------------------------------------------------------------- #
# Simulated customer
# --------------------------------------------------------------------------- #

# Ordered: at most one fires per turn, highest priority first. Each carries the
# friction signal name that the harvester will aggregate across sessions.
FOLLOWUP_RULES = [
    (
        "unresolved_request",
        lambda memo, ref: memo["resolution"] == "needs_more_info",
        "You should have everything you need on your side - it is my account. "
        "Can you please just look into it?",
    ),
    (
        "missing_transaction_reference",
        lambda memo, ref: not grading.cites_transaction_id(
            memo["customer_message"], ref["transaction_id"]
        ),
        "Which charge are you actually talking about? I have several recent "
        "transactions and I need the reference number for my records.",
    ),
    (
        "ambiguous_currency",
        lambda memo, ref: not grading.cites_currency(
            memo["customer_message"], ref["currency"]
        ),
        "What currency is that amount in? My account holds more than one and I "
        "cannot tell from your reply.",
    ),
    (
        "missing_policy_citation",
        lambda memo, ref: not grading.cites_policy_code(
            memo["customer_message"], ref["policy_code"]
        ),
        "Which policy does this fall under? I want the code so I can quote it if I "
        "have to escalate.",
    ),
]


def next_followup(memo: dict, reference: dict, already_fired: set[str]) -> tuple[str, str] | None:
    """Return ``(signal, customer_message)`` for the next push-back, or ``None``.

    ``None`` means the customer is satisfied and the session ends.
    """
    for signal, predicate, message in FOLLOWUP_RULES:
        if signal in already_fired:
            continue
        if predicate(memo, reference):
            return signal, message
    return None
