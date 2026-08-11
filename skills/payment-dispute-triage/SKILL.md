---
name: payment-dispute-triage
version: 2
source: evolved from LangSmith trace history
evolved_at: 2026-08-10T19:55:21.325329+00:00
derived_from: digest-v1-train.json
evidence_sessions: 6
audit_trail: skills/payment-dispute-triage/history/v2.rationale.json
---

# Payment Dispute Triage

## Procedure

1. **Know the policy map before calling tools** Use this map and fetch each policy at most once: duplicate/repeat postings from one merchant -> P-101 (reverse the later posting within 3 business days, no provisional credit); declined or failed authorization -> P-204 (no funds captured, nothing to refund, advise retry); customer-denied charge -> P-311 (provisional credit within 10 business days, which itself opens fraud review).
2. **Gather all facts in one pass, no repeat lookups** Plan one lookup pass: resolve the cited transaction, list the account's transactions once with a single sensible window, check fraud flags, and fetch the one governing policy. Never re-run a lookup whose result is already in the conversation, and never widen the transaction window repeatedly when the first result already shows the full picture.
3. **Check fraud holds before any credit or reversal** Always check the account's fraud flags before recommending a reversal or provisional credit. If an open fraud hold exists, self-service reversal and provisional credit are frozen: route the case to the fraud team, cite the hold reference, and explain the exception in the governing policy.

## Every reply must include

- **Cite policy code and transaction id upfront** In the very first reply, always state the governing policy code and title verbatim, plus the exact transaction reference(s) you are acting on with amount, currency and timestamp. Customers escalate and re-ask when either is missing, costing extra turns.
- **State currency name explicitly, never symbol alone** Always write amounts with the three-letter currency code from the transaction record (e.g. '250.00 EUR'), not a bare symbol or number, and confirm whether the currency matches the account's supported currencies. Flag any cross-currency charge and explain FX handling rather than treating it as a plain domestic amount.
- **Name which posting is reversed and which stays** When two or more related postings exist, identify each by reference and timestamp and say explicitly which one is being actioned and which remains, then give the customer the single reference number to quote on follow-up. Do not describe charges only by merchant, time of day or amount.

## Precedence and edge cases

- **Never call a tool with unknown placeholder ids** Derive required identifiers from an earlier tool result before calling a tool; if an id is not yet known, look it up from the transaction record first. Do not invoke a tool with a placeholder or guessed identifier.
- **Confirm a real duplicate pair before reversing** Only apply the duplicate policy when the account history actually shows two postings from the same merchant for the same amount within 72 hours. A single charge the customer believes they already paid elsewhere, or a pending authorization with no matching second posting, is not a duplicate; handle it as an authorization/merchant-billing issue instead of recommending reversal.
- **Handle declines and add-on fees separately** For a declined authorization, state that no funds were captured so there is nothing to refund, advise retry, and note escalation only applies if a hold is still visible after the policy window. If the customer also reports a fee, treat it as a distinct item: look for the fee posting and address it under its own policy rather than folding it into the decline answer.
