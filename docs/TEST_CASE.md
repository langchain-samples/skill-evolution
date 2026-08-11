# Test case: evolving an agent skill from multi-session trace history

## Hypothesis

Given an agent whose behaviour is governed by a `SKILL.md` file, the conversation
history of its past sessions — captured in LangSmith — contains enough signal to
rewrite that file into a measurably better one, without a human authoring the
improvements and without labelled training data.

The claim under test is deliberately narrow and falsifiable:

> Skill versions derived from cross-session trace history score higher on a held-out
> scenario set than the hand-written baseline, and the improvement comes from
> generalisable procedure rather than memorised answers.

## What makes it a test rather than a demonstration

Four controls, each of which can fail:

| Control | Mechanism | What it rules out |
|---|---|---|
| **Held-out measurement** | The optimizer only ever sees `train` sessions. Scoring happens on a disjoint LangSmith dataset. `H3` is a composition never seen in training. | Improvement that is really memorisation of the training scenarios. |
| **No answer key** | The digest handed to the optimizer excludes reference labels. It sees tool calls, customer push-back and metric scores — never the correct resolution. | The optimizer writing a lookup table instead of a procedure. |
| **Cross-session evidence gate** | A proposed rule is discarded unless ≥2 **distinct, verified** sessions support it. Enforced in `reflect.screen()`, not requested in a prompt. | Over-fitting to a single anecdote; hallucinated citations. |
| **Regression gate** | A new skill version is reverted unless its holdout composite beats the best seen so far by more than a margin. | Silent degradation across iterations. |

The evidence gate is the control that makes this specifically about *multi-session*
history. A single session cannot produce a rule, by construction.

## Procedure

```
                    ┌──────────────────────────────────────────┐
                    │  SKILL.md  v1  (thin, hand-written)      │
                    └────────────────────┬─────────────────────┘
                                         │ system prompt
                    ┌────────────────────▼─────────────────────┐
   1. run           │  6 training sessions, multi-turn         │
                    │  1 trace per turn, grouped by session_id │──► LangSmith
                    └────────────────────┬─────────────────────┘     (threads
                                         │                            + feedback)
                    ┌────────────────────▼─────────────────────┐
   2. harvest       │  read traces back: trajectories,         │◄── LangSmith
                    │  push-back signals, feedback scores      │
                    │  → cross-session evidence table          │
                    └────────────────────┬─────────────────────┘
                    ┌────────────────────▼─────────────────────┐
   3. reflect       │  optimizer PROPOSES rules + citations    │
                    │  code SCREENS: evidence ≥2, no answer    │
                    │  key, sanitised, rendered from template  │
                    └────────────────────┬─────────────────────┘
                    ┌────────────────────▼─────────────────────┐
   4. evaluate      │  holdout dataset, 8 evaluators           │──► LangSmith
                    │  improved? keep : revert                 │     (experiment)
                    └────────────────────┬─────────────────────┘
                                         └──► back to 1 with v2
```

## The task

A payment dispute triage agent with five read-only tools over a synthetic bank
(`lookup_transaction`, `list_account_transactions`, `check_fraud_flags`, `get_policy`,
`lookup_account`). It returns a structured `resolution` plus a `customer_message`.

The correct procedure is **not** in the baseline skill. It is discoverable only by
using the tools and observing what customers push back about:

- two postings, same merchant and amount within 72h, is a duplicate — not a failure;
- an open fraud hold freezes all self-service remediation and takes precedence over
  everything else, including duplicate reversal;
- a declined authorisation never captured funds, so there is nothing to refund;
- replies need the transaction reference, an unambiguous currency, and the governing
  policy code, or the customer has to ask again.

### Scenario design

Each training session exposes one or two failure modes; no session exposes the whole
procedure. Failure modes recur across sessions so that they clear the ≥2 evidence gate.

The gate's discriminating power is verified by `tests/test_controls.py` rather than by
observing a live run, because whether the gate fires depends on what the optimizer
happens to propose. The tests feed it single-session evidence, duplicate citations of
the same session, and citations of session ids that do not exist, and assert each is
dropped.

| Split | Sessions | Notes |
|---|---|---|
| `train` | S1–S6 | Duplicates ×2, declined payment, unauthorised + open fraud hold, NSF, authorisation hold (EUR) |
| `holdout` | H1–H5 | Different accounts, merchants, amounts, phrasings. H2 is unauthorised with *no* hold (credit is correct); **H3 is a duplicate on an account with an open hold** — unseen composition; H4 is GBP |

### The simulated customer is code, not a model

Follow-up turns fire from the same deterministic predicates the evaluators use
(`scenarios.FOLLOWUP_RULES`). If a reply omits the transaction reference, the customer
asks for it — costing a turn and emitting a friction signal. This keeps the friction
signal reproducible and free, and guarantees in-session friction and offline scoring
measure the same thing.

That guarantee cuts both ways: it also means the signal the optimizer learns from is
not independent of the metric it is scored on. This is the sharpest limitation of the
setup and is stated as such under *Signal and metric are coupled* in
[threats to validity](#threats-to-validity-stated-up-front) below.

## Metrics

Seven of the eight evaluators are deterministic string/sequence checks, so the
headline result does not depend on a judge.

| Metric | Type | In composite |
|---|---|---|
| `resolution_correct` | exact match vs reference | ✅ |
| `compliance_fraud_check` | trajectory: fraud cleared *before* the first remediation recommendation | ✅ |
| `cites_transaction_id` | reference appears in the transcript | ✅ |
| `cites_currency` | currency code or symbol present | ✅ |
| `cites_policy_code` | the correct `P-nnn` is cited | ✅ |
| `first_contact_resolution` | resolved without the customer re-asking | ✅ |
| `tool_efficiency` | minimal calls ÷ actual calls | reported only |
| `message_quality` | LLM judge on the writing | reported only |

`tool_efficiency` and `message_quality` are excluded from the composite on purpose:
the first is a cost measure that must not be traded against correctness, the second is
judge-dependent.

`compliance_fraud_check` is graded at the **first** turn that recommends remediation,
using the tools called up to that point. Checking fraud after already promising a
refund does not count.

## Pass criteria

1. Holdout composite for the final kept version > baseline composite.
2. No headline metric regresses below its baseline value.
3. `compliance_fraud_check` reaches 1.0 — for a bank this is a gate, not an average.
4. Every accepted rule in `history/vN.rationale.json` cites ≥2 verified sessions.
5. At least one proposed rule is rejected with a recorded reason, demonstrating the
   screen is load-bearing.

## Results

Two iterations. Agent under test: Haiku 4.5, pinned to temperature 0. Optimizer: Opus 5.
6 training sessions per iteration. Each version scored on 15 runs — 5 holdout scenarios
× 3 repetitions.

| metric | v1 (baseline) | v2 | v3 |
|---|---|---|---|
| `resolution_correct` | 0.800 | 0.867 | **1.000** |
| `compliance_fraud_check` | 0.600 | **1.000** | 0.800 |
| `cites_transaction_id` | 1.000 | 1.000 | 1.000 |
| `cites_currency` | 1.000 | 1.000 | 1.000 |
| `cites_policy_code` | 0.733 | 0.800 | 0.800 |
| `first_contact_resolution` | 0.000 | **0.800** | 0.200 |
| `tool_efficiency` | 0.729 | 0.902 | 0.760 |
| `message_quality` | 0.692 | 0.835 | 0.769 |
| **composite** | **0.689** | **0.911** | **0.800** |
| gate verdict | baseline | **kept** | **reverted** |

The baseline agent usually reached the right answer but *never* resolved on first
contact, and skipped the fraud check in 2 of 5 scenarios. That is the realistic
enterprise failure profile: not "wrong", but expensive and intermittently non-compliant.

### Finding 1 — cross-session history was sufficient to close the gap

v2 was written entirely from the six training traces, with no human authoring and no
labels. It took the compliance gate to 1.000 and first-contact resolution from 0.000 to
0.800, clearing the gate's threshold of 0.709 by a wide margin.

All 9 accepted rules cited ≥2 verified sessions — by construction, since the screen
discards anything that does not — and **zero citations were hallucinated**.

### Finding 2 — the second iteration made the skill worse, and the gate caught it

With v2 running, there was almost no friction left to learn from: 2 signals across six
sessions, down from 14 under v1. The optimizer nevertheless proposed a fresh 8-rule set.
Holdout composite fell to 0.800, driven by `first_contact_resolution` dropping 0.800 →
0.200, and v3 was reverted.

This is the most useful result in the run. A loop like this does not converge on its
own — left ungated it would have shipped a worse skill while reporting that it had
"learned more". The regression gate, not the optimizer, is what makes the technique safe
to iterate.

### Finding 3 — the loop converges fast, then degrades

One iteration consumed almost all the available friction signal (14 signals → 2). The
technique is not a perpetual improvement engine: it buys one or two productive iterations
per batch of new traffic and then needs new traffic. For a production deployment this
argues for triggering the loop on accumulated friction volume rather than on a schedule.

### Finding 4 — the measurement reproduces; the scenario set is still small

Two independent 15-run scorings of the *same* v1 skill reproduced five of six headline
metrics exactly, with composites of 0.700 and 0.689. The entire difference was one binary
flip in `resolution_correct` — the only headline metric that is a semantic judgement
rather than a string or sequence check. Both the v2 gain (+0.222) and the v3 drop
(−0.111) are well outside that movement.

Two caveats. `message_quality` cannot be pinned at all, because the judge model rejects
the `temperature` parameter — one more reason it sits outside the composite. And two
replications on 5 scenarios do not support a confidence interval: this establishes that
the harness reproduces, not that the effect sizes are precise.

### Deviations from the pre-registered criteria

- **Criterion 5 was met, but not by the evidence gate.** The single rejection across both
  iterations was the 500-character rule-length cap, at v3. The evidence gate was never
  exercised by a live proposal, so its discriminating power is demonstrated in
  `tests/test_controls.py` instead.
- **`resolution_correct` was not at ceiling.** An earlier single-sample run measured it at
  1.000 and that was recorded here as a threat to validity. With repetitions the baseline
  is 0.800 and v3 reaches 1.000, so the metric does have headroom — the ceiling finding
  was an artifact of scoring each scenario once.
- **No total size budget.** `MAX_RULE_CHARS` caps individual rules, but nothing caps rule
  count or total skill length. v2 came out at 9 rules and 3,681 characters; an earlier run
  over the same six scenarios produced 6 rules and 2,592. Rule volume appears to track
  digest volume rather than the amount of distinct signal in it, which is the mechanism
  behind skill bloat across iterations.

## Threats to validity, stated up front

- **Signal and metric are coupled.** The friction the optimizer learns from and the
  scores it is judged by come from the same predicates in `grading.py`. That is what
  makes the friction signal reproducible and free, but it also means the loop is
  improving against a proxy for its own evaluator, so the effect size here is
  optimistic relative to production, where a thumbs-down correlates only loosely with
  any offline metric. The generalization controls still bind independently of it — the
  optimizer never sees reference labels, train and holdout share no account,
  transaction or scenario id, H2 requires a resolution class no training scenario
  demonstrates, and H3 is an unseen composition — so the holdout gain is not
  memorization. But it is measured on a friendlier signal than production would give.
  Substituting a real one (thumbs-down, escalation, handoff rate) is the second control
  worth adding, after the blind-rewrite arm below.
- **Small n.** 6 training and 5 holdout scenarios, scored 3 times each. Enough to
  demonstrate the mechanism, not enough for a confidence interval. Scale the scenario set
  before quoting numbers as a benchmark.
- **Non-determinism, partly controlled.** The agent is a sampled LLM pinned to
  temperature 0, which makes the deterministic checks reproduce exactly but does not
  eliminate variation in tool ordering or the semantic outcome. Repetitions (
  `SKILLEVO_EVAL_REPETITIONS`) and the gate margin (`SKILLEVO_GATE_MARGIN`) exist because
  a single scoring pass is too coarse to decide a verdict.
- **The judge cannot be pinned.** Newer Claude models reject the `temperature` parameter
  outright, so `message_quality` keeps full sampling variance. It is advisory and
  deliberately outside the composite. It can also fail outright — in a verification run
  roughly one row in fifteen came back with the score fused into the reasoning string
  rather than parsed as a field. Those rows are recorded as a null score and skipped in
  the mean rather than imputed, so `message_quality` is a mean over fewer runs than the
  deterministic metrics.
- **The traces are not isolated as a cause.** The claim under test is that trace history
  carries the signal, but there is no arm that removes the history. What is demonstrated
  is that the loop improved the skill — not that the trace content is what improved it. A
  blind-rewrite arm (same optimizer, no digest, same holdout) would settle this and is the
  most valuable control still missing.
- **The optimizer reads policy text from traces.** Consolidating it into the skill is
  legitimate learning from history, but it means part of the gain is caching knowledge
  rather than discovering procedure. Both are reported in the rationale artifact.

## Why the two-region prompt matters

The agent's system prompt is `HARNESS_PREAMBLE` + `<skill>…</skill>`. The optimizer
can only write the skill region. The output contract, the "state only what tools
returned" rule, and the "you have no authority to move money" limit live in the
preamble and are not reachable by the loop.

This is what makes the loop safe to run unattended: a drifting or poisoned optimizer
can produce worse *advice*, which the regression gate catches, but it cannot widen the
agent's authority or remove its output contract. Notably, the preamble contains **no
dispute procedure** — seeding it there would void the experiment.
