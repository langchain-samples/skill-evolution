# Evolving an agent skill from LangSmith trace history

A proof of concept for one question: **can the conversation history an agent has
already produced be used to improve the agent's own instructions?**

The agent's behaviour lives in a single `SKILL.md` file. It starts thin and
hand-written. The loop runs the agent across a set of multi-turn sessions, reads those
traces back out of LangSmith, and rewrites the skill from what the history shows —
then proves the rewrite helped by scoring it on scenarios it has never seen.

```
run sessions ──► LangSmith ──► harvest ──► reflect ──► evaluate ──► keep or revert
   (traced)      (threads,     (digest)   (rules +    (holdout      (regression
                  feedback)                evidence)   dataset)       gate)
```

The domain is payment dispute triage over a synthetic bank. **All data is invented
fixture data** — no real accounts, customers, or systems, and no network calls beyond
LangSmith and the model API.

## What is actually being demonstrated

Anyone can put an LLM in a loop rewriting a prompt. Four things here are load-bearing,
and each is enforced by code rather than requested in a prompt:

1. **Rules must be corroborated across sessions.** A candidate rule needs ≥2 distinct
   *verified* sessions behind it or it is discarded. Citations are checked against the
   registry of sessions that actually exist, so a hallucinated citation fails. This is
   what makes the technique depend on multi-session history: one session cannot produce
   a rule, by construction.

2. **The optimizer never sees the answer key.** The digest contains trajectories,
   customer push-back and scores — never the correct resolution. Rule text naming a
   specific transaction or account is rejected. So the skill cannot become a lookup
   table, and holdout improvement means something. What it does *not* mean is a
   production-sized effect: the simulated customer pushes back using the same
   predicates the evaluators score with, so the loop learns against a proxy for its own
   metric. The generalization controls hold regardless — but the gain is measured on a
   friendlier signal than real traffic gives. See
   [threats to validity](docs/TEST_CASE.md#threats-to-validity-stated-up-front).

3. **The model proposes; code decides and renders.** The optimizer returns structured
   rules. Code screens them, sanitizes the text, and renders `SKILL.md` from a fixed
   template. The model never writes the file, so harvested trace content — untrusted
   input — cannot restructure the document or escape the `<skill>` block.

4. **A version only survives if it measurably wins.** Every candidate is scored on a
   held-out LangSmith dataset — each scenario run several times, with the agent pinned to
   temperature 0 — and reverted unless the composite beats the best seen so far by more
   than a margin. A strict `>` would let one scenario flipping once decide the gate.
   Compliance is exempt from that arithmetic: the composite is an unweighted mean, so
   `compliance_fraud_check` would otherwise be worth 0.167 and could be bought back with
   gains in convenience metrics. It is gated separately, and any regression against the
   incumbent reverts the candidate whatever the composite does.

The full specification, including scenario design, metrics and threats to validity, is
in [`docs/TEST_CASE.md`](docs/TEST_CASE.md).

## Result from the run in this repo

Each version scored on 15 runs — 5 holdout scenarios × 3 repetitions — with the agent
pinned to temperature 0.

| metric | v1 baseline | v2 | v3 |
|---|---|---|---|
| `compliance_fraud_check` | 0.600 | **1.000** | 0.800 |
| `first_contact_resolution` | 0.000 | **0.800** | 0.200 |
| `cites_policy_code` | 0.733 | 0.800 | 0.800 |
| `resolution_correct` | 0.800 | 0.867 | 1.000 |
| `tool_efficiency` | 0.729 | 0.902 | 0.760 |
| **composite** | **0.689** | **0.911** | **0.800** |
| gate | baseline | kept | **reverted** |

Two findings worth more than the first row:

- **The second iteration did not improve the skill, and the gate refused it.** With
  little friction left to learn from — 2 signals across six sessions, down from 14 — the
  optimizer still proposed a fresh rule set. Here first-contact resolution fell from
  0.800 to 0.200 and `compliance_fraud_check` from 1.000 to 0.800, so v3 is rejected by
  the composite and by the compliance floor independently. A later run degraded
  differently: v3 came back *exactly level* with v2 on every headline metric and was
  refused for failing to clear the margin rather than for regressing. Both failure modes
  end the same way, and neither is the optimizer declining to act. **The gate, not the
  optimizer, is what makes this safe to iterate.**
- **The loop converges fast, then degrades.** One iteration consumed almost all the
  available friction signal. This is not a perpetual improvement engine: it buys one or
  two good iterations per batch of new traffic, and then needs new traffic.

Run-to-run, two independent 15-run scorings of the same v1 skill reproduced five of six
headline metrics exactly, with composites of 0.700 and 0.689. So the v2 gain is a wide
margin and the v3 drop of 0.111 is well outside that movement — but this is still 5
scenarios. Scale the scenario set before quoting any of it as a benchmark.

Two later verification runs, from a clean checkout on a freshly resolved dependency set,
reproduced this independently. Across four scorings the v1 baseline takes only two
values — 0.689, 0.700, 0.700, 0.689 — one metric-flip apart, so the 0.02 gate margin
sits just above the entire observed noise band. v2 landed at 0.933 and 0.945 against the
0.911 recorded here, from rule sets of nine, ten and eleven rules respectively: the
optimizer cannot be pinned, so it proposes something different every time and arrives in
the same place. **The rule text does not reproduce; the improvement does.**

The second run also carried through to a second iteration, where the friction available
to the optimizer collapsed from 12 signals to *zero* — v2 had eliminated it. Given an
empty evidence table the optimizer still proposed 11 rules, and the gate reverted them.
That is the clearest statement of what this loop is: it converts a batch of friction
into one good iteration, and then has nothing to work with until new traffic arrives.

## Setup

```bash
uv sync                      # or: pip install -e .
cp .env.example .env         # then fill in the keys
```

`uv.lock` is committed, so `uv sync` reproduces the environment the documented run was
produced on. The LangChain dependencies are compatible-release pinned rather than left
open: the SDK surface the harvester uses moves across minor versions, and an unpinned
resolve months from now would build an environment nobody has tested.

Requires a LangSmith API key (or an authenticated `langsmith` CLI profile) and model
credentials. The agent under test defaults to Haiku 4.5 and the optimizer to Opus 5 —
see `.env.example` to change either.

> The agent under test is deliberately a small model. A well-evolved skill making a
> cheap model reliable is the actual value of this technique; a frontier model scores
> near ceiling from the thin baseline and leaves nothing to measure.

## Run it

```bash
python -m skillevo loop --iterations 2
```

That is the whole test case. It evaluates the baseline, then for each iteration runs
the training sessions, harvests them from LangSmith, proposes a new skill version,
scores it on the holdout set, and keeps or reverts it. Budget about 15 minutes.

The repo ships with `SKILL.md` at the v1 baseline, so that command starts where the
documented run started. Every evolved version is kept in `history/` — the loop rewrites
`SKILL.md` in place, so expect a dirty working tree afterwards.

**Run `python -m skillevo reset --hard` before any re-run.** The scoreboard in
`artifacts/` persists between invocations, and `loop` skips the baseline evaluation for a
version it has already scored and gates the next candidate against the best composite on
record. Re-running without a reset therefore measures a *new* candidate against the
*previous* run's winner and never re-establishes the baseline — both behaviours are
deliberate, but together they silently make the second run a different experiment.

Set `SKILLEVO_RUN_ID` to a fresh value per invocation. The harvester filters on it, so
without it a re-run mines the previous run's sessions too and its evidence counts
inflate. `.env.example` lists the other knobs — repetitions, gate margin, concurrency.

Individual steps, if you would rather watch them one at a time:

```bash
python -m skillevo status      # current skill version + scoreboard
python -m skillevo sessions    # run the 6 training sessions, traced
python -m skillevo harvest     # pull them back from LangSmith into a digest
python -m skillevo reflect     # propose + screen + write the next SKILL.md
python -m skillevo evaluate    # score the current skill on the holdout dataset
python -m skillevo reset --hard  # back to the v1 baseline for a clean re-run
```

## What to look at in LangSmith

| Surface | Where | What it shows |
|---|---|---|
| **Threads** | project `skill-evolution`, Threads tab | Each session as one thread. The extra turns *are* the friction signal — the customer re-asking because the reply was incomplete. |
| **Trace detail** | any `triage_turn` trace | The tool trajectory the harvester mines. On a baseline failure, `check_fraud_flags` is missing before a refund is promised. |
| **Feedback** | any final turn | The deterministic scores, written back onto the trace so the signal is queryable in the platform rather than only on disk. |
| **Filter by version** | run filter `has(metadata, '{"skill_version": 2}')` | Slice all sessions by the skill version that produced them. |
| **Experiment comparison** | dataset `payment-dispute-triage-holdout` → Compare | v1 vs v2 vs v3 side by side, per metric and per scenario. This is the result. |

## Layout

```
skills/payment-dispute-triage/
  SKILL.md                  the live skill — rewritten in place each iteration
  history/v1.md, v2.md …    every version, archived
  history/vN.rationale.json audit trail: each rule, its evidence sessions,
                            and every rejected rule with the reason
skillevo/
  agent.py       agent under test; fixed preamble + mutable <skill> region
  domain.py      synthetic bank fixtures and the five read-only tools
  scenarios.py   train/holdout splits + the deterministic simulated customer
  sessions.py    multi-turn runner; one trace per turn, grouped by session_id
  harvest.py     LangSmith → cross-session digest (no reference labels)
  reflect.py     propose → screen → render the next SKILL.md
  evaluate.py    holdout dataset, 8 evaluators, one experiment per version
  grading.py     the deterministic checks, shared by customer and evaluators
  cli.py         commands above, including the regression gate
tests/
  test_controls.py  adversarial tests for the four controls (offline, no API calls)
artifacts/       digests, evidence registry, scoreboard (gitignored)
```

```bash
python3 tests/test_controls.py    # 20 tests, no LangSmith or model calls
```

The tests matter because a live loop run only shows what the optimizer *happened* to
propose. They feed the screen single-session evidence, duplicate citations of the same
session, citations of sessions that do not exist, rules naming a specific transaction,
and prompt-injection payloads — and assert each is dropped. They also drive the gate
with a candidate whose aggregate gains outweigh a total compliance failure, and assert
the floor vetoes it regardless.

## Reading the audit trail

`history/vN.rationale.json` is the point of the exercise for a regulated setting: every
rule in the skill traces to the sessions that justify it, and every rejected proposal
is recorded with why. A reviewer can answer "why does the agent behave this way?"
without rerunning anything.

```json
{
  "accepted_rules": [{
    "title": "...", "guidance": "...",
    "evidence_verified": ["S1-v1-…", "S4-v1-…"],
    "failure_mode": "...", "target_metric": "..."
  }],
  "rejected_rules": [{
    "title": "...",
    "rejected_because": "only 1 verified evidence session(s), need 2"
  }]
}
```

## Extending it

The pieces most worth swapping for a real evaluation:

- **Scenario set** — `scenarios.py`. The 6/5 split demonstrates the mechanism; scale it
  before quoting numbers.
- **Signal source** — the friction signals here are synthetic predicates. In
  production, substitute real thumbs-down feedback, escalation events, or handoff rates,
  read from the same LangSmith feedback API.
- **A repository of skills** — a skill here is just a file injected into a system
  prompt, so the loop is framework-agnostic and pointing it at a Deep Agents
  `SkillsMiddleware` directory is mechanically easy. Doing it *correctly* is not a
  config change, because on-demand loading breaks credit assignment. A session may carry
  several skills, and a bad outcome can mean the wrong skill was selected, the right
  skill had bad content, or two skills conflicted — only the middle one is fixable by
  rewriting text. Record `skills_loaded` per turn in run metadata and harvest per skill
  against it, or the loop will confidently rewrite a good skill in response to a routing
  failure. Three further costs to budget for: friction divides across the repository, so
  each skill clears the ≥2-session evidence gate far more slowly than the single-skill
  case here (which went from 12 signals to zero in one iteration) and the long tail may
  never clear it at all; skill *selection* is a routing problem this method does not
  address; and a per-skill gate cannot see one skill regressing another, which needs a
  cross-cutting regression suite in addition to per-skill holdouts. The expensive part
  of that list is the eval content, not the code.

## Known limits

The controls are the point of this repo; the execution layer around them is a spike.
Three things to change before pointing it at production traffic:

- **Harvest reads at most 100 runs** (`harvest.py`) and does not warn when it truncates,
  so at volume the evidence counts behind every rule would be silently wrong. The cap is
  self-imposed — the client already paginates — so replace it with a time-windowed
  incremental harvest: filter on `start_time` since the last watermark and let the window
  bound the read rather than the row count. Windowing rather than plain pagination also
  keeps evidence recent, which stops rules being justified by a retired policy regime.

- **The skill is written before the gate runs** (`reflect.py`), so a crash mid-evaluation
  leaves an ungated version live on disk. Write the candidate to a staging path, evaluate
  it there, and promote it to `SKILL.md` by atomic rename only once the gate passes, so
  the live file only ever holds a version that cleared evaluation. That staging path is
  also the natural place to hang a human approval step.

- **Single writer.** The skill file, scoreboard and evidence registry are
  read-modify-write local files with no locking, and the skill name is a module constant
  — two concurrent loops clobber each other. Move that state into a store keyed by
  `(tenant, skill, version)` with compare-and-swap on write, so concurrent runs serialize
  instead of overwriting.

## Better signal — traces for coverage, outcomes for labels

Traces are the right substrate for *improving* a skill and the wrong one for *scoring*
it. This repo already splits those jobs: harvested history feeds `reflect`, which only
proposes, and the verdict comes from a labelled holdout the optimizer never sees. That
split is deliberate. A trace records what the agent did; it carries no counterfactual
and no ground truth, so on its own it cannot say what the agent should have done.

What traces uniquely provide is coverage. You cannot write an eval for a failure mode
you have not observed, and traces are the only signal spanning the whole live
distribution — which is exactly how this loop discovers that a refund gets promised
before the fraud check runs.

What they miss is the error class that matters most. The loop learns from *friction* —
the customer having to re-ask. An incomplete answer produces friction and gets learned
from. A wrong answer the customer pushes back on produces friction and gets learned
from. But an answer that is confidently wrong and simply **accepted** produces no
friction at all: it enters no digest and generates no rule. Here that gap is masked,
because the simulated customer is a perfect oracle that pushes back precisely when a
grading predicate fails. In production, silence means "satisfied or wrong", and a trace
cannot tell those apart.

The fix is not a different substrate but a second one joined to it. Ordered by
information per unit — which runs opposite to how much of it exists:

| signal | density | what it establishes |
|---|---|---|
| downstream outcome — dispute reopened, credit reversed, callback inside 7 days | sparse, delayed | what actually happened |
| human override or correction | sparse | a labelled counterfactual: agent said X, right answer was Y |
| escalation / handoff | sparse | unambiguous failure, with the case attached |
| explicit thumbs-down | sparse, biased | dissatisfaction only |
| trace friction (used here) | dense | weak proxy, silent on the worst class |

Join a trace to its downstream outcome and the digest gains what it currently lacks:
full-distribution coverage *with* ground truth, on real cases, including the silent
failures friction can never surface. For payment disputes those outcomes already exist —
reopen rates, chargeback reversals and regulatory timers are tracked because they have
to be. Wiring that join is a larger change than the signal swap suggested under
[Extending it](#extending-it), and a more valuable one.
