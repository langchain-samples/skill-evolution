"""Offline measurement: a LangSmith dataset over the holdout split, one experiment
per skill version.

This is what turns the loop from a demo into a test case. The optimizer only ever
sees training sessions; every claim of improvement is measured here, on scenarios it
has not observed, including one composition (``H3``) that never appears in training.

The evaluators recompute every deterministic score from the raw trajectory rather
than reading numbers the target produced, so the scoring path is independent of the
code being scored.
"""

from __future__ import annotations

from langsmith import Client, evaluate
from pydantic import BaseModel, Field

from skillevo import config, grading
from skillevo.agent import build_agent, load_skill
from skillevo.scenarios import BY_ID, HOLDOUT
from skillevo.sessions import run_session, session_compliance

_client = Client()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def upsert_dataset(force: bool = False) -> str:
    """Create the holdout dataset if absent. Returns the dataset id."""
    existing = list(_client.list_datasets(dataset_name=config.DATASET_NAME))
    if existing and not force:
        dataset = existing[0]
        count = len(list(_client.list_examples(dataset_id=dataset.id)))
        if count == len(HOLDOUT):
            print(f"  dataset {config.DATASET_NAME!r} already has {count} examples")
            return str(dataset.id)
        print(f"  dataset has {count} examples, expected {len(HOLDOUT)}; recreating")
        _client.delete_dataset(dataset_id=dataset.id)
    elif existing and force:
        _client.delete_dataset(dataset_id=existing[0].id)

    dataset = _client.create_dataset(
        dataset_name=config.DATASET_NAME,
        description=(
            "Held-out payment dispute scenarios. Disjoint from the training sessions "
            "the skill optimizer observes. H3 is a composition unseen in training: a "
            "duplicate charge on an account under an active fraud hold."
        ),
    )
    _client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {
                    "scenario_id": scenario.id,
                    "opening_message": scenario.opening_message,
                    "account_id": scenario.account_id,
                    "transaction_id": scenario.transaction_id,
                },
                "outputs": scenario.reference,
            }
            for scenario in HOLDOUT
        ],
    )
    print(f"  created dataset {config.DATASET_NAME!r} with {len(HOLDOUT)} examples")
    return str(dataset.id)


# --------------------------------------------------------------------------- #
# Evaluators
# --------------------------------------------------------------------------- #


def resolution_correct(outputs: dict, reference_outputs: dict) -> dict:
    return {
        "key": "resolution_correct",
        "score": float(outputs["resolution"] == reference_outputs["resolution"]),
    }


def compliance_fraud_check(outputs: dict) -> dict:
    """Compliance gate: fraud cleared before any remediation was recommended."""
    return {"key": "compliance_fraud_check", "score": outputs["compliance"]}


def cites_transaction_id(outputs: dict, reference_outputs: dict) -> dict:
    return {
        "key": "cites_transaction_id",
        "score": float(
            grading.cites_transaction_id(outputs["transcript"], reference_outputs["transaction_id"])
        ),
    }


def cites_currency(outputs: dict, reference_outputs: dict) -> dict:
    return {
        "key": "cites_currency",
        "score": float(grading.cites_currency(outputs["transcript"], reference_outputs["currency"])),
    }


def cites_policy_code(outputs: dict, reference_outputs: dict) -> dict:
    return {
        "key": "cites_policy_code",
        "score": float(
            grading.cites_policy_code(outputs["transcript"], reference_outputs["policy_code"])
        ),
    }


def first_contact_resolution(outputs: dict) -> dict:
    """Did the customer get a complete answer without having to re-ask?"""
    return {"key": "first_contact_resolution", "score": float(outputs["turns_used"] <= 1)}


def tool_efficiency(outputs: dict) -> dict:
    return {"key": "tool_efficiency", "score": grading.tool_efficiency(outputs["tool_sequence"])}


class Judgment(BaseModel):
    score: float = Field(description="0.0 to 1.0.")
    reasoning: str = Field(description="One or two sentences.")


JUDGE_PROMPT = """\
You are auditing a bank's reply to a customer's payment dispute.

Customer's opening message:
{opening}

The bank's reply(ies):
{transcript}

Score the reply from 0.0 to 1.0 on whether it is something a bank would be happy to
send: does it clearly state what happened, what will happen next, and by when, in
plain language, without hedging or unnecessary jargon? Judge only the writing and
completeness of the explanation, not whether the underlying decision was correct.\
"""


def message_quality(inputs: dict, outputs: dict) -> dict:
    """The one judged metric. Secondary to the deterministic gates above."""
    from langchain.chat_models import init_chat_model

    judge = init_chat_model(
        config.JUDGE_MODEL, **config.sampling_kwargs(config.JUDGE_MODEL)
    ).with_structured_output(Judgment)
    verdict = judge.invoke(
        JUDGE_PROMPT.format(opening=inputs["opening_message"], transcript=outputs["transcript"])
    )
    return {
        "key": "message_quality",
        "score": max(0.0, min(1.0, verdict.score)),
        "comment": verdict.reasoning,
    }


EVALUATORS = [
    resolution_correct,
    compliance_fraud_check,
    cites_transaction_id,
    cites_currency,
    cites_policy_code,
    first_contact_resolution,
    tool_efficiency,
    message_quality,
]

# The gates that constitute "did the skill get better". message_quality and
# tool_efficiency are reported but excluded: the first is judge-dependent, the second
# is a cost measure that should not be traded against correctness.
HEADLINE_METRICS = [
    "resolution_correct",
    "compliance_fraud_check",
    "cites_transaction_id",
    "cites_currency",
    "cites_policy_code",
    "first_contact_resolution",
]


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #


def run_experiment(
    skill_version: int | None = None,
    max_concurrency: int | None = None,
    num_repetitions: int | None = None,
):
    """Score the current skill file against the holdout dataset.

    Each example is run ``num_repetitions`` times. The agent is a sampled LLM pinned
    to temperature 0, which removes most but not all run-to-run variance -- tool
    ordering can still differ -- so a single pass over 5 examples is too coarse to
    decide a gate whose margin is one scenario wide.
    """
    max_concurrency = config.EVAL_CONCURRENCY if max_concurrency is None else max_concurrency
    num_repetitions = config.EVAL_REPETITIONS if num_repetitions is None else num_repetitions
    skill_body, version = load_skill()
    version = skill_version if skill_version is not None else version
    upsert_dataset()
    agent = build_agent(skill_body)

    def target(inputs: dict) -> dict:
        scenario = BY_ID[inputs["scenario_id"]]
        record = run_session(
            scenario,
            skill_version=version,
            agent=agent,
            phase="eval",
            post_feedback=False,  # the experiment's own evaluators are the scores here
        )
        return {
            "resolution": record["final_resolution"],
            "transcript": "\n\n".join(t["customer_message"] for t in record["turns"]),
            "tool_sequence": record["tool_sequence"],
            "turns_used": record["turns_used"],
            "compliance": session_compliance(record["turns"]),
            "friction_signals": record["friction_signals"],
        }

    results = evaluate(
        target,
        data=config.DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix=f"skill-v{version}",
        description=f"Holdout scoring for {config.SKILL_NAME} skill v{version}",
        metadata={
            "skill_version": version,
            "agent_model": config.AGENT_MODEL,
            "agent_temperature_pinned": config.pins_temperature(config.AGENT_MODEL),
            "num_repetitions": num_repetitions,
            **({"run_id": config.RUN_ID} if config.RUN_ID else {}),
        },
        max_concurrency=max_concurrency,
        num_repetitions=num_repetitions,
        client=_client,
    )
    return results


def summarize(results) -> dict:
    """Mean score per metric, plus the composite over the headline gates.

    Also reports the per-run composite range. With repetitions enabled, that range is
    the honest read on whether a composite delta is signal: a 0.03 improvement inside
    a 0.20-wide spread is not one.
    """
    per_metric: dict[str, list[float]] = {}
    run_composites: list[float] = []

    for row in results:
        row_scores: dict[str, float] = {}
        for evaluation in row.get("evaluation_results", {}).get("results", []):
            if evaluation.score is not None:
                per_metric.setdefault(evaluation.key, []).append(float(evaluation.score))
                row_scores[evaluation.key] = float(evaluation.score)
        present = [row_scores[k] for k in HEADLINE_METRICS if k in row_scores]
        if present:
            run_composites.append(sum(present) / len(present))

    means = {key: round(sum(v) / len(v), 3) for key, v in sorted(per_metric.items())}
    headline = [means[k] for k in HEADLINE_METRICS if k in means]
    means["composite"] = round(sum(headline) / len(headline), 3) if headline else 0.0

    if run_composites:
        means["n_runs"] = len(run_composites)
        means["composite_min"] = round(min(run_composites), 3)
        means["composite_max"] = round(max(run_composites), 3)
    return means
