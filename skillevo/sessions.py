"""Run scenarios as multi-turn sessions, traced to LangSmith.

Trace shape, and why it matters for the test case:

* one **trace per turn** (``triage_turn``), which is how a deployed chat agent
  actually looks -- one trace per inbound customer message;
* all turns of a session share ``metadata.session_id``, so LangSmith groups them
  into a **thread**. The thread *is* the conversation history the loop later mines;
* every turn carries ``metadata.skill_version``, which is what lets the harvester
  select "all sessions run under skill v2" and compare across versions.

Feedback is attached programmatically to each turn, so the friction signal is
queryable in LangSmith rather than living only in local files.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

from skillevo import config, grading
from skillevo.agent import DisputeMemo, build_agent, load_skill, tool_sequence
from skillevo.scenarios import Scenario, next_followup

_client = Client()


@traceable(name="triage_turn", run_type="chain")
def _run_turn(agent, user_message: str, thread_config: dict, prior_len: int) -> dict:
    """One inbound customer message. Returns the turn summary and its run id."""
    run_tree = get_current_run_tree()
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]}, config=thread_config)

    messages = result["messages"]
    memo: DisputeMemo | None = result.get("structured_response")
    if memo is None:
        raise RuntimeError("agent returned no structured_response; check response_format wiring")

    return {
        "run_id": str(run_tree.id) if run_tree else None,
        "resolution": memo["resolution"],
        "customer_message": memo["customer_message"],
        "tool_calls": tool_sequence(messages[prior_len:]),
        "message_count": len(messages),
    }


def session_compliance(turns: list[dict]) -> float:
    """Was fraud ever cleared *before* the first remediation recommendation?

    Graded at the first turn that recommends anything other than ``needs_more_info``,
    using the tools called up to and including that turn. Checking fraud only after
    already telling the customer they will be refunded does not count.
    """
    cumulative: list[str] = []
    for turn in turns:
        cumulative.extend(turn["tool_calls"])
        if turn["resolution"] != "needs_more_info":
            return float(grading.has_fraud_check(cumulative))
    return 1.0  # never recommended remediation, so the gate was never crossed


def run_session(
    scenario: Scenario,
    skill_version: int,
    agent=None,
    phase: str = "train",
    post_feedback: bool = True,
) -> dict:
    """Play one scenario to completion and return a structured session record."""
    agent = agent or build_agent()
    session_id = f"{scenario.id}-v{skill_version}-{uuid.uuid4().hex[:8]}"

    # History accumulates within the session and cannot leak between sessions,
    # because session_id is the checkpointer thread_id and is unique per run.
    turns: list[dict] = []
    signals: list[str] = []
    fired: set[str] = set()
    user_message = scenario.opening_message
    prior_len = 0

    for turn_index in range(config.MAX_TURNS_PER_SESSION):
        turn = _run_turn(
            agent,
            user_message,
            {"configurable": {"thread_id": session_id}},
            prior_len,
            langsmith_extra={
                "project_name": config.PROJECT,
                "metadata": {
                    # session_id is what LangSmith groups threads on.
                    "session_id": session_id,
                    "thread_id": session_id,
                    "scenario_id": scenario.id,
                    "split": scenario.split,
                    "phase": phase,
                    "skill_version": skill_version,
                    "turn_index": turn_index,
                    # Present only when set, so the harvester's filter stays exactly
                    # as it was for traces recorded before run tagging existed.
                    **({"run_id": config.RUN_ID} if config.RUN_ID else {}),
                },
            },
        )
        prior_len = turn["message_count"]
        turn["turn_index"] = turn_index
        turn["user_message"] = user_message
        turns.append(turn)

        followup = next_followup(
            {"resolution": turn["resolution"], "customer_message": turn["customer_message"]},
            scenario.reference,
            fired,
        )
        if followup is None:
            turn["followup_signal"] = None
            break

        signal, user_message = followup
        turn["followup_signal"] = signal
        signals.append(signal)
        fired.add(signal)
    else:
        # Ran out of turns with the customer still dissatisfied.
        signals.append("unresolved_at_turn_limit")

    cumulative_tools = [name for turn in turns for name in turn["tool_calls"]]
    final = turns[-1]
    transcript = "\n\n".join(turn["customer_message"] for turn in turns)
    scores = grading.score_memo(
        resolution=final["resolution"],
        customer_message=final["customer_message"],
        tool_sequence=cumulative_tools,
        reference=scenario.reference,
        turns_used=len(turns),
        transcript=transcript,
    )
    # Replace the presence-only check with the stricter ordering-aware one.
    scores["compliance_fraud_check"] = session_compliance(turns)
    scores["composite"] = (
        scores["resolution_correct"]
        + scores["compliance_fraud_check"]
        + scores["cites_transaction_id"]
        + scores["cites_currency"]
        + scores["cites_policy_code"]
        + scores["first_contact_resolution"]
    ) / 6

    record = {
        "session_id": session_id,
        "scenario_id": scenario.id,
        "split": scenario.split,
        "phase": phase,
        "skill_version": skill_version,
        "turns_used": len(turns),
        "turns": turns,
        "friction_signals": signals,
        "tool_sequence": cumulative_tools,
        "redundant_tool_calls": grading.redundant_tool_calls(cumulative_tools),
        "final_resolution": final["resolution"],
        "final_message": final["customer_message"],
        "scores": scores,
    }

    if post_feedback:
        _attach_feedback(record)
    return record


def _attach_feedback(record: dict) -> None:
    """Push the deterministic scores back onto the traces as LangSmith feedback."""
    last_run_id = record["turns"][-1].get("run_id")
    if not last_run_id:
        return
    for key, score in record["scores"].items():
        try:
            _client.create_feedback(run_id=last_run_id, key=key, score=score)
        except Exception as exc:  # feedback is observability, never fatal
            print(f"  ! feedback {key} failed: {type(exc).__name__}: {exc}")
    for turn in record["turns"]:
        signal = turn.get("followup_signal")
        if signal and turn.get("run_id"):
            try:
                _client.create_feedback(
                    run_id=turn["run_id"],
                    key="customer_pushed_back",
                    score=0,
                    comment=signal,
                )
            except Exception:
                pass


def run_split(
    scenarios: list[Scenario],
    phase: str = "train",
    skill_version: int | None = None,
) -> list[dict]:
    """Run every scenario in a split under the current skill file."""
    skill_body, version = load_skill()
    version = skill_version if skill_version is not None else version
    agent = build_agent(skill_body)

    records = []
    for scenario in scenarios:
        print(f"  [{scenario.id}] {scenario.split} ...", end=" ", flush=True)
        record = run_session(scenario, skill_version=version, agent=agent, phase=phase)
        print(
            f"{record['final_resolution']} "
            f"({record['turns_used']} turn(s), composite={record['scores']['composite']:.2f})"
        )
        records.append(record)

    wait_for_all_tracers()  # make sure every turn is flushed before harvesting
    return records
