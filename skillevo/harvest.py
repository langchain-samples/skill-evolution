"""Read session history back out of LangSmith and reduce it to a digest.

This module is the reason the test case is about LangSmith rather than about a
local log file. Everything the optimizer sees is reconstructed from the platform:

* turn structure and thread grouping, from ``metadata.session_id``;
* the tool trajectory, from the child ``tool`` runs of each turn's trace;
* what the customer complained about, from ``customer_pushed_back`` feedback;
* how well the session scored, from the deterministic feedback keys.

Deliberately **not** included: the ground-truth reference labels. The optimizer
never sees the answer key, so it cannot memorize scenario-to-resolution mappings --
it has to infer general procedure from friction and from policy text it observes in
tool outputs. That is what makes an improvement on the holdout set meaningful.

Harvested trace content is untrusted input. It is read as JSON/plain strings only,
never deserialized into project classes, and every string that reaches the
optimizer's prompt is truncated and sanitized downstream.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from langsmith import Client

from skillevo import config

_client = Client()

MAX_TOOL_OUTPUT_CHARS = 420
MAX_REPLY_CHARS = 400

# Keys written by sessions._attach_feedback that represent scores, not signals.
_SIGNAL_KEY = "customer_pushed_back"


def _metadata(run) -> dict:
    return ((run.extra or {}).get("metadata") or {}) if run is not None else {}


def _truncate(value, limit: int) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _tool_output_text(outputs) -> str:
    """Pull the payload out of a serialized ``ToolMessage``.

    Tool run outputs arrive as ``{"output": {"content": "<json>", "type": "tool", …}}``.
    Digesting the wrapper verbatim would spend the whole truncation budget on escaped
    envelope fields instead of the policy text the optimizer needs to read.
    """
    if isinstance(outputs, dict):
        inner = outputs.get("output", outputs)
        if isinstance(inner, dict) and "content" in inner:
            return _truncate(inner["content"], MAX_TOOL_OUTPUT_CHARS)
    return _truncate(outputs, MAX_TOOL_OUTPUT_CHARS)


def _filter_for(skill_version: int, phase: str) -> str:
    """A LangSmith run filter selecting one skill version's sessions.

    ``has(metadata, ...)`` matches the whole JSON object at once, which keeps the
    integer ``skill_version`` typed. The alternative ``metadata_key``/
    ``metadata_value`` pair form stringifies values and reads ambiguously when
    filtering on more than one key.

    When ``config.RUN_ID`` is set it joins the selector, which is what stops a second
    loop invocation from mining the first one's sessions: without it, a v1/train
    harvest matches every v1/train session ever recorded in the project, so evidence
    counts accumulate across runs and the ≥2-session gate gets easier to clear each
    time. ``RUN_ID`` is allowlisted at import in ``config``; it reaches a
    single-quoted filter expression here, so it must not be interpolated unchecked.
    """
    selector: dict = {"phase": phase, "skill_version": int(skill_version)}
    if config.RUN_ID:
        selector["run_id"] = config.RUN_ID
    return f"has(metadata, '{json.dumps(selector)}')"


MAX_RUNS_PER_QUERY = 100  # server-side cap on /runs/query


def fetch_turn_runs(skill_version: int, phase: str = "train", limit: int = MAX_RUNS_PER_QUERY) -> list:
    """All turn-root runs for one skill version and phase."""
    runs = list(
        _client.list_runs(
            project_name=config.PROJECT,
            is_root=True,
            filter=_filter_for(skill_version, phase),
            limit=min(limit, MAX_RUNS_PER_QUERY),
        )
    )
    return sorted(runs, key=lambda r: (_metadata(r).get("session_id", ""), _metadata(r).get("turn_index", 0)))


def wait_for_traces(
    skill_version: int,
    expected_sessions: int,
    phase: str = "train",
    timeout_s: int = 90,
    poll_s: int = 3,
) -> list:
    """Poll until every expected session is queryable, or give up and warn.

    Trace ingestion is asynchronous, so harvesting immediately after a run can see
    a partial picture.
    """
    deadline = time.time() + timeout_s
    runs: list = []
    while time.time() < deadline:
        runs = fetch_turn_runs(skill_version, phase)
        sessions = {_metadata(r).get("session_id") for r in runs}
        sessions.discard(None)
        if len(sessions) >= expected_sessions:
            return runs
        time.sleep(poll_s)
    print(
        f"  ! ingestion timeout: found {len({_metadata(r).get('session_id') for r in runs})}"
        f"/{expected_sessions} sessions after {timeout_s}s; digesting what is available"
    )
    return runs


def _tool_calls_for_trace(trace_id) -> list[dict]:
    """The ordered tool calls inside one turn, read from the trace's child runs."""
    tool_runs = list(_client.list_runs(project_name=config.PROJECT, trace_id=trace_id, run_type="tool"))
    tool_runs.sort(key=lambda r: (r.start_time or datetime.min.replace(tzinfo=timezone.utc)))
    calls = []
    for run in tool_runs:
        calls.append(
            {
                "tool": run.name,
                "args": _truncate(run.inputs, 160),
                "output": _tool_output_text(run.outputs),
                "error": bool(run.error),
            }
        )
    return calls


def _feedback_for(run_ids: list[str]) -> dict[str, dict]:
    """Map run_id -> {"scores": {...}, "signals": [...]}."""
    out: dict[str, dict] = defaultdict(lambda: {"scores": {}, "signals": []})
    if not run_ids:
        return out
    for item in _client.list_feedback(run_ids=run_ids):
        entry = out[str(item.run_id)]
        if item.key == _SIGNAL_KEY:
            if item.comment:
                entry["signals"].append(item.comment)
        elif item.score is not None:
            entry["scores"][item.key] = float(item.score)
    return out


def build_digest(skill_version: int, phase: str = "train", expected_sessions: int | None = None) -> dict:
    """Reduce LangSmith history for one skill version into an optimizer-ready digest."""
    runs = (
        wait_for_traces(skill_version, expected_sessions, phase)
        if expected_sessions
        else fetch_turn_runs(skill_version, phase)
    )
    if not runs:
        raise RuntimeError(
            f"no traces found in project {config.PROJECT!r} for skill v{skill_version} "
            f"phase={phase!r}. Run the sessions step first."
        )

    feedback = _feedback_for([str(r.id) for r in runs])

    sessions: dict[str, dict] = {}
    for run in runs:
        md = _metadata(run)
        session_id = md.get("session_id") or str(run.id)
        session = sessions.setdefault(
            session_id,
            {"session_id": session_id, "scenario_id": md.get("scenario_id"), "turns": [], "scores": {}},
        )
        fb = feedback.get(str(run.id), {"scores": {}, "signals": []})

        outputs = run.outputs or {}
        session["turns"].append(
            {
                "turn_index": md.get("turn_index", len(session["turns"])),
                "customer_said": _truncate((run.inputs or {}).get("user_message"), 300),
                "tool_calls": _tool_calls_for_trace(run.trace_id),
                "agent_resolution": outputs.get("resolution"),
                "agent_reply": _truncate(outputs.get("customer_message"), MAX_REPLY_CHARS),
                "customer_pushed_back_about": fb["signals"][0] if fb["signals"] else None,
            }
        )
        # Session-level scores are attached to the final turn.
        if fb["scores"]:
            session["scores"] = fb["scores"]

    for session in sessions.values():
        session["turns"].sort(key=lambda t: t["turn_index"])
        session["turns_used"] = len(session["turns"])
        session["tool_sequence"] = [c["tool"] for t in session["turns"] for c in t["tool_calls"]]
        repeats = Counter(session["tool_sequence"])
        session["repeated_tools"] = {k: v for k, v in repeats.items() if v > 1}

    session_list = sorted(sessions.values(), key=lambda s: s["session_id"])

    # ---- cross-session aggregation: which failures recur, and in how many sessions
    signal_sessions: dict[str, list[str]] = defaultdict(list)
    for session in session_list:
        for turn in session["turns"]:
            signal = turn["customer_pushed_back_about"]
            if signal and session["session_id"] not in signal_sessions[signal]:
                signal_sessions[signal].append(session["session_id"])

    metric_means: dict[str, float] = {}
    scored = [s for s in session_list if s["scores"]]
    if scored:
        for key in sorted({k for s in scored for k in s["scores"]}):
            values = [s["scores"][key] for s in scored if key in s["scores"]]
            metric_means[key] = round(sum(values) / len(values), 3)

    digest = {
        "skill_version": skill_version,
        "phase": phase,
        "harvested_at": datetime.now(timezone.utc).isoformat(),
        "langsmith_project": config.PROJECT,
        "session_count": len(session_list),
        "aggregate": {
            "metric_means": metric_means,
            "mean_turns_used": round(
                sum(s["turns_used"] for s in session_list) / len(session_list), 2
            ),
            # The cross-session evidence table the optimizer is required to cite from.
            "friction_signal_sessions": {k: v for k, v in sorted(signal_sessions.items())},
            "friction_signal_counts": {k: len(v) for k, v in sorted(signal_sessions.items())},
            "repeated_tool_totals": dict(
                Counter(
                    tool
                    for s in session_list
                    for tool, count in s["repeated_tools"].items()
                    for _ in range(count - 1)
                )
            ),
        },
        "sessions": session_list,
    }

    path = config.artifact_file(f"digest-v{skill_version}-{phase}.json")
    path.write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(
        f"  digested {len(session_list)} sessions from LangSmith -> {path.relative_to(config.REPO_ROOT)}"
    )
    return digest
