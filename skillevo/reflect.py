"""Turn a multi-session digest into the next version of SKILL.md.

The controlling idea: **the model proposes, code decides.**

The optimizer LLM never writes the skill file. It emits a structured list of
candidate rules, each of which must cite the sessions that justify it. Code then:

1. **verifies the citations exist** -- a rule citing a session id that is not in the
   observed evidence registry is dropped as hallucinated;
2. **enforces the cross-session gate** -- a rule corroborated by fewer than
   ``MIN_EVIDENCE_SESSIONS`` distinct sessions is dropped. This is the mechanism
   that makes the technique depend on *multi-session* history rather than a single
   anecdote, and it is code, not a request in a prompt;
3. **rejects memorization** -- rule text naming a specific transaction or account is
   dropped, so the skill cannot encode the answer key for individual scenarios;
4. **sanitizes** the surviving text and **renders the file from a fixed template**.

Because the document is rendered rather than transcribed, untrusted content reaching
the optimizer (harvested traces) cannot restructure the skill file, escape the
``<skill>`` block, or address the harness.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from skillevo import config
from skillevo.agent import load_skill

Section = Literal["procedure", "output_requirements", "edge_cases"]

# Rule text must generalize. Naming a concrete transaction or account is how an
# optimizer would smuggle the answer key into the skill and fake an improvement.
_OVERFIT_PATTERNS = [
    re.compile(r"\bTXN-[A-Z0-9]+\b", re.IGNORECASE),
    re.compile(r"\bACC-\d+\b", re.IGNORECASE),
]

SECTION_TITLES = {
    "procedure": "Procedure",
    "output_requirements": "Every reply must include",
    "edge_cases": "Precedence and edge cases",
}


class ProposedRule(BaseModel):
    title: str = Field(description="Short imperative label, at most 8 words.")
    guidance: str = Field(
        description=(
            "One to three sentences of general procedural guidance. Must not name any "
            "specific transaction id, account id, merchant or customer."
        )
    )
    section: Section = Field(description="Where this rule belongs in the skill document.")
    evidence_session_ids: list[str] = Field(
        description=(
            "Session ids from the supplied history that demonstrate the need for this "
            "rule. Cite every session that supports it."
        )
    )
    failure_mode: str = Field(description="What went wrong in those sessions, in one sentence.")
    target_metric: str = Field(description="The metric this rule is expected to move.")


class SkillProposal(BaseModel):
    rules: list[ProposedRule] = Field(description="The complete rule set for the next skill version.")
    summary: str = Field(description="Two or three sentences on what changed and why.")


REFLECT_PROMPT = """\
You are improving the operating instructions ("skill") for a payment dispute triage
agent, using the agent's own conversation history across multiple sessions.

Here is the skill the agent was running:
<current_skill>
{current_skill}
</current_skill>

Here is aggregated evidence from {session_count} traced sessions under that skill.
`friction_signal_sessions` maps each thing customers had to push back about to the
sessions where it happened. Lower metric means are worse; 1.0 is perfect.
<aggregate>
{aggregate}
</aggregate>

Here are the session transcripts, including every tool call and its result:
<sessions>
{sessions}
</sessions>

Produce the complete rule set for the next version of the skill.

Hard requirements:
- Every rule must cite, in `evidence_session_ids`, at least {min_evidence} DISTINCT
  session ids drawn from the history above. Rules citing fewer will be discarded.
  Do not cite a session id that does not appear above.
- Rules must generalize. Never name a specific transaction id, account id, merchant
  or customer -- such rules will be discarded. Describe the *condition*, not the case.
- Keep each `guidance` under {max_rule_chars} characters. The skill is injected into
  the agent's prompt on every turn, so the budget is enforced and longer rules are
  discarded outright. Split a long rule into two rather than losing it.
- Carry forward any rule from the current skill that the evidence still supports,
  re-citing its evidence. Drop rules the evidence does not support.
- Prefer a small number of decisive rules over many weak ones. Do not restate the
  output contract the agent already has; add only what the evidence shows is missing.
- You may consolidate knowledge you observed in tool results (for example policy
  rules and their codes) so the agent does not have to rediscover it, and so it
  knows which policy governs which condition.
"""


def _load_registry() -> dict:
    path = config.artifact_file("evidence_registry.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"sessions": []}


def _update_registry(session_ids: list[str]) -> list[str]:
    """Accumulate every session id ever observed, so rules learned in earlier
    iterations can be carried forward by re-citing their original evidence."""
    registry = _load_registry()
    known = set(registry["sessions"]) | set(session_ids)
    registry["sessions"] = sorted(known)
    config.artifact_file("evidence_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )
    return registry["sessions"]


def _compact_sessions(digest: dict) -> str:
    """Render sessions for the prompt without the ground-truth labels."""
    lines = []
    for session in digest["sessions"]:
        lines.append(f"### session {session['session_id']} ({session['turns_used']} turn(s))")
        if session.get("scores"):
            lines.append(f"scores: {json.dumps(session['scores'], sort_keys=True)}")
        for turn in session["turns"]:
            lines.append(f"- turn {turn['turn_index']} | customer: {turn['customer_said']}")
            for call in turn["tool_calls"]:
                lines.append(f"    tool {call['tool']}({call['args']}) -> {call['output']}")
            lines.append(f"    agent resolution: {turn['agent_resolution']}")
            lines.append(f"    agent reply: {turn['agent_reply']}")
            if turn["customer_pushed_back_about"]:
                lines.append(f"    !! customer pushed back about: {turn['customer_pushed_back_about']}")
        lines.append("")
    return "\n".join(lines)


def propose(digest: dict) -> SkillProposal:
    """Ask the optimizer for a candidate rule set."""
    current_skill, _ = load_skill()
    model = init_chat_model(config.REFLECT_MODEL).with_structured_output(SkillProposal)
    prompt = REFLECT_PROMPT.format(
        current_skill=current_skill,
        session_count=digest["session_count"],
        aggregate=json.dumps(digest["aggregate"], indent=2),
        sessions=_compact_sessions(digest),
        min_evidence=config.MIN_EVIDENCE_SESSIONS,
        max_rule_chars=config.MAX_RULE_CHARS,
    )
    return model.invoke(prompt)


def screen(proposal: SkillProposal, known_sessions: list[str]) -> tuple[list[dict], list[dict]]:
    """Apply the evidence gate and the safety checks. Returns ``(accepted, rejected)``."""
    known = set(known_sessions)
    accepted: list[dict] = []
    rejected: list[dict] = []

    for rule in proposal.rules:
        cited = [s for s in dict.fromkeys(rule.evidence_session_ids)]
        verified = [s for s in cited if s in known]
        hallucinated = [s for s in cited if s not in known]

        record = {
            "title": rule.title,
            "section": rule.section,
            "guidance": rule.guidance,
            "failure_mode": rule.failure_mode,
            "target_metric": rule.target_metric,
            "evidence_cited": cited,
            "evidence_verified": verified,
            "evidence_hallucinated": hallucinated,
        }

        clean_title, title_reason = config.sanitize_rule_text(rule.title)
        clean_guidance, guidance_reason = config.sanitize_rule_text(rule.guidance)
        overfit = next(
            (p.pattern for p in _OVERFIT_PATTERNS if p.search(f"{rule.title} {rule.guidance}")),
            None,
        )

        if title_reason or guidance_reason:
            record["rejected_because"] = f"unsafe text: {title_reason or guidance_reason}"
        elif overfit:
            record["rejected_because"] = f"scenario-specific reference matching /{overfit}/"
        elif len(verified) < config.MIN_EVIDENCE_SESSIONS:
            record["rejected_because"] = (
                f"only {len(verified)} verified evidence session(s), "
                f"need {config.MIN_EVIDENCE_SESSIONS}"
                + (f"; unknown ids cited: {hallucinated}" if hallucinated else "")
            )
        else:
            record["title"] = clean_title
            record["guidance"] = clean_guidance
            accepted.append(record)
            continue

        rejected.append(record)

    return accepted, rejected


def render_skill(accepted: list[dict], version: int, digest: dict) -> str:
    """Build SKILL.md from accepted rules using a fixed template.

    Provenance is intentionally kept out of this file -- it lives in the sidecar
    rationale JSON. The skill is loaded into a system prompt on every turn, so it
    carries only what the agent needs to act on.
    """
    lines = [
        "---",
        f"name: {config.SKILL_NAME}",
        f"version: {version}",
        "source: evolved from LangSmith trace history",
        f"evolved_at: {datetime.now(timezone.utc).isoformat()}",
        f"derived_from: digest-v{digest['skill_version']}-{digest['phase']}.json",
        f"evidence_sessions: {digest['session_count']}",
        f"audit_trail: skills/{config.SKILL_NAME}/history/v{version}.rationale.json",
        "---",
        "",
        "# Payment Dispute Triage",
        "",
    ]

    for section in ("procedure", "output_requirements", "edge_cases"):
        rules = [r for r in accepted if r["section"] == section]
        if not rules:
            continue
        lines += [f"## {SECTION_TITLES[section]}", ""]
        for index, rule in enumerate(rules, start=1):
            bullet = f"{index}." if section == "procedure" else "-"
            lines.append(f"{bullet} **{rule['title']}** {rule['guidance']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def evolve(digest: dict) -> dict:
    """Full reflect step: propose, screen, render, archive, write."""
    _, current_version = load_skill()
    next_version = current_version + 1

    known_sessions = _update_registry([s["session_id"] for s in digest["sessions"]])
    proposal = propose(digest)
    accepted, rejected = screen(proposal, known_sessions)

    print(f"  proposed {len(proposal.rules)} rule(s): {len(accepted)} accepted, {len(rejected)} rejected")
    for rule in rejected:
        print(f"    - dropped {rule['title']!r}: {rule['rejected_because']}")

    if not accepted:
        raise RuntimeError(
            "no proposed rule survived screening; skill left unchanged. "
            "Inspect the rationale artifact for why."
        )

    # Archive the outgoing version before overwriting.
    skill_path = config.skill_file()
    archive = config.skill_history_file(current_version)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copy2(skill_path, archive)

    markdown = render_skill(accepted, next_version, digest)
    skill_path.write_text(markdown, encoding="utf-8")
    config.skill_history_file(next_version).write_text(markdown, encoding="utf-8")

    rationale = {
        "version": next_version,
        "previous_version": current_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_model": config.REFLECT_MODEL,
        "derived_from_digest": f"digest-v{digest['skill_version']}-{digest['phase']}.json",
        "min_evidence_sessions": config.MIN_EVIDENCE_SESSIONS,
        "summary": proposal.summary,
        "accepted_rules": accepted,
        "rejected_rules": rejected,
    }
    rationale_path = config.safe_path(
        config.SKILLS_DIR, config.SKILL_NAME, "history", f"v{next_version}.rationale.json"
    )
    rationale_path.write_text(json.dumps(rationale, indent=2), encoding="utf-8")

    print(f"  wrote skill v{next_version} ({len(markdown)} chars) + audit trail")
    return rationale
