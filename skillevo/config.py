"""Configuration, path confinement, and untrusted-content sanitization."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")

SKILLS_DIR = REPO_ROOT / "skills"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

SKILL_NAME = "payment-dispute-triage"

# Tracing is the whole point of the test case, so default it on.
os.environ.setdefault("LANGSMITH_TRACING", "true")
PROJECT = os.environ.setdefault("LANGSMITH_PROJECT", "skill-evolution")

# The agent under test is deliberately a small, cheap model. That is the realistic
# setting for this technique: a well-evolved skill is what makes a cheap model
# reliable enough to deploy. A frontier model already scores near ceiling on this
# task from the thin baseline skill, which would leave nothing to measure.
AGENT_MODEL = os.environ.get("SKILLEVO_AGENT_MODEL", "anthropic:claude-haiku-4-5-20251001")
REFLECT_MODEL = os.environ.get("SKILLEVO_REFLECT_MODEL", "anthropic:claude-opus-5")
JUDGE_MODEL = os.environ.get("SKILLEVO_JUDGE_MODEL", "anthropic:claude-sonnet-5")

DATASET_NAME = "payment-dispute-triage-holdout"

# A proposed rule must be corroborated by at least this many distinct sessions
# before it is allowed into the skill. Enforced in code, not left to the model.
MIN_EVIDENCE_SESSIONS = 2

MAX_TURNS_PER_SESSION = 3  # opening message + up to 2 simulated follow-ups


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def _clamped_int(name: str, default: int, low: int, high: int) -> int:
    """Read an int from the environment, clamped to a sane range.

    Clamped rather than trusted: these values size loops and API concurrency, so a
    typo or a stray value in a shell profile should not be able to fire thousands of
    requests at LangSmith and the model provider.
    """
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        return default
    return max(low, min(high, value))


def _clamped_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return max(low, min(high, value))


# The agent under test is pinned to greedy decoding. Without this, a holdout mean
# over 5 examples is a single sample of a sampled model, and the keep/revert verdict
# turns on run-to-run noise rather than on the skill. The *optimizer* is left
# sampled on purpose -- proposal diversity is what the screen is there to filter.
AGENT_TEMPERATURE = _clamped_float("SKILLEVO_AGENT_TEMPERATURE", 0.0, 0.0, 1.0)

# Claude Opus 5, Sonnet 5, Opus 4.8/4.7 and Fable 5 removed the sampling parameters:
# sending `temperature` returns 400 "`temperature` is deprecated for this model."
# Haiku 4.5 and the 4.5/4.6 families still accept it. There is no greedy-decoding
# substitute on the newer models -- `effort` and prompting are the only levers -- so
# a run whose agent is one of those cannot be pinned, and its holdout scores keep
# sampling variance no matter how many repetitions are used.
_SAMPLING_REMOVED = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
)


def sampling_kwargs(model: str) -> dict:
    """Decoding kwargs for ``model``, omitting what the model would reject."""
    if any(tag in model for tag in _SAMPLING_REMOVED):
        return {}
    return {"temperature": AGENT_TEMPERATURE}


def pins_temperature(model: str) -> bool:
    return "temperature" in sampling_kwargs(model)

# Each holdout example is scored this many times per experiment. Composite
# granularity is (1 / n_examples) / n_headline_metrics, so with 5 examples and 6
# metrics a single binary flip moves the composite by 0.033 -- the same order as the
# deltas being measured. Repetitions are what make the mean mean something.
EVAL_REPETITIONS = _clamped_int("SKILLEVO_EVAL_REPETITIONS", 3, 1, 20)

EVAL_CONCURRENCY = _clamped_int("SKILLEVO_EVAL_CONCURRENCY", 4, 1, 16)

# A candidate must beat the best composite so far by more than this to be kept.
# A strict `>` makes the gate decidable by one scenario flipping.
GATE_MARGIN = _clamped_float("SKILLEVO_GATE_MARGIN", 0.02, 0.0, 1.0)

# Optional tag isolating one loop invocation's traces from every earlier one. The
# harvester filters on it when set, so re-running the loop does not mine the
# previous run's sessions and inflate its own evidence counts.
#
# Interpolated into a LangSmith filter expression delimited by single quotes, so it
# is allowlisted here at import rather than escaped at the call site.
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

RUN_ID = os.environ.get("SKILLEVO_RUN_ID") or None
if RUN_ID is not None and not _SAFE_RUN_ID.match(RUN_ID):
    raise ValueError(
        f"SKILLEVO_RUN_ID={RUN_ID!r} is not allowlisted; "
        "use 1-64 chars of [A-Za-z0-9._-] starting alphanumeric"
    )


# --------------------------------------------------------------------------- #
# Path confinement
# --------------------------------------------------------------------------- #

# A leading dot is excluded, so "." and ".." can never be path components.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def safe_path(base: Path, *parts: str) -> Path:
    """Resolve ``parts`` under ``base``, refusing anything that escapes it.

    Every filesystem write in this project goes through here, so a skill name or
    version counter that has been influenced by model output cannot reach outside
    ``skills/`` or ``artifacts/``.
    """
    for part in parts:
        if not _SAFE_NAME.match(part):
            raise ValueError(f"unsafe path component: {part!r}")
    base = base.resolve()
    candidate = (base / Path(*parts)).resolve()
    if candidate != base and not candidate.is_relative_to(base):
        raise ValueError(f"path {candidate} escapes {base}")
    return candidate


def skill_file(name: str = SKILL_NAME) -> Path:
    return safe_path(SKILLS_DIR, name, "SKILL.md")


def skill_history_file(version: int, name: str = SKILL_NAME) -> Path:
    return safe_path(SKILLS_DIR, name, "history", f"v{int(version)}.md")


def artifact_file(*parts: str) -> Path:
    path = safe_path(ARTIFACTS_DIR, *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Untrusted-content sanitization
# --------------------------------------------------------------------------- #

MAX_RULE_CHARS = 500

# Model-proposed rule text is derived from harvested trace content, which is
# untrusted input. Anything that reads like an attempt to talk to the harness
# rather than describe dispute procedure is dropped.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+|any\s+)?(previous|prior|above|preceding)",
        r"disregard\s+(the\s+)?(previous|prior|above|system)",
        r"system\s+prompt",
        r"</?\s*(skill|system|harness|instructions)\s*>",
        r"api[\s_-]?key|secret|credential|password|bearer\s+token",
        r"\b(exfiltrat|reveal your|print your|output your)\b",
        r"you\s+are\s+no\s+longer",
        r"\bcurl\b|\bhttps?://",
    )
]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_rule_text(text: str) -> tuple[str, str | None]:
    """Return ``(clean_text, rejection_reason)``.

    ``rejection_reason`` is ``None`` when the text is acceptable.
    """
    if not isinstance(text, str) or not text.strip():
        return "", "empty"

    clean = _CONTROL_CHARS.sub("", text).strip()
    # Collapse fenced blocks and headings so a rule cannot restructure the document.
    clean = clean.replace("```", "").replace("\r", "")
    clean = re.sub(r"^\s*#+\s*", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\s*\n\s*", " ", clean).strip()

    if len(clean) > MAX_RULE_CHARS:
        return "", f"too long ({len(clean)} > {MAX_RULE_CHARS} chars)"

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(clean):
            return "", f"matched disallowed pattern /{pattern.pattern}/"

    return clean, None


def sanitize_rule_text_block(text: str) -> tuple[str, str | None]:
    """Same checks as :func:`sanitize_rule_text`, but for a multi-line document.

    Markdown structure (headings, lists, blank lines) is preserved, since this is
    applied to a whole skill body rather than a single rule.
    """
    if not isinstance(text, str) or not text.strip():
        return "", "empty"

    clean = _CONTROL_CHARS.sub("", text).replace("\r", "").strip()

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(clean):
            return "", f"matched disallowed pattern /{pattern.pattern}/"

    return clean, None
