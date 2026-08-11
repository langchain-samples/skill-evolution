"""Command line entry point: ``python -m skillevo <command>``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from skillevo import config
from skillevo.agent import load_skill
from skillevo.evaluate import as_score, floor_breaches, run_experiment, summarize, upsert_dataset
from skillevo.harvest import build_digest
from skillevo.reflect import evolve
from skillevo.scenarios import HOLDOUT, TRAIN
from skillevo.sessions import run_split

BASELINE_SKILL = """\
---
name: payment-dispute-triage
version: 1
source: hand-written baseline
---

# Payment Dispute Triage

Help the customer resolve their payment dispute. Look up whatever you need with
the tools available to you, then reply with a clear explanation and a recommended
resolution.
"""


def _scoreboard_path():
    return config.artifact_file("scoreboard.json")


def _load_scoreboard() -> dict:
    path = _scoreboard_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"entries": []}


def _save_scoreboard(board: dict) -> None:
    _scoreboard_path().write_text(json.dumps(board, indent=2), encoding="utf-8")


def _record(board: dict, version: int, metrics: dict, kept: bool | None = None) -> None:
    board["entries"] = [e for e in board["entries"] if e["version"] != version]
    board["entries"].append({"version": version, "metrics": metrics, "kept": kept})
    board["entries"].sort(key=lambda e: e["version"])
    _save_scoreboard(board)


def _ranked_entries(board: dict) -> list[dict]:
    """Scoreboard entries usable as an incumbent, best composite first.

    The scoreboard is a JSON file on disk and its numbers decide promotions, so an
    entry whose composite is missing or non-numeric is dropped rather than compared.
    Otherwise a corrupt row could be selected as the incumbent and lower the bar.
    """
    usable = [
        e for e in board.get("entries", [])
        if as_score(e.get("metrics", {}).get("composite")) is not None
    ]
    return sorted(usable, key=lambda e: e["metrics"]["composite"], reverse=True)


def _print_scoreboard(board: dict) -> None:
    entries = board["entries"]
    if not entries:
        print("no results yet")
        return
    all_keys = {k for entry in entries for k in entry["metrics"]}
    keys = ["composite"] + sorted(all_keys - {"composite"})
    width = max(len(k) for k in keys) + 2
    print(f"\n{'metric':<{width}}" + "".join(f"v{e['version']:<8}" for e in entries))
    print("-" * (width + 9 * len(entries)))
    for key in keys:
        row = f"{key:<{width}}"
        for entry in entries:
            value = entry["metrics"].get(key)
            row += f"{value if value is not None else '-':<9}"
        print(row)
    verdicts = [f"v{e['version']}: {'kept' if e['kept'] else 'reverted'}" for e in entries if e["kept"] is not None]
    if verdicts:
        print("\ngate: " + " | ".join(verdicts))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_reset(args) -> None:
    """Restore the hand-written v1 skill and clear local artifacts."""
    config.skill_file().write_text(BASELINE_SKILL, encoding="utf-8")
    history = config.safe_path(config.SKILLS_DIR, config.SKILL_NAME, "history")
    if args.hard and history.exists():
        shutil.rmtree(history)
        history.mkdir(parents=True, exist_ok=True)
    if args.hard and config.ARTIFACTS_DIR.exists():
        shutil.rmtree(config.ARTIFACTS_DIR)
    print(f"skill reset to v1{' (history and artifacts cleared)' if args.hard else ''}")


def cmd_sessions(args) -> None:
    _, version = load_skill()
    print(f"running {len(TRAIN)} training sessions under skill v{version} -> project {config.PROJECT!r}")
    run_split(TRAIN, phase="train")


def cmd_harvest(args) -> None:
    _, version = load_skill()
    version = args.version if args.version is not None else version
    print(f"harvesting LangSmith history for skill v{version}")
    digest = build_digest(version, phase="train", expected_sessions=len(TRAIN))
    print(json.dumps(digest["aggregate"], indent=2))


def cmd_reflect(args) -> None:
    _, version = load_skill()
    version = args.version if args.version is not None else version
    digest = build_digest(version, phase="train")
    print(f"reflecting over {digest['session_count']} sessions with {config.REFLECT_MODEL}")
    rationale = evolve(digest)
    print("\n" + rationale["summary"])


def cmd_evaluate(args) -> None:
    _, version = load_skill()
    print(f"evaluating skill v{version} on {len(HOLDOUT)} holdout scenarios")
    metrics = summarize(run_experiment())
    board = _load_scoreboard()
    _record(board, version, metrics)
    print(json.dumps(metrics, indent=2))
    _print_scoreboard(board)


def cmd_dataset(args) -> None:
    upsert_dataset(force=args.force)


def cmd_status(args) -> None:
    body, version = load_skill()
    print(f"skill: {config.SKILL_NAME} v{version} ({len(body)} chars)")
    print(f"project: {config.PROJECT} | agent: {config.AGENT_MODEL} | optimizer: {config.REFLECT_MODEL}")
    _print_scoreboard(_load_scoreboard())


def cmd_loop(args) -> None:
    """The full test case: measure, learn, re-measure, keep only what improves."""
    board = _load_scoreboard()
    _, version = load_skill()

    known = {e["version"]: e for e in board["entries"]}
    if version not in known:
        print(f"\n=== baseline: evaluating skill v{version} on holdout ===")
        baseline = summarize(run_experiment())
        _record(board, version, baseline, kept=True)
        print(json.dumps(baseline, indent=2))
    # Best score seen so far, not the most recent -- a previously reverted candidate
    # sitting at the end of the scoreboard must not lower the bar. The whole entry is
    # kept, not just the composite: the floor check needs the incumbent's per-metric
    # levels to measure a regression against.
    ranked = _ranked_entries(board)
    if not ranked:
        raise RuntimeError(
            "scoreboard has no entry with a usable composite; run `evaluate` first"
        )
    incumbent = ranked[0]
    best = float(incumbent["metrics"]["composite"])

    for iteration in range(1, args.iterations + 1):
        _, version = load_skill()
        print(f"\n=== iteration {iteration}/{args.iterations}: skill v{version} ===")

        print(f"\n-- 1. run {len(TRAIN)} training sessions (traced to LangSmith)")
        run_split(TRAIN, phase="train")

        print("\n-- 2. harvest multi-session history back from LangSmith")
        digest = build_digest(version, phase="train", expected_sessions=len(TRAIN))
        print(f"   friction across sessions: {digest['aggregate']['friction_signal_counts']}")

        print(f"\n-- 3. reflect and propose skill v{version + 1}")
        try:
            evolve(digest)
        except RuntimeError as exc:
            print(f"   ! {exc}")
            break

        _, new_version = load_skill()
        print(f"\n-- 4. evaluate v{new_version} on the holdout set")
        metrics = summarize(run_experiment())
        # Two independent conditions. The composite must clear the incumbent by a
        # margin, not merely exceed it: on this scenario count a strict `>` is
        # decidable by one scenario flipping once. And no floor metric may regress at
        # all -- that one is not tradeable against the aggregate.
        threshold = best + config.GATE_MARGIN
        breaches = floor_breaches(metrics, incumbent["metrics"])
        improved = metrics["composite"] > threshold and not breaches
        _record(board, new_version, metrics, kept=improved)

        delta = (
            f"composite {best:.3f} -> {metrics['composite']:.3f} "
            f"(needed > {threshold:.3f})"
        )
        if improved:
            print(f"   {delta}: KEPT v{new_version}")
            incumbent = {"version": new_version, "metrics": metrics}
            best = metrics["composite"]
        else:
            previous = config.skill_history_file(new_version - 1)
            shutil.copy2(previous, config.skill_file())
            reason = (
                "floor breached [" + "; ".join(breaches) + "]"
                if breaches
                else "no improvement"
            )
            print(f"   {delta}: {reason}, REVERTED to v{new_version - 1}")
            if args.stop_on_regression:
                break

    _print_scoreboard(board)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="skillevo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reset", help="restore the v1 baseline skill")
    p.add_argument("--hard", action="store_true", help="also delete history and artifacts")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("sessions", help="run the training sessions, traced to LangSmith")
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("harvest", help="reduce LangSmith history to a digest")
    p.add_argument("--version", type=int, default=None)
    p.set_defaults(func=cmd_harvest)

    p = sub.add_parser("reflect", help="propose and write the next skill version")
    p.add_argument("--version", type=int, default=None)
    p.set_defaults(func=cmd_reflect)

    p = sub.add_parser("evaluate", help="score the current skill on the holdout dataset")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("dataset", help="create or refresh the holdout dataset")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("status", help="show the current skill version and scoreboard")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("loop", help="run the full evolve-and-gate loop")
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--stop-on-regression", action="store_true")
    p.set_defaults(func=cmd_loop)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
