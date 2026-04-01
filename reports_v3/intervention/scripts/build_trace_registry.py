"""Build trace registry from baseline run outputs.

After running baseline (no intervention) for each (env, agent), this script
scans the baseline run directories for agent_logs, matches traces to trial
outcomes from the report JSON, and picks one success + one failure trace
per task.

If multiple success/failure traces exist, one is picked randomly (seeded).

Outputs trace_registry.json used by intervention runs.

Usage:
    uv run python scripts/build_trace_registry.py
"""

import json
import random
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ENVIRONMENTS, INTERVENTION_ROOT

SEED = 42


def find_baseline_report(baseline_dir: Path) -> Path | None:
    """Find the *_report.json in a baseline run directory."""
    reports = list(baseline_dir.glob("*_report.json"))
    if not reports:
        return None
    return reports[0]


def find_trace_dir(baseline_dir: Path) -> Path | None:
    """Find the agent_logs-* directory in a baseline run directory."""
    log_dirs = list(baseline_dir.glob("agent_logs-*"))
    if not log_dirs:
        return None
    return log_dirs[0]


def get_trace_files_for_task(trace_dir: Path, task_id: str) -> list[Path]:
    """Get all trace files for a task, sorted chronologically."""
    return sorted(trace_dir.glob(f"{task_id}_*.json"))


def count_assistant_steps(trace_path: Path) -> int:
    """Count assistant messages in a trace file."""
    with Path(trace_path).open() as f:
        trace = json.load(f)
    return sum(1 for m in trace.get("messages", []) if m.get("role") == "assistant")


def build_registry_for_baseline(
    env: str, agent: str, baseline_dir: Path, rng: random.Random
) -> dict:
    """Build registry entries from one baseline run directory.

    Returns dict of task_id -> entry, only for tasks with mixed results.
    """
    report_path = find_baseline_report(baseline_dir)
    if report_path is None:
        logger.info(f"    WARNING: No report found in {baseline_dir}")
        return {}

    trace_dir = find_trace_dir(baseline_dir)
    if trace_dir is None:
        logger.info(f"    WARNING: No agent_logs dir found in {baseline_dir}")
        return {}

    with report_path.open() as f:
        report = json.load(f)

    entries = {}
    for task_id, task_data in report.get("task_results", {}).items():
        trials = task_data["trials"]
        trace_files = get_trace_files_for_task(trace_dir, task_id)

        if len(trace_files) < len(trials):
            logger.info(
                f"    WARNING: {task_id}: {len(trace_files)} traces "
                f"< {len(trials)} trials"
            )

        # Match traces to trials (both chronological)
        matched = [
            {
                "path": str(trace_path),
                "success": trial["success"],
                "score": trial["score"],
            }
            for trace_path, trial in zip(trace_files, trials, strict=False)
        ]

        successes = [m for m in matched if m["success"]]
        failures = [m for m in matched if not m["success"]]

        if not successes or not failures:
            sr = task_data["success_rate"]
            label = "ALL_SUCCESS" if sr == 1 else "ALL_FAIL"
            logger.info(
                f"    {task_id}: {label} ({sr:.0%}) -> SKIPPED (no mixed results)"
            )
            continue

        # Pick one randomly from each pool
        success_trace = rng.choice(successes)
        failed_trace = rng.choice(failures)

        success_steps = count_assistant_steps(success_trace["path"])
        failed_steps = count_assistant_steps(failed_trace["path"])

        entries[task_id] = {
            "environment": env,
            "agent_type": agent,
            "task_id": task_id,
            "success_trace": success_trace["path"],
            "failed_trace": failed_trace["path"],
            "success_trace_steps": success_steps,
            "failed_trace_steps": failed_steps,
            "success_rate": task_data["success_rate"],
            "n_success_traces": len(successes),
            "n_failed_traces": len(failures),
        }
        logger.info(
            f"    {task_id}: OK "
            f"(picked 1/{len(successes)} success, 1/{len(failures)} failed, "
            f"steps={success_steps}/{failed_steps}, "
            f"baseline_sr={task_data['success_rate']:.0%})"
        )

    return entries


def main():
    runs_dir = INTERVENTION_ROOT / "runs"
    rng = random.Random(SEED)
    registry = {}

    for env_name in ENVIRONMENTS:
        logger.info(f"\n=== {env_name.upper()} ===")

        for agent_type in ["react", "toolcalling"]:
            baseline_dir = runs_dir / env_name / agent_type / "baseline"
            logger.info(f"\n  Agent: {agent_type}")
            logger.info(f"    Baseline dir: {baseline_dir}")

            if not baseline_dir.exists():
                logger.info("    NOT FOUND — run baseline first")
                continue

            entries = build_registry_for_baseline(
                env_name, agent_type, baseline_dir, rng
            )

            for task_id, entry in entries.items():
                registry_key = f"{env_name}/{agent_type}/{task_id}"
                registry[registry_key] = entry

    # Write registry
    output_path = INTERVENTION_ROOT / "trace_registry.json"
    with output_path.open("w") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"\n\nTrace registry written to {output_path}")
    logger.info(f"Total entries: {len(registry)}")

    for env in ENVIRONMENTS:
        entries = [v for v in registry.values() if v["environment"] == env]
        react_entries = [e for e in entries if e["agent_type"] == "react"]
        tc_entries = [e for e in entries if e["agent_type"] == "toolcalling"]
        logger.info(
            f"  {env}: {len(react_entries)} react, {len(tc_entries)} toolcalling"
        )


if __name__ == "__main__":
    main()
