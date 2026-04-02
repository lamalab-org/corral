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
    """Find the directory containing trace JSON files.

    Agent logs are stored in a nested structure:
      agent_logs-{AgentType}-{provider}/{model}-{verbosity}/
    We search recursively for the deepest directory containing trace JSONs.
    """
    # Look for the deepest subdirectory under agent_logs-* that contains .json files
    for log_dir in baseline_dir.glob("agent_logs-*"):
        # Check subdirectories (e.g. model-verbosity dirs)
        for sub in log_dir.iterdir():
            if sub.is_dir() and list(sub.glob("*.json")):
                return sub
        # Fall back to the log_dir itself if it directly contains traces
        if list(log_dir.glob("*.json")):
            return log_dir
    return None


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
            sr = task_data.get("Task Success Rate", task_data.get("success_rate", 0))
            label = "ALL_SUCCESS" if sr == 1 else "ALL_FAIL"
            logger.info(
                f"    {task_id}: {label} ({sr:.0%}) -> SKIPPED (no mixed results)"
            )
            continue

        # Store all traces for both pools (shuffled for randomness)
        rng.shuffle(successes)
        rng.shuffle(failures)

        entries[task_id] = {
            "environment": env,
            "agent_type": agent,
            "task_id": task_id,
            "success_traces": [s["path"] for s in successes],
            "failed_traces": [f["path"] for f in failures],
            "success_rate": task_data.get(
                "Task Success Rate", task_data.get("success_rate", 0)
            ),
            "n_success_traces": len(successes),
            "n_failed_traces": len(failures),
        }
        logger.info(
            f"    {task_id}: OK "
            f"({len(successes)} success, {len(failures)} failed traces, "
            f"baseline_sr={task_data.get('Task Success Rate', task_data.get('success_rate', 0)):.0%})"
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
