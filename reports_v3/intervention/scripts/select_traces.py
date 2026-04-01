"""Select which tasks to run, based on existing reports_v2 data.

Scans reports_v2/claude_sonnet_45/ workflow reports to find tasks with mixed
success/failure results (20-80% success rate). These are the tasks where
intervention is most interesting.

Outputs task_selection.json with task IDs per (env, agent).
Traces come later from the fresh baseline run.
"""

import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ENVIRONMENTS, INTERVENTION_ROOT, SELECTED_TASKS


def load_report(env_config: dict, agent_type: str) -> dict:
    report_key = "react_report" if agent_type == "react" else "toolcalling_report"
    report_path = env_config["report_v2_path"] / env_config[report_key]
    with report_path.open() as f:
        return json.load(f)


def get_mixed_tasks(report: dict) -> dict[str, dict]:
    """Find tasks with mixed success/failure from a report."""
    mixed = {}
    for task_id, data in report.get("task_results", {}).items():
        sr = data["success_rate"]
        if 0 < sr < 1:
            trials = data["trials"]
            mixed[task_id] = {
                "success_rate": sr,
                "n_trials": len(trials),
                "successes": sum(1 for t in trials if t["success"]),
                "failures": sum(1 for t in trials if not t["success"]),
            }
    return mixed


def build_task_selection() -> dict:
    """Build task selection from reports_v2.

    For each (env, agent), validates that SELECTED_TASKS from config
    have mixed results, and reports which are usable.
    """
    selection = {}

    for env_name, env_config in ENVIRONMENTS.items():
        target_tasks = SELECTED_TASKS.get(env_name, [])
        logger.info(f"\n=== {env_name.upper()} ===")

        for agent_type in ["react", "toolcalling"]:
            report = load_report(env_config, agent_type)
            mixed = get_mixed_tasks(report)
            key = f"{env_name}/{agent_type}"

            logger.info(f"\n  Agent: {agent_type}")

            valid_tasks = []
            for task_id in target_tasks:
                if task_id in mixed:
                    info = mixed[task_id]
                    valid_tasks.append(task_id)
                    logger.info(
                        f"    {task_id}: {info['successes']}/{info['n_trials']} "
                        f"success ({info['success_rate']:.0%}) -> INCLUDED"
                    )
                else:
                    all_tasks = report.get("task_results", {})
                    if task_id in all_tasks:
                        sr = all_tasks[task_id]["success_rate"]
                        label = "ALL_SUCCESS" if sr == 1 else "ALL_FAIL"
                        logger.info(
                            f"    {task_id}: {label} for {agent_type} -> SKIPPED"
                        )
                    else:
                        logger.info(f"    {task_id}: not found in report -> SKIPPED")

            selection[key] = {
                "environment": env_name,
                "agent_type": agent_type,
                "task_ids": valid_tasks,
            }

    return selection


def main():
    selection = build_task_selection()

    output_path = INTERVENTION_ROOT / "task_selection.json"
    with output_path.open("w") as f:
        json.dump(selection, f, indent=2)

    logger.info(f"\nTask selection written to {output_path}")

    total = 0
    for key, entry in sorted(selection.items()):
        n = len(entry["task_ids"])
        total += n
        logger.info(f"  {key}: {n} tasks -> {entry['task_ids']}")
    logger.info(f"  Total: {total} (env, agent, task) combos")


if __name__ == "__main__":
    main()
