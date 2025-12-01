"""Collect and aggregate task and main results information from JSON reports across different models, environments, and levels, and save the aggregated data to JSON files.

This script collects data from both regular tasks (in 'tasks' directories) and subtasks
(in 'subtasks' directories). The output files are:
- all_tasks_info.json: Detailed per-task results for regular tasks
- main_results.json: Aggregated metrics for regular tasks
- all_subtasks_info.json: Detailed per-task results for subtasks
- main_results_subtasks.json: Aggregated metrics for subtasks
"""

import json
from pathlib import Path

from loguru import logger

root_path = Path(__file__).parent.parent


def collect_results_from_directory(
    task_dir: Path, model_name: str, env_name: str, level_num: int
) -> tuple[list[dict], list[dict]]:
    """Collect results from a task or subtask directory.

    Args:
        task_dir: Path to the tasks or subtasks directory
        model_name: Name of the model
        env_name: Name of the environment
        level_num: Level number

    Returns:
        Tuple of (tasks_info list, main_results list)
    """
    tasks_info = []
    main_results = []

    for report_file in task_dir.glob("*.json"):
        with report_file.open() as f:
            report_data = json.load(f)

        # Skip files that don't have the expected metrics structure
        # (e.g., individual task log files mixed with aggregated reports)
        metrics = report_data.get("metrics", {})
        if not metrics:
            continue

        task_results = report_data.get("task_results", {})
        # Skip if task_results is not a dict (some files have lists)
        if not isinstance(task_results, dict):
            continue

        verbosity_level = metrics.get("tool_verbosity", "N/A")
        agent = "react" if "react" in report_file.stem.lower() else "tool"
        main_results.append(
            {
                "model": model_name,
                "environment": env_name,
                "level": level_num,
                "agent_type": agent,
                "tool_verbosity": metrics.get("tool_verbosity", "N/A"),
                "average_score": metrics.get("average_score", "N/A"),
                "overall_success_rate": metrics.get("overall_success_rate", "N/A"),
                "pass@1": metrics.get("pass@1", "N/A"),
                "pass@2": metrics.get("pass@2", "N/A"),
                "pass@3": metrics.get("pass@3", "N/A"),
                "pass@4": metrics.get("pass@4", "N/A"),
                "pass@5": metrics.get("pass@5", "N/A"),
                "pass^1": metrics.get("pass^1", "N/A"),
                "pass^2": metrics.get("pass^2", "N/A"),
                "pass^3": metrics.get("pass^3", "N/A"),
                "pass^4": metrics.get("pass^4", "N/A"),
                "pass^5": metrics.get("pass^5", "N/A"),
                "total_tool_calls": metrics.get("total_tool_calls", "N/A"),
                "successful_tool_calls": metrics.get("successful_tool_calls", "N/A"),
                "failed_tool_calls": metrics.get("failed_tool_calls", "N/A"),
                "total_prompt_tokens": metrics.get("total_token_usage", {}).get(
                    "prompt_tokens", "N/A"
                ),
                "total_completion_tokens": metrics.get("total_token_usage", {}).get(
                    "completion_tokens", "N/A"
                ),
                "total_overall_tokens": metrics.get("total_token_usage", {}).get(
                    "total_tokens", "N/A"
                ),
                "total_tool_execution_time": metrics.get(
                    "total_tool_execution_duration", "N/A"
                ),
                "total_benchmark_time": metrics.get("total_benchmark_duration", "N/A"),
            }
        )
        task_results = report_data.get("task_results", [])
        for task, results in task_results.items():
            tasks_info.append(
                {
                    "model": model_name,
                    "environment": env_name,
                    "level": level_num,
                    "agent_type": agent,
                    "task_id": task,
                    "tool_verbosity": verbosity_level,
                    "success_rate": results.get("success_rate", "N/A"),
                    "average_score": results.get("average_score", "N/A"),
                    "pass@1": results.get("pass@1", "N/A"),
                    "pass@2": results.get("pass@2", "N/A"),
                    "pass@3": results.get("pass@3", "N/A"),
                    "pass@4": results.get("pass@4", "N/A"),
                    "pass@5": results.get("pass@5", "N/A"),
                    "pass^1": results.get("pass^1", "N/A"),
                    "pass^2": results.get("pass^2", "N/A"),
                    "pass^3": results.get("pass^3", "N/A"),
                    "pass^4": results.get("pass^4", "N/A"),
                    "pass^5": results.get("pass^5", "N/A"),
                    "total_prompt_tokens": results.get("total_token_usage", {}).get(
                        "prompt_tokens", "N/A"
                    ),
                    "total_completion_tokens": results.get("total_token_usage", {}).get(
                        "completion_tokens", "N/A"
                    ),
                    "total_overall_tokens": results.get("total_token_usage", {}).get(
                        "total_tokens", "N/A"
                    ),
                    "trials": results.get("trials", []),
                }
            )

    return tasks_info, main_results


def main():
    tasks_info = []
    main_results = []
    subtasks_info = []
    main_results_subtasks = []

    for model in root_path.iterdir():
        if not model.is_dir():
            continue
        if "claude" not in model.name and "gpt" not in model.name:
            continue
        for env in model.iterdir():
            if not env.is_dir():
                continue
            for level in env.iterdir():
                if not level.is_dir():
                    continue

                level_num = int(level.name.split("_")[-1])

                # Collect regular tasks
                task_dir = level / "tasks"
                if task_dir.exists():
                    t_info, m_results = collect_results_from_directory(
                        task_dir, model.name, env.name, level_num
                    )
                    tasks_info.extend(t_info)
                    main_results.extend(m_results)
                else:
                    logger.warning(f"Tasks directory does not exist: {task_dir}")

                # Collect subtasks
                subtask_dir = level / "subtasks"
                if subtask_dir.exists():
                    # Subtask aggregated reports are JSON files directly in the subtasks directory
                    # (not inside agent_logs-* subdirectories)
                    st_info, st_results = collect_results_from_directory(
                        subtask_dir, model.name, env.name, level_num
                    )
                    subtasks_info.extend(st_info)
                    main_results_subtasks.extend(st_results)

    # Save regular task results
    all_tasks_path = (
        root_path.parent / "reports_v2" / "annotations" / "data" / "all_tasks_info.json"
    )
    with all_tasks_path.open("w") as f:
        json.dump(tasks_info, f, indent=4)

    main_results_path = (
        root_path.parent / "reports_v2" / "annotations" / "data" / "main_results.json"
    )
    with main_results_path.open("w") as f:
        json.dump(main_results, f, indent=4)

    logger.info(f"Saved {len(tasks_info)} task entries to {all_tasks_path}")
    logger.info(f"Saved {len(main_results)} main result entries to {main_results_path}")

    # Save subtask results
    all_subtasks_path = (
        root_path.parent
        / "reports_v2"
        / "annotations"
        / "data"
        / "all_subtasks_info.json"
    )
    with all_subtasks_path.open("w") as f:
        json.dump(subtasks_info, f, indent=4)

    main_results_subtasks_path = (
        root_path.parent
        / "reports_v2"
        / "annotations"
        / "data"
        / "main_results_subtasks.json"
    )
    with main_results_subtasks_path.open("w") as f:
        json.dump(main_results_subtasks, f, indent=4)

    logger.info(f"Saved {len(subtasks_info)} subtask entries to {all_subtasks_path}")
    logger.info(
        f"Saved {len(main_results_subtasks)} subtask main result entries to {main_results_subtasks_path}"
    )


if __name__ == "__main__":
    main()
