"""Combine melting, quenching, and surface_energy reports into a unified 'md' report.

For each matching combination of (level, section, agent_type, verbosity), this script:
1. Loads the individual reports from each MD sub-environment
2. Merges the task_results from all three environments
3. Recomputes the overall metrics using the same formulas as src/corral/report/metrics/core.py
4. Writes the combined report to reports_v2/claude_sonnet_45/md/<level>/<section>/

Usage:
    python scripts/combine_md_reports.py [--base_dir REPORTS_DIR] [--dry_run]
"""

import json
import shutil
from pathlib import Path
from statistics import mean

import fire
from constants import (
    AGENT_CLASS_NAMES,
    AGENT_TYPES,
    MD_ENVS,
    VERBOSITIES,
)
from loguru import logger


def compute_pass_at_k(trials: list[dict], k: int) -> float:
    """Compute pass@k = 1 - (1 - c/n)^k for a single task's trials."""
    n = len(trials)
    if n < k:
        # Match the behavior in the original reports - still compute it
        # using available trials
        pass
    c = sum(1 for t in trials if t["success"])
    if c == n:
        return 1.0
    return 1.0 - (1.0 - c / n) ** k


def compute_pass_hat_k(trials: list[dict], k: int) -> float:
    """Compute pass^k = (c/n)^k for a single task's trials."""
    n = len(trials)
    if n == 0:
        return 0.0
    c = sum(1 for t in trials if t["success"])
    return (c / n) ** k


def compute_task_metrics(task_result: dict) -> dict:
    """Recompute per-task metrics from its trials."""
    trials = task_result["trials"]
    if not trials:
        return task_result

    scores = [t["score"] for t in trials]
    successes = [1 if t["success"] else 0 for t in trials]

    result = {
        "success_rate": mean(successes),
        "average_score": mean(scores),
    }

    for k in range(1, 6):
        result[f"pass@{k}"] = compute_pass_at_k(trials, k)
        result[f"pass^{k}"] = compute_pass_hat_k(trials, k)

    result["trials"] = trials

    total_tokens: dict[str, int] = {}
    for trial in trials:
        if trial.get("token_usage"):
            for key, value in trial["token_usage"].items():
                total_tokens[key] = total_tokens.get(key, 0) + value
    result["total_token_usage"] = total_tokens

    return result


def compute_overall_metrics(task_results: dict[str, dict], verbosity: str) -> dict:
    """Compute overall metrics from merged task_results, mirroring core.py formulas."""
    task_ids = list(task_results.keys())
    num_tasks = len(task_ids)

    if num_tasks == 0:
        return {}

    task_avg_scores = []
    for tid in task_ids:
        trials = task_results[tid]["trials"]
        if trials:
            task_avg_scores.append(mean(t["score"] for t in trials))
    average_score = mean(task_avg_scores) if task_avg_scores else 0.0

    task_success_rates = []
    for tid in task_ids:
        trials = task_results[tid]["trials"]
        if trials:
            task_success_rates.append(mean(1 if t["success"] else 0 for t in trials))
    overall_success_rate = mean(task_success_rates) if task_success_rates else 0.0

    metrics: dict = {
        "average_score": average_score,
        "overall_success_rate": overall_success_rate,
    }

    for k in range(1, 6):
        task_pass_at_k = []
        task_pass_hat_k = []
        for tid in task_ids:
            trials = task_results[tid]["trials"]
            if trials:
                task_pass_at_k.append(compute_pass_at_k(trials, k))
                task_pass_hat_k.append(compute_pass_hat_k(trials, k))
        metrics[f"pass@{k}"] = mean(task_pass_at_k) if task_pass_at_k else 0.0
        metrics[f"pass^{k}"] = mean(task_pass_hat_k) if task_pass_hat_k else 0.0

    metrics["total_tasks"] = num_tasks
    metrics["tool_verbosity"] = verbosity

    total_tool_calls = 0
    successful_tool_calls = 0
    failed_tool_calls = 0
    surrendered_trials = 0
    total_token_usage: dict[str, int] = {}
    total_tool_execution_duration = 0.0
    total_benchmark_duration = 0.0

    for tid in task_ids:
        for trial in task_results[tid]["trials"]:
            total_tool_calls += trial.get("total_calls", 0)
            successful_tool_calls += trial.get("successful_calls", 0)
            failed_tool_calls += trial.get("failed_calls", 0)

            if trial.get("surrendered", False):
                surrendered_trials += 1

            if trial.get("token_usage"):
                for key, value in trial["token_usage"].items():
                    total_token_usage[key] = total_token_usage.get(key, 0) + value

            total_tool_execution_duration += trial.get("tool_execution_duration", 0.0)

    metrics["total_tool_calls"] = total_tool_calls
    metrics["successful_tool_calls"] = successful_tool_calls
    metrics["failed_tool_calls"] = failed_tool_calls
    metrics["surrendered_trials"] = surrendered_trials
    metrics["total_token_usage"] = total_token_usage
    metrics["total_tool_execution_duration"] = total_tool_execution_duration

    # total_benchmark_duration: sum from the source reports (set later by caller)
    metrics["total_benchmark_duration"] = total_benchmark_duration

    return metrics


def find_report_path(
    base_dir: Path,
    env: str,
    level: str,
    section: str,
    agent_type: str,
    verbosity: str,
) -> Path | None:
    """Find the report file path for a given combination."""
    dir_path = base_dir / env / level / section
    filename = f"claude_45-{agent_type}-{env}-{verbosity}-{section.rstrip('s')}-{level}_verbosity_try.json"

    # section is 'tasks' or 'subtasks', the filename uses 'task' or 'subtask' (singular)
    full_path = dir_path / filename
    if full_path.exists():
        return full_path
    return None


def combine_reports(base_dir: Path, dry_run: bool = False) -> None:
    """Main function to combine MD reports."""
    # Discover all available combinations
    # For each env, find what levels and sections exist
    available: dict[str, dict[str, list[str]]] = {}
    for env in MD_ENVS:
        env_dir = base_dir / env
        if not env_dir.exists():
            continue
        available[env] = {}
        for level_dir in sorted(env_dir.iterdir()):
            if not level_dir.is_dir() or not level_dir.name.startswith("level_"):
                continue
            level = level_dir.name
            available[env][level] = []
            for section_dir in sorted(level_dir.iterdir()):
                if not section_dir.is_dir():
                    continue
                if section_dir.name in ("tasks", "subtasks"):
                    available[env][level].append(section_dir.name)

    # Find the union of all (level, section) combos
    all_level_sections: set[tuple[str, str]] = set()
    for env_data in available.values():
        for level, sections in env_data.items():
            for section in sections:
                all_level_sections.add((level, section))

    logger.info(f"Found environments: {list(available.keys())}")
    logger.info(f"Level/section combinations: {sorted(all_level_sections)}")
    logger.info("")

    # Assert no env directories exist in base_dir that are not covered by MD_ENVS.
    # If a new MD sub-environment is added on disk but not to this list, its
    # tasks would be silently excluded from every combined report.
    actual_env_dirs = {
        d.name
        for d in base_dir.iterdir()
        if d.is_dir() and not d.name.startswith("level_")
    }
    unknown_envs = actual_env_dirs - set(MD_ENVS)
    assert not unknown_envs, (
        f"Directories found in {base_dir} that are not listed in MD_ENVS: "
        f"{sorted(unknown_envs)}. Add them to MD_ENVS or remove the directories."
    )

    combined_count = 0

    for level, section in sorted(all_level_sections):
        for agent_type in AGENT_TYPES:
            for verbosity in VERBOSITIES:
                # Collect reports from all envs that have this combination
                source_reports: list[tuple[str, dict]] = []
                total_benchmark_duration = 0.0

                for env in MD_ENVS:
                    report_path = find_report_path(
                        base_dir, env, level, section, agent_type, verbosity
                    )
                    if report_path is not None:
                        with report_path.open() as f:
                            data = json.load(f)
                        source_reports.append((env, data))
                        total_benchmark_duration += data["metrics"].get(
                            "total_benchmark_duration", 0.0
                        )

                if not source_reports:
                    continue

                # Assert every expected MD env contributed a report for this
                # combination.  A missing env means its tasks are silently lost.
                envs_found = {env for env, _ in source_reports}
                missing_envs = set(MD_ENVS) - envs_found
                assert not missing_envs, (
                    f"[{level}/{section}] {agent_type}/{verbosity}: "
                    f"reports missing for envs {sorted(missing_envs)}. "
                    f"Their tasks would be silently excluded from the combined report."
                )

                # Merge task_results from all source reports
                merged_task_results: dict[str, dict] = {}
                expected_total_tasks = 0
                for _env, report_data in source_reports:
                    env_task_ids = list(report_data["task_results"].keys())
                    expected_total_tasks += len(env_task_ids)
                    for task_id, task_result in report_data["task_results"].items():
                        assert task_id not in merged_task_results, (
                            f"Duplicate task_id '{task_id}' found across envs "
                            f"for [{level}/{section}] {agent_type}/{verbosity}. "
                            f"One copy would be silently overwritten."
                        )
                        merged_task_results[task_id] = task_result

                # Assert no tasks were lost during the merge.
                assert len(merged_task_results) == expected_total_tasks, (
                    f"[{level}/{section}] {agent_type}/{verbosity}: "
                    f"expected {expected_total_tasks} tasks after merge but got "
                    f"{len(merged_task_results)}. Some tasks were silently dropped."
                )

                # Recompute per-task metrics from trials
                for task_id, value in merged_task_results.items():
                    merged_task_results[task_id] = compute_task_metrics(value)

                # Recompute overall metrics
                overall_metrics = compute_overall_metrics(
                    merged_task_results, verbosity
                )
                overall_metrics["total_benchmark_duration"] = total_benchmark_duration

                combined_report = {
                    "metrics": overall_metrics,
                    "task_results": merged_task_results,
                }

                # Determine output path
                # section singular for filename — safe only for 'tasks'/'subtasks'
                assert section in ("tasks", "subtasks"), (
                    f"Unexpected section name '{section}': rstrip('s') is only safe "
                    f"for 'tasks' and 'subtasks'. Add explicit handling for this value."
                )
                section_singular = section.rstrip("s")
                out_filename = f"claude_45-{agent_type}-md-{verbosity}-{section_singular}-{level}_verbosity_try.json"
                out_dir = base_dir / level / section
                out_path = out_dir / out_filename

                envs_used = [env for env, _ in source_reports]
                tasks_merged = list(merged_task_results.keys())

                logger.info(
                    f"  [{level}/{section}] {agent_type} / {verbosity}: "
                    f"merged {len(source_reports)} envs ({', '.join(envs_used)}) -> "
                    f"{len(tasks_merged)} tasks {tasks_merged}"
                )

                if not dry_run:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with out_path.open("w") as f:
                        json.dump(combined_report, f, indent=2)
                    logger.info(
                        f"    -> Written to {out_path.relative_to(base_dir.parent.parent)}"
                    )
                else:
                    logger.info(
                        f"    -> Would write to {out_path.relative_to(base_dir.parent.parent)}"
                    )

                # Copy agent logs from each source env into the combined output dir
                agent_class = AGENT_CLASS_NAMES[agent_type]
                # Find the agent_logs dir name pattern for this agent/verbosity
                # Pattern: agent_logs-{AgentClass}-claude-sonnet-4-5-20250929-{verbosity}_{verbosity}
                for env in MD_ENVS:
                    src_section_dir = base_dir / env / level / section
                    if not src_section_dir.exists():
                        continue
                    for logs_dir in sorted(src_section_dir.iterdir()):
                        if not logs_dir.is_dir():
                            continue
                        if not logs_dir.name.startswith(f"agent_logs-{agent_class}-"):
                            continue
                        if f"-{verbosity}_{verbosity}" not in logs_dir.name:
                            continue
                        # This is the matching agent_logs dir for this env
                        dest_logs_dir = out_dir / logs_dir.name
                        log_files = sorted(logs_dir.glob("*.json"))
                        if not log_files:
                            continue
                        if not dry_run:
                            dest_logs_dir.mkdir(parents=True, exist_ok=True)
                            for log_file in log_files:
                                shutil.copy2(log_file, dest_logs_dir / log_file.name)
                        logger.info(
                            f"    -> {'Copied' if not dry_run else 'Would copy'} "
                            f"{len(log_files)} agent logs from {env} to {logs_dir.name}/"
                        )

                combined_count += 1

    logger.info(f"\nTotal combined reports: {combined_count}")


def main(
    base_dir: str = "reports_v2/claude_sonnet_45/md",
    dry_run: bool = False,
) -> None:
    """Combine MD sub-environment reports into a unified 'md' report.

    Args:
        base_dir: Base directory containing melting/, quenching/, surface_energy/.
        dry_run: Print what would be done without writing files.
    """
    base_dir_path = Path(base_dir).resolve()
    if not base_dir_path.exists():
        logger.error(f"Error: Base directory does not exist: {base_dir_path}")
        return

    logger.info(f"Base directory: {base_dir_path}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")

    combine_reports(base_dir_path, dry_run=dry_run)


if __name__ == "__main__":
    fire.Fire(main)
