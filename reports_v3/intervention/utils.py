"""Shared utilities for intervention analysis plots.

Handles task-matched baseline filtering: ensures baseline metrics
only include tasks that also appear in intervention runs, avoiding
inflated baselines from easy tasks with no failure traces.
"""

import json
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parent / "runs"

AGENTS = ["react", "toolcalling"]
ALL_INTERVENTION_STEPS = [
    "success_step1",
    "success_step2",
    "success_stepn1",
    "success_stepn2",
    "failed_step1",
    "failed_step2",
    "failed_stepn1",
    "failed_stepn2",
]


def get_intervention_task_ids(env: str, agent: str) -> set[str]:
    """Get the set of task_ids that appear in any intervention run for env/agent."""
    task_ids = set()
    for step in ALL_INTERVENTION_STEPS:
        step_dir = RUNS_DIR / env / agent / step
        for report_path in step_dir.glob("*_report.json"):
            with open(report_path) as f:
                data = json.load(f)
            task_ids.update(data.get("task_results", {}).keys())
    return task_ids


def load_baseline_report(env: str, agent: str) -> dict | None:
    """Load the full baseline report JSON."""
    report_glob = list((RUNS_DIR / env / agent / "baseline").glob("*_report.json"))
    if not report_glob:
        return None
    with open(report_glob[0]) as f:
        return json.load(f)


def get_matched_baseline_metrics(
    env: str, agent: str, metric_prefix: str = "Task Pass@"
) -> dict | None:
    """Compute baseline metrics using only tasks present in intervention runs.

    Returns a dict like {"Pass@1": 0.63, "Pass@2": 0.75, ...} averaged
    over only the matched tasks.
    """
    report = load_baseline_report(env, agent)
    if report is None:
        return None

    matched_tasks = get_intervention_task_ids(env, agent)
    if not matched_tasks:
        return None

    # Collect per-task metric values, filtered to matched tasks
    metric_keys = {}  # e.g. "Pass@1" -> [val_task1, val_task2, ...]
    for task_id, task_data in report.get("task_results", {}).items():
        if task_id not in matched_tasks:
            continue
        for key, val in task_data.items():
            if key.startswith(metric_prefix):
                short_key = key.replace("Task ", "")
                metric_keys.setdefault(short_key, []).append(val)

    if not metric_keys:
        return None

    return {k: np.mean(v) for k, v in metric_keys.items()}


def get_matched_baseline_pass_at(env: str, agent: str) -> dict | None:
    """Baseline Pass@k metrics matched to intervention tasks."""
    return get_matched_baseline_metrics(env, agent, "Task Pass@")


def get_matched_baseline_pass_caret(env: str, agent: str) -> dict | None:
    """Baseline Pass^k metrics matched to intervention tasks."""
    return get_matched_baseline_metrics(env, agent, "Task Pass^")


def extract_matched_pass_at(
    env: str, agent: str
) -> tuple[list[int], list[float]] | None:
    """Extract matched baseline Pass@k as (ks, vals) arrays."""
    metrics = get_matched_baseline_pass_at(env, agent)
    if metrics is None:
        return None
    ks, vals = [], []
    for k in range(1, 16):
        key = f"Pass@{k}"
        if key in metrics:
            ks.append(k)
            vals.append(metrics[key])
    return (ks, vals) if ks else None


def extract_matched_pass_caret(
    env: str, agent: str
) -> tuple[list[int], list[float]] | None:
    """Extract matched baseline Pass^k as (ks, vals) arrays."""
    metrics = get_matched_baseline_pass_caret(env, agent)
    if metrics is None:
        return None
    ks, vals = [], []
    for k in range(1, 16):
        key = f"Pass^{k}"
        if key in metrics:
            ks.append(k)
            vals.append(metrics[key])
    return (ks, vals) if ks else None


def avg_matched_baseline(
    env: str, metric_type: str = "pass_at"
) -> tuple[list[int], list[float]] | None:
    """Average matched baseline across agents for an environment.

    metric_type: "pass_at" or "pass_caret"
    """
    extract_fn = (
        extract_matched_pass_at
        if metric_type == "pass_at"
        else extract_matched_pass_caret
    )
    all_vals = []
    for agent in AGENTS:
        result = extract_fn(env, agent)
        if result is None:
            continue
        _, vals = result
        all_vals.append(vals)
    if not all_vals:
        return None
    avg = np.mean(all_vals, axis=0)
    return list(range(1, len(avg) + 1)), avg.tolist()


def get_matched_baseline_success_rate(env: str, agent: str) -> float | None:
    """Get baseline success rate matched to intervention tasks."""
    report = load_baseline_report(env, agent)
    if report is None:
        return None

    matched_tasks = get_intervention_task_ids(env, agent)
    if not matched_tasks:
        return None

    task_rates = []
    for task_id, task_data in report.get("task_results", {}).items():
        if task_id not in matched_tasks:
            continue
        rate = task_data.get("Task Success Rate")
        if rate is not None:
            task_rates.append(rate)
        else:
            # Compute from trials
            trials = task_data.get("trials", [])
            if trials:
                task_rates.append(
                    sum(1 for t in trials if t.get("success")) / len(trials)
                )

    return np.mean(task_rates) if task_rates else None


def get_matched_baseline_pass1(env: str, agent: str) -> float | None:
    """Get baseline Pass@1 matched to intervention tasks."""
    metrics = get_matched_baseline_pass_at(env, agent)
    if metrics is None:
        return None
    return metrics.get("Pass@1")


def _is_agent_error(trial: dict) -> bool:
    """Check if a trial failed due to an agent error (not a legitimate failure)."""
    answer = str(trial.get("submitted_answer", ""))
    return answer.startswith("Error running agent")


def load_report_filtered(env: str, agent: str, step: str) -> dict | None:
    """Load a report JSON and recompute metrics excluding agent-error trials.

    Returns a metrics dict with Pass@k and Pass^k recomputed from valid trials only.
    """
    report_glob = list((RUNS_DIR / env / agent / step).glob("*_report.json"))
    if not report_glob:
        return None
    with open(report_glob[0]) as f:
        report = json.load(f)

    # Collect per-task success counts (excluding errored trials)
    task_metrics = {}
    for task_id, task_data in report.get("task_results", {}).items():
        valid_trials = [
            t for t in task_data.get("trials", []) if not _is_agent_error(t)
        ]
        if not valid_trials:
            continue
        c = sum(1 for t in valid_trials if t.get("success"))
        n = len(valid_trials)
        p = c / n
        task_metrics[task_id] = {"p": p, "n": n, "c": c}

    if not task_metrics:
        return None

    # Compute Pass@k and Pass^k averaged over tasks
    metrics = {}
    max_k = min(t["n"] for t in task_metrics.values())
    for k in range(1, max_k + 1):
        pass_at_vals = []
        pass_hat_vals = []
        for tm in task_metrics.values():
            pass_at_vals.append(
                1.0 if tm["c"] == tm["n"] else 1.0 - (1.0 - tm["p"]) ** k
            )
            pass_hat_vals.append(tm["p"] ** k)
        metrics[f"Pass@{k}"] = np.mean(pass_at_vals)
        metrics[f"Pass^{k}"] = np.mean(pass_hat_vals)

    return metrics


def filter_baseline_to_matched_tasks(
    df,
    env_col="env",
    agent_col="agent",
    intervention_col="intervention",
    task_col="task_id",
):
    """Filter a DataFrame so baseline rows only include tasks present in interventions.

    Returns a new DataFrame with unmatched baseline rows removed.
    """
    import pandas as pd

    mask = pd.Series(True, index=df.index)

    for (env, agent), group in df.groupby([env_col, agent_col]):
        # Get task_ids present in any intervention run
        intervention_tasks = set(
            group[group[intervention_col] != "none"][task_col].unique()
        )
        if not intervention_tasks:
            continue

        # Mark baseline rows for tasks NOT in interventions for removal
        baseline_mask = (
            (df[env_col] == env)
            & (df[agent_col] == agent)
            & (df[intervention_col] == "none")
            & (~df[task_col].isin(intervention_tasks))
        )
        mask = mask & ~baseline_mask

    return df[mask].copy()
