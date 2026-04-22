"""Shared utilities for intervention analysis plots.

Reads from a downloaded JSONL file (via HuggingFace) instead of the local
filesystem.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_JSONL = _SCRIPT_DIR / "results" / "data" / "intervention_reports.jsonl"

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

# HF uses "tool_calling"; original scripts use "toolcalling"
_AGENT_HF_MAP = {"react": "react", "toolcalling": "tool_calling"}
_AGENT_REVERSE = {v: k for k, v in _AGENT_HF_MAP.items()}


@lru_cache(maxsize=1)
def _load_reports(path: str | Path | None = None) -> pd.DataFrame:
    """Load intervention reports JSONL. Cached after first call."""
    p = Path(path) if path else _DEFAULT_JSONL
    reports = pd.read_json(p, lines=True)
    # Normalise agent_type to match original convention
    reports["agent_type"] = reports["agent_type"].map(
        lambda a: _AGENT_REVERSE.get(a, a)
    )
    return reports


def load_reports(path: str | Path | None = None) -> pd.DataFrame:
    """Public entry point. Returns the full reports DataFrame."""
    return _load_reports(str(path) if path else None)


def _get_row(df: pd.DataFrame, env: str, agent: str, condition: str) -> dict | None:
    """Get a single report row as a dict."""
    mask = (
        (df["environment"] == env)
        & (df["agent_type"] == agent)
        & (df["condition"] == condition)
    )
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _decode_task_results(row: dict) -> dict:
    """Decode Task Results from JSON string if needed."""
    tr = row.get("Task Results")
    if isinstance(tr, str):
        return json.loads(tr)
    return tr or {}


def get_intervention_task_ids(
    env: str, agent: str, df: pd.DataFrame | None = None
) -> set[str]:
    """Get the set of task_ids that appear in any intervention run for env/agent."""
    if df is None:
        df = load_reports()  # noqa: PD901
    task_ids = set()
    for step in ALL_INTERVENTION_STEPS:
        row = _get_row(df, env, agent, step)
        if row is None:
            continue
        task_ids.update(_decode_task_results(row).keys())
    return task_ids


def get_matched_baseline_metrics(
    env: str,
    agent: str,
    metric_prefix: str = "Task Pass@",
    df: pd.DataFrame | None = None,
) -> dict | None:
    """Compute baseline metrics using only tasks present in intervention runs."""
    if df is None:
        df = load_reports()  # noqa: PD901
    row = _get_row(df, env, agent, "baseline")
    if row is None:
        return None

    matched_tasks = get_intervention_task_ids(env, agent, df)
    if not matched_tasks:
        return None

    task_results = _decode_task_results(row)
    metric_keys: dict[str, list[float]] = {}
    for task_id, task_data in task_results.items():
        if task_id not in matched_tasks:
            continue
        for key, val in task_data.items():
            if key.startswith(metric_prefix):
                short_key = key.replace("Task ", "")
                metric_keys.setdefault(short_key, []).append(val)

    if not metric_keys:
        return None
    return {k: np.mean(v) for k, v in metric_keys.items()}


def get_matched_baseline_pass_at(
    env: str, agent: str, df: pd.DataFrame | None = None
) -> dict | None:
    return get_matched_baseline_metrics(env, agent, "Task Pass@", df)


def get_matched_baseline_pass_caret(
    env: str, agent: str, df: pd.DataFrame | None = None
) -> dict | None:
    return get_matched_baseline_metrics(env, agent, "Task Pass^", df)


def extract_matched_pass_at(
    env: str, agent: str, df: pd.DataFrame | None = None
) -> tuple[list[int], list[float]] | None:
    metrics = get_matched_baseline_pass_at(env, agent, df)
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
    env: str, agent: str, df: pd.DataFrame | None = None
) -> tuple[list[int], list[float]] | None:
    metrics = get_matched_baseline_pass_caret(env, agent, df)
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
    env: str, metric_type: str = "pass_at", df: pd.DataFrame | None = None
) -> tuple[list[int], list[float]] | None:
    """Average matched baseline across agents for an environment."""
    extract_fn = (
        extract_matched_pass_at
        if metric_type == "pass_at"
        else extract_matched_pass_caret
    )
    all_vals = []
    for agent in AGENTS:
        result = extract_fn(env, agent, df)
        if result is None:
            continue
        _, vals = result
        all_vals.append(vals)
    if not all_vals:
        return None
    avg = np.mean(all_vals, axis=0)
    return list(range(1, len(avg) + 1)), avg.tolist()


def get_matched_baseline_success_rate(
    env: str, agent: str, df: pd.DataFrame | None = None
) -> float | None:
    if df is None:
        df = load_reports()  # noqa: PD901
    row = _get_row(df, env, agent, "baseline")
    if row is None:
        return None

    matched_tasks = get_intervention_task_ids(env, agent, df)
    if not matched_tasks:
        return None

    task_results = _decode_task_results(row)
    task_rates = []
    for task_id, task_data in task_results.items():
        if task_id not in matched_tasks:
            continue
        rate = task_data.get("Task Success Rate")
        if rate is not None:
            task_rates.append(rate)
        else:
            trials = task_data.get("trials", [])
            if trials:
                task_rates.append(
                    sum(1 for t in trials if t.get("success")) / len(trials)
                )

    return np.mean(task_rates) if task_rates else None


def load_metrics(
    env: str, agent: str, step: str, df: pd.DataFrame | None = None
) -> dict | None:
    """Load metrics for a given env/agent/condition. Returns a dict of metric keys."""
    if df is None:
        df = load_reports()  # noqa: PD901
    row = _get_row(df, env, agent, step)
    if row is None:
        return None
    # Extract Pass@k and Pass^k from the row's columns
    metrics = {}
    for k in range(1, 16):
        for prefix in ["Pass@", "Pass^"]:
            key = f"{prefix}{k}"
            if (
                key in row
                and row[key] is not None
                and not (isinstance(row[key], float) and np.isnan(row[key]))
            ):
                metrics[key] = row[key]
    # Also include other metric columns
    for col in [
        "Average Score",
        "Overall Success Rate",
        "Overall Average Duration",
        "Overall Total Duration",
        "Total Tool Execution Duration",
        "Total Benchmark Duration",
        "Tool Verbosity",
    ]:
        if col in row and row[col] is not None:
            metrics[col] = row[col]
    return metrics if metrics else None


def _is_agent_error(trial: dict) -> bool:
    answer = str(trial.get("submitted_answer", ""))
    return answer.startswith("Error running agent")


def load_report_filtered(
    env: str, agent: str, step: str, df: pd.DataFrame | None = None
) -> dict | None:
    """Load a report and recompute metrics excluding agent-error trials."""
    if df is None:
        df = load_reports()  # noqa: PD901
    row = _get_row(df, env, agent, step)
    if row is None:
        return None

    task_results = _decode_task_results(row)
    task_metrics = {}
    for task_id, task_data in task_results.items():
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
    """Filter a DataFrame so baseline rows only include tasks present in interventions."""
    mask = pd.Series(True, index=df.index)

    for (env, agent), group in df.groupby([env_col, agent_col]):
        intervention_tasks = set(
            group[group[intervention_col] != "none"][task_col].unique()
        )
        if not intervention_tasks:
            continue
        baseline_mask = (
            (df[env_col] == env)
            & (df[agent_col] == agent)
            & (df[intervention_col] == "none")
            & (~df[task_col].isin(intervention_tasks))
        )
        mask = mask & ~baseline_mask

    return df[mask].copy()
