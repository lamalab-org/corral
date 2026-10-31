"""
Push a plotting-friendly summary of the October benchmark reports to HuggingFace Hub.

Reads the final reports under `reports/<environment>/` (see `reports/README.md`)
and pushes three flat tables, without trial transcripts, to
`jablonkagroup/corral-reports-october-runs-summary-v1`:

- `runs`: one row per run (environment x level x agent x model) with scores,
  pass@k / pass^k for k = 1..5, error counts, durations and token usage;
- `tasks`: one row per run and task with the task's pass rate, pass@k /
  pass^k, durations and token usage;
- `trials`: one row per final trial with score, success, error, error
  category, duration, token usage and Langfuse IDs/URLs.

All tables share `environment`, `level`, `agent`, `model` and `run_id`, so they
can be grouped, filtered and joined directly. The full trials with messages are
in `jablonkagroup/corral-reports-october-runs-v1` (push_october_reports_to_hf.py).

Usage
-----
# Build the tables and log their sizes, without pushing
python scripts/push_october_summary_to_hf.py --dry_run

# Push (the dataset repo is created private if it does not exist)
python scripts/push_october_summary_to_hf.py
"""

import os
import re
from typing import Any

import fire
from constants import HF_REPO_OCTOBER_SUMMARY as HF_REPO
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger
from push_october_reports_to_hf import (
    ENVIRONMENTS,
    REPORTS_DIR,
    WORKSPACE,
    _load_report,
    _optional,
    _read_csv,
)

load_dotenv(dotenv_path=WORKSPACE / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

K_VALUES = (1, 2, 3, 4, 5)

# (pattern, error_category, error_kind). Agent errors are the agent's own
# failures and count against it; infrastructure errors are rerun candidates.
ERROR_RULES = (
    (
        r"interaction budget|maximum iterations|error_max_turns",
        "budget_exhausted",
        "agent",
    ),
    (r"without calling submit_answer|without a FinishAction", "no_submission", "agent"),
    (r"Input tokens exceed", "context_limit", "agent"),
    (r"flagged as potentially violating", "policy_refusal", "infrastructure"),
    (r"out-of-memory", "out_of_memory", "infrastructure"),
    (
        r"Operation not permitted|messages\[\d+\]\.name",
        "harness_error",
        "infrastructure",
    ),
    (
        r"Internal server error|Bad Request|Timeout on reading|interrupted unexpectedly",
        "provider_error",
        "infrastructure",
    ),
)


def _classify(error: str) -> tuple[str | None, str | None]:
    if not error:
        return None, None
    for pattern, category, kind in ERROR_RULES:
        if re.search(pattern, error):
            return category, kind
    return "other", "infrastructure"


def _usage(usage: dict | None, prefix: str = "") -> dict[str, Any]:
    usage = usage or {}
    return {
        f"{prefix}{key}": usage.get(key)
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "llm_calls",
            "tool_calls",
            "agent_steps",
        )
    }


def _pass_metrics(metrics: dict, prefix: str) -> dict[str, float | None]:
    out = {}
    for k in K_VALUES:
        out[f"pass_at_{k}"] = metrics.get(f"{prefix}Pass@{k}")
        out[f"pass_hat_{k}"] = metrics.get(f"{prefix}Pass^{k}")
    return out


def _collect() -> tuple[list[dict], list[dict], list[dict]]:
    runs, tasks, trials = [], [], []
    for env in ENVIRONMENTS:
        env_dir = REPORTS_DIR / env
        index = {row["execution_id"]: row for row in _read_csv(env_dir / "trials.csv")}
        for run in _read_csv(env_dir / "runs.csv"):
            logger.info(f"  {env} L{run['level']} {run['agent']} / {run['model']}")
            report = _load_report(env_dir / run["report_file"])
            key = {
                "environment": env,
                "level": int(run["level"]),
                "agent": run["agent"],
                "model": run["model"],
                "model_id": run["model_id"],
                "run_id": run["run_id"],
            }

            run_trials = []
            for task_id, task in sorted(report["task_results"].items()):
                task_trials = []
                for trial in task["trials"]:
                    row = index[trial["trial_id"]]
                    category, kind = _classify(row["error"])
                    task_trials.append(
                        {
                            **key,
                            "task_id": task_id,
                            "trial": int(row["trial"]),
                            "execution_id": row["execution_id"],
                            "score": _optional(row["score"], float),
                            "success": row["success"] == "True",
                            "surrendered": bool(trial.get("surrendered")),
                            "error": row["error"] or None,
                            "error_category": category,
                            "error_kind": kind,
                            "duration_s": _optional(row["duration_s"], float),
                            "tool_execution_duration_s": trial.get(
                                "tool_execution_duration"
                            ),
                            **_usage(trial.get("token_usage")),
                            "langfuse_session_id": row["langfuse_session_id"],
                            "langfuse_trace_id": row["langfuse_trace_id"],
                            "langfuse_trace_url": row["langfuse_trace_url"],
                        }
                    )
                tasks.append(
                    {
                        **key,
                        "task_id": task_id,
                        "trials": len(task_trials),
                        "passes": sum(t["success"] for t in task_trials),
                        "error_trials": sum(
                            t["error"] is not None for t in task_trials
                        ),
                        "average_score": task.get("Task Average Score"),
                        "success_rate": task.get("Task Success Rate"),
                        **_pass_metrics(task, "Task "),
                        "average_duration_s": task.get("Task Average Duration"),
                        "total_duration_s": task.get("Task Total Duration"),
                        **_usage(task.get("Task Total Token Usage"), "total_"),
                    }
                )
                run_trials.extend(task_trials)

            metrics = report["metrics"]
            runs.append(
                {
                    **key,
                    "tasks": int(run["tasks"]),
                    "trials_per_task": int(run["trials_per_task"]),
                    "trials": len(run_trials),
                    "passes": int(run["passes"]),
                    "error_trials": int(run["error_trials"]),
                    "agent_error_trials": sum(
                        t["error_kind"] == "agent" for t in run_trials
                    ),
                    "infrastructure_error_trials": sum(
                        t["error_kind"] == "infrastructure" for t in run_trials
                    ),
                    "average_score": float(run["average_score"]),
                    **_pass_metrics(metrics, ""),
                    "average_duration_s": metrics.get("Overall Average Duration"),
                    "total_duration_s": metrics.get("Overall Total Duration"),
                    "wall_clock_s": metrics.get("total_benchmark_duration"),
                    **_usage(metrics.get("Total Token Usage"), "total_"),
                    "generated_at": run["generated_at"],
                    "report_file": f"reports/{env}/{run['report_file']}",
                    "report_sha256": run["report_sha256"],
                    "langfuse_session_url": run["langfuse_session_url"],
                }
            )
            trials.extend(run_trials)
            del report

        found = sum(t["environment"] == env for t in trials)
        if found != len(index):
            raise ValueError(
                f"{env}: reports hold {found} trials, trials.csv has {len(index)}"
            )
    return runs, tasks, trials


def main(dry_run: bool = False, public: bool = False, repo: str = HF_REPO) -> None:
    """Push the runs, tasks and trials summary tables.

    Args:
        dry_run: Build the tables and log their sizes without pushing.
        public: Create the dataset repo public instead of private. Has no
            effect if the repo already exists.
        repo: Target dataset repo.
    """
    if not dry_run:
        if not HF_TOKEN:
            raise SystemExit("ERROR: HF_TOKEN is not set (add it to .env)")
        HfApi(token=HF_TOKEN).create_repo(
            repo_id=repo, repo_type="dataset", private=not public, exist_ok=True
        )

    logger.info("Reading reports …")
    runs, tasks, trials = _collect()
    for config_name, rows in (("runs", runs), ("tasks", tasks), ("trials", trials)):
        dataset = Dataset.from_list(rows)
        logger.info(
            f"Config '{config_name}': {len(dataset)} rows, {len(dataset.column_names)} columns"
        )
        if dry_run:
            continue
        dataset.push_to_hub(repo, config_name=config_name, token=HF_TOKEN)
        logger.success(f"  ✓ pushed {config_name}")

    if not dry_run:
        logger.success(f"Done: 3 configs pushed to {repo}")


if __name__ == "__main__":
    fire.Fire(main)
