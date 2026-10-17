"""
Push the standardized October benchmark reports to HuggingFace Hub.

Reads the final reports under `reports/<environment>/` (see `reports/README.md`)
and pushes them to `jablonkagroup/corral-reports-october-runs-v1`:

- one config per environment and level, e.g. `inference_opt-level_1`, with one
  row per final trial: the trial's index fields from `trials.csv` (scores,
  errors, tokens, Langfuse session/trace IDs and URLs) plus its full content
  from the report (messages, tool calls, output, evaluation) as JSON strings;
- a `runs` config with every environment's `runs.csv`.

Reports are read one at a time and rows are streamed to disk, so the large
AI Scientist reports (up to ~700 MB of JSON) do not all sit in memory at once.

Usage
-----
# Build every config locally and print row counts, without pushing
python scripts/push_october_reports_to_hf.py --dry_run

# Push everything (the dataset repo is created private if it does not exist)
python scripts/push_october_reports_to_hf.py

# Push only some configs (plus `runs`, which is always pushed)
python scripts/push_october_reports_to_hf.py --env=psychometrics --level=2
python scripts/push_october_reports_to_hf.py --runs_only
"""

import csv
import gzip
import json
import lzma
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fire
from constants import HF_REPO_OCTOBER_REPORTS as HF_REPO
from datasets import Dataset, Features, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger

WORKSPACE = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE / "reports"
ENVIRONMENTS = ("inference_opt", "psychometrics", "resistor_network")

load_dotenv(dotenv_path=WORKSPACE / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

TRIAL_FEATURES = Features(
    {
        "environment": Value("string"),
        "level": Value("int32"),
        "agent": Value("string"),
        "model": Value("string"),
        "model_id": Value("string"),
        "run_id": Value("string"),
        "task_id": Value("string"),
        "trial": Value("int32"),
        "execution_id": Value("string"),
        "score": Value("float64"),
        "success": Value("bool"),
        "surrendered": Value("bool"),
        "error": Value("string"),
        "duration_s": Value("float64"),
        "tool_execution_duration_s": Value("float64"),
        "input_tokens": Value("int64"),
        "output_tokens": Value("int64"),
        "reasoning_tokens": Value("int64"),
        "llm_calls": Value("int64"),
        "tool_calls": Value("int64"),
        "langfuse_session_id": Value("string"),
        "langfuse_trace_id": Value("string"),
        "langfuse_trace_url": Value("string"),
        "report_file": Value("string"),
        "output_json": Value("large_string"),
        "evaluation_json": Value("large_string"),
        "messages_json": Value("large_string"),
        "tool_calls_json": Value("large_string"),
    }
)


def _load_report(path: Path) -> dict[str, Any]:
    opener = lzma.open if path.suffix == ".xz" else gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _optional(value: str, kind: type) -> Any:
    return kind(value) if value not in ("", None) else None


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _trial_rows(env: str, level: int) -> Iterator[dict[str, Any]]:
    """Yield one row per final trial of *env* at *level*, report by report."""
    env_dir = REPORTS_DIR / env
    index = {
        row["execution_id"]: row
        for row in _read_csv(env_dir / "trials.csv")
        if int(row["level"]) == level
    }
    model_ids = {
        (row["agent"], row["model"]): row["model_id"]
        for row in _read_csv(env_dir / "runs.csv")
    }
    seen = 0
    for report_path in sorted(env_dir.glob(f"level_{level}/*/report.json.[gx]z")):
        report = _load_report(report_path)
        for task in report["task_results"].values():
            for trial in task["trials"]:
                row = index[trial["trial_id"]]
                usage = trial.get("token_usage") or {}
                seen += 1
                yield {
                    "environment": env,
                    "level": level,
                    "agent": row["agent"],
                    "model": row["model"],
                    "model_id": model_ids[(row["agent"], row["model"])],
                    "run_id": row["run_id"],
                    "task_id": row["task_id"],
                    "trial": int(row["trial"]),
                    "execution_id": row["execution_id"],
                    "score": _optional(row["score"], float),
                    "success": row["success"] == "True",
                    "surrendered": bool(trial.get("surrendered")),
                    "error": row["error"] or None,
                    "duration_s": _optional(row["duration_s"], float),
                    "tool_execution_duration_s": trial.get("tool_execution_duration"),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_tokens": usage.get("reasoning_tokens"),
                    "llm_calls": usage.get("llm_calls"),
                    "tool_calls": usage.get("tool_calls"),
                    "langfuse_session_id": row["langfuse_session_id"],
                    "langfuse_trace_id": row["langfuse_trace_id"],
                    "langfuse_trace_url": row["langfuse_trace_url"],
                    "report_file": row["report_file"],
                    "output_json": _json(trial.get("output")),
                    "evaluation_json": _json(trial.get("evaluation")),
                    "messages_json": _json(trial.get("messages")),
                    "tool_calls_json": _json(trial.get("tool_calls")),
                }
        del report
    if seen != len(index):
        raise ValueError(
            f"{env} level {level}: reports hold {seen} trials, trials.csv has {len(index)}"
        )


def _trial_dataset(env: str, level: int, cache_dir: str) -> Dataset:
    return Dataset.from_generator(
        _trial_rows,
        gen_kwargs={"env": env, "level": level},
        features=TRIAL_FEATURES,
        cache_dir=cache_dir,
    )


def _runs_dataset() -> Dataset:
    rows = []
    for env in ENVIRONMENTS:
        for row in _read_csv(REPORTS_DIR / env / "runs.csv"):
            for key in (
                "level",
                "tasks",
                "trials_per_task",
                "trials",
                "passes",
                "error_trials",
            ):
                row[key] = int(row[key])
            for key in ("average_score", "pass_at_1", "pass_at_k", "pass_hat_k"):
                row[key] = float(row[key])
            for key in ("input_tokens", "output_tokens"):
                row[key] = _optional(row[key], int)
            rows.append(row)
    return Dataset.from_list(rows)


def main(
    env: str | None = None,
    level: int | None = None,
    runs_only: bool = False,
    dry_run: bool = False,
    public: bool = False,
    repo: str = HF_REPO,
) -> None:
    """Push the October reports as one config per environment and level.

    Args:
        env: Push only this environment (inference_opt, psychometrics,
            resistor_network).
        level: Push only this level (1 or 2).
        runs_only: Push only the `runs` config.
        dry_run: Build the configs and log their sizes without pushing.
        public: Create the dataset repo public instead of private. Has no
            effect if the repo already exists.
        repo: Target dataset repo.
    """
    if env is not None and env not in ENVIRONMENTS:
        raise SystemExit(
            f"ERROR: Unknown environment '{env}', expected one of {ENVIRONMENTS}"
        )
    if level is not None and level not in (1, 2):
        raise SystemExit("ERROR: --level must be 1 or 2")
    if not dry_run:
        if not HF_TOKEN:
            raise SystemExit("ERROR: HF_TOKEN is not set (add it to .env)")
        HfApi(token=HF_TOKEN).create_repo(
            repo_id=repo, repo_type="dataset", private=not public, exist_ok=True
        )

    # (config name, environment, level); the `runs` config has neither. It
    # indexes every environment and is small, so it is always pushed.
    configs: list[tuple[str, str | None, int | None]] = []
    if not runs_only:
        configs = [
            (f"{env_name}-level_{level_number}", env_name, level_number)
            for env_name in (ENVIRONMENTS if env is None else (env,))
            for level_number in ((1, 2) if level is None else (level,))
        ]
    configs.append(("runs", None, None))

    # A fresh cache per run: datasets keys generator caches on code and
    # arguments, not on the report files, so a reused cache could be stale.
    with tempfile.TemporaryDirectory(prefix="corral-october-reports-") as cache:
        for config_name, env_name, level_number in configs:
            logger.info(f"Building config '{config_name}' …")
            dataset = (
                _runs_dataset()
                if env_name is None
                else _trial_dataset(env_name, level_number, cache)
            )
            logger.info(f"  {len(dataset)} rows, {dataset.data.nbytes / 1e6:.0f} MB")
            if dry_run:
                continue
            dataset.push_to_hub(repo, config_name=config_name, token=HF_TOKEN)
            logger.success(f"  ✓ pushed {config_name}")

    if not dry_run:
        logger.success(f"Done: {len(configs)} configs pushed to {repo}")


if __name__ == "__main__":
    fire.Fire(main)
