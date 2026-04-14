"""
Push intervention agent traces to HuggingFace Hub.

Extracts one row **per trial** from the report JSON (messages are already
embedded) under ``reports_v3/intervention/runs/{env}/{agent}/{condition}/``.

Each row contains the full message list plus trial-level metadata, making it
straightforward to analyse individual conversations.

HF config (subset) = environment name.

Usage
-----
# Dry-run: collect and print summary without pushing
uv run python scripts/push_intervention_traces_to_hf.py --dry_run

# Push everything
uv run python scripts/push_intervention_traces_to_hf.py

# Push a single environment
uv run python scripts/push_intervention_traces_to_hf.py --env spectra
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import fire
from constants import HF_REPO_INTERVENTION_TRACES as HF_REPO
from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

RUNS_ROOT = (
    Path(__file__).resolve().parent.parent / "reports_v3" / "intervention" / "runs"
)

_SKIP_SUFFIXES = ("_old", "_archive", "_test", "_rerun")

CANONICAL_ENVS = {
    "catalyst",
    "md",
    "ml",
    "resistor",
    "retrosynthesis",
    "spectra",
    "wetlab",
}

AGENT_MAP = {
    "react": "react",
    "toolcalling": "tool_calling",
}

_CONDITION_RE = re.compile(r"^(baseline|success|failed)(?:_step(n?\d+))?$")


def _parse_condition(dirname: str) -> tuple[str, str, int] | None:
    m = _CONDITION_RE.match(dirname)
    if not m:
        return None
    ctype = m.group(1)
    raw_step = m.group(2)
    if raw_step is None:
        step = 0
    elif raw_step.startswith("n"):
        step = -int(raw_step[1:])
    else:
        step = int(raw_step)
    return dirname, ctype, step


def _should_skip(name: str) -> bool:
    return any(name.endswith(s) or s + "_" in name for s in _SKIP_SUFFIXES)


def _find_report(condition_dir: Path) -> Path | None:
    for f in condition_dir.iterdir():
        if f.is_file() and f.name.endswith("_report.json"):
            return f
    return None


def _extract_trials(
    data: dict,
    environment: str,
    agent_type: str,
    condition: str,
    condition_type: str,
    step: int,
    verbosity: str,
) -> list[dict[str, Any]]:
    """Extract one row per trial from a report's task_results."""
    rows: list[dict[str, Any]] = []
    for task_id, task_data in data["task_results"].items():
        trials = task_data.get("trials", [])
        for trial in trials:
            messages = trial.get("messages", [])
            tu = trial.get("token_usage") or {}
            row: dict[str, Any] = {
                "model": "claude-4.5",
                "environment": environment,
                "agent_type": agent_type,
                "condition": condition,
                "condition_type": condition_type,
                "step": step,
                "verbosity": verbosity,
                "task_id": task_id,
                "trial_id": trial.get("trial_id"),
                "score": trial.get("score"),
                "success": trial.get("success"),
                "surrendered": trial.get("surrendered", False),
                "duration": trial.get("duration"),
                "tool_execution_duration": trial.get("tool_execution_duration"),
                "prompt_tokens": tu.get("prompt_tokens", 0),
                "completion_tokens": tu.get("completion_tokens", 0),
                "total_tokens": tu.get("total_tokens", 0),
                "total_tool_calls": trial.get("total_calls", 0),
                "successful_tool_calls": trial.get("successful_calls", 0),
                "failed_tool_calls": trial.get("failed_calls", 0),
                "num_messages": len(messages),
                "messages": json.dumps(messages),
            }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _collect(runs_root: Path, env_filter: str | None = None) -> dict[str, list[dict]]:
    configs: dict[str, list[dict]] = defaultdict(list)

    for env_dir in sorted(runs_root.iterdir()):
        if not env_dir.is_dir():
            continue
        env_name = env_dir.name
        if _should_skip(env_name) or env_name not in CANONICAL_ENVS:
            continue
        if env_filter and env_name != env_filter:
            continue

        for agent_dir in sorted(env_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent_type = AGENT_MAP.get(agent_dir.name)
            if agent_type is None:
                continue

            for cond_dir in sorted(agent_dir.iterdir()):
                if not cond_dir.is_dir() or _should_skip(cond_dir.name):
                    continue
                parsed = _parse_condition(cond_dir.name)
                if parsed is None:
                    continue
                condition, ctype, step = parsed

                report_path = _find_report(cond_dir)
                if report_path is None:
                    logger.warning(f"No report in {cond_dir}")
                    continue

                try:
                    data = json.loads(report_path.read_text("utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(f"Failed to read {report_path}: {exc}")
                    continue

                if "metrics" not in data or "task_results" not in data:
                    continue

                verbosity = (
                    data["metrics"].get("tool_verbosity")
                    or data["metrics"].get("Tool Verbosity")
                    or "unknown"
                )

                rows = _extract_trials(
                    data, env_name, agent_type, condition, ctype, step, str(verbosity)
                )
                configs[env_name].extend(rows)
                logger.debug(
                    f"  {env_name}/{agent_dir.name}/{condition}: {len(rows)} trials"
                )

    return dict(configs)


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------


def _push_configs(configs: dict[str, list[dict]]) -> None:
    """Push configs as parquet files via HfApi (bypasses datasets README bug)."""
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)

    for cname in sorted(configs):
        recs = configs[cname]
        logger.info(f"Pushing config '{cname}' ({len(recs)} rows) ...")
        ds = Dataset.from_list(recs)
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = Path(tmpdir) / "train-00000-of-00001.parquet"
            ds.to_parquet(str(pq))
            api.upload_file(
                path_or_fileobj=str(pq),
                path_in_repo=f"{cname}/train-00000-of-00001.parquet",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
        logger.success(f"  -> {cname}")


def main(
    env: str | None = None,
    dry_run: bool = False,
) -> None:
    """Push intervention traces (one row per trial) to HuggingFace Hub.

    Args:
        env: Push only this environment (e.g. 'spectra').
        dry_run: Collect and print summary without pushing.
    """
    if not RUNS_ROOT.is_dir():
        raise SystemExit(f"Runs root not found: {RUNS_ROOT}")

    configs = _collect(RUNS_ROOT, env_filter=env)
    if not configs:
        raise SystemExit("No intervention traces found.")

    total = sum(len(v) for v in configs.values())
    logger.info(f"Collected {total} trial rows across {len(configs)} configs:")
    for cname in sorted(configs):
        recs = configs[cname]
        conditions = sorted({r["condition"] for r in recs})
        agents = sorted({r["agent_type"] for r in recs})
        logger.info(
            f"  {cname:<20} {len(recs):>5} rows  "
            f"agents: {agents}  conditions: {conditions}"
        )

    if dry_run:
        logger.info("Dry run -- nothing pushed.")
        return

    _push_configs(configs)
    logger.success(f"All done: {len(configs)} configs pushed to {HF_REPO}")


if __name__ == "__main__":
    fire.Fire(main)
