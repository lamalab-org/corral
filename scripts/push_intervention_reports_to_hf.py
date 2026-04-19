"""
Push intervention benchmark reports to HuggingFace Hub.

Walks ``analysis/intervention/runs/{env}/{agent}/{condition}/*_report.json``,
adds intervention-specific columns (condition, condition_type, step), and
pushes one HF config (subset) per environment.

Usage
-----
# Dry-run: collect and print summary without pushing
uv run python scripts/push_intervention_reports_to_hf.py --dry_run

# Push everything
uv run python scripts/push_intervention_reports_to_hf.py

# Push a single environment
uv run python scripts/push_intervention_reports_to_hf.py --env catalyst
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import fire
from constants import HF_REPO_INTERVENTION_REPORTS as HF_REPO
from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

RUNS_ROOT = (
    Path(__file__).resolve().parent.parent / "analysis" / "intervention" / "runs"
)

# Directories to skip (old reruns, archives, test runs)
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

# ---------------------------------------------------------------------------
# Condition parsing
# ---------------------------------------------------------------------------

_CONDITION_RE = re.compile(r"^(baseline|success|failed)(?:_step(n?\d+))?$")


def _parse_condition(dirname: str) -> tuple[str, str, int] | None:
    """Parse a condition directory name into (condition, condition_type, step).

    Examples
    --------
    >>> _parse_condition("baseline")
    ('baseline', 'baseline', 0)
    >>> _parse_condition("success_step1")
    ('success_step1', 'success', 1)
    >>> _parse_condition("failed_stepn2")
    ('failed_stepn2', 'failed', -2)
    """
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


# ---------------------------------------------------------------------------
# Metrics normalisation
# ---------------------------------------------------------------------------

_METRICS_SKIP = frozenset({"Total Token Usage", "Total Tool Calls"})


def _normalise_metrics(metrics: dict) -> dict[str, Any]:
    """Flatten token usage and tool calls; pass through all other keys."""
    out: dict[str, Any] = {}

    tu = metrics.get("Total Token Usage") or metrics.get("total_token_usage") or {}
    out["prompt_tokens"] = tu.get("prompt_tokens", 0)
    out["completion_tokens"] = tu.get("completion_tokens", 0)
    out["total_tokens"] = tu.get("total_tokens", 0)

    tc = metrics.get("Total Tool Calls")
    if isinstance(tc, dict):
        out["Total Tool Calls"] = json.dumps(tc)
    else:
        out["Total Tool Calls"] = json.dumps({"successful": 0, "failed": 0, "total": 0})

    for key, value in metrics.items():
        if key in _METRICS_SKIP:
            continue
        # Normalise the two snake_case leftovers
        if key == "tool_verbosity":
            out["Tool Verbosity"] = value
        elif key == "total_benchmark_duration":
            out["Total Benchmark Duration"] = value
        else:
            out[key] = value

    return out


def _build_record(
    data: dict,
    environment: str,
    agent_type: str,
    condition: str,
    condition_type: str,
    step: int,
) -> dict[str, Any]:
    norm = _normalise_metrics(data["metrics"])
    record: dict[str, Any] = {
        "model": "claude-4.5",
        "agent_type": agent_type,
        "environment": environment,
        "condition": condition,
        "condition_type": condition_type,
        "step": step,
    }
    record.update(norm)
    record["Task Results"] = json.dumps(data["task_results"])
    return record


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _find_report(condition_dir: Path) -> Path | None:
    """Return the first *_report.json in a condition directory."""
    for f in condition_dir.iterdir():
        if f.is_file() and f.name.endswith("_report.json"):
            return f
    return None


def _collect(runs_root: Path, env_filter: str | None = None) -> dict[str, list[dict]]:
    """Walk the intervention runs tree and return {config_name: [records]}."""
    configs: dict[str, list[dict]] = defaultdict(list)

    for env_dir in sorted(runs_root.iterdir()):
        if not env_dir.is_dir():
            continue
        env_name = env_dir.name
        if _should_skip(env_name) or env_name not in CANONICAL_ENVS:
            logger.debug(f"Skipping env dir: {env_dir.name}")
            continue
        if env_filter and env_name != env_filter:
            continue

        for agent_dir in sorted(env_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent_type = AGENT_MAP.get(agent_dir.name)
            if agent_type is None:
                logger.debug(f"Skipping unknown agent dir: {agent_dir.name}")
                continue

            for cond_dir in sorted(agent_dir.iterdir()):
                if not cond_dir.is_dir() or _should_skip(cond_dir.name):
                    continue
                parsed = _parse_condition(cond_dir.name)
                if parsed is None:
                    logger.debug(f"Skipping unrecognised condition: {cond_dir.name}")
                    continue
                condition, ctype, step = parsed

                report_path = _find_report(cond_dir)
                if report_path is None:
                    logger.warning(f"No report found in {cond_dir}")
                    continue

                try:
                    data = json.loads(report_path.read_text("utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(f"Failed to read {report_path}: {exc}")
                    continue

                if "metrics" not in data or "task_results" not in data:
                    logger.warning(f"Not a valid report: {report_path}")
                    continue

                record = _build_record(
                    data, env_name, agent_type, condition, ctype, step
                )
                configs[env_name].append(record)
                logger.debug(
                    f"  {env_name}/{agent_dir.name}/{condition} -> "
                    f"{len(data['task_results'])} tasks"
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
        logger.info(f"Pushing config '{cname}' ({len(recs)} records) ...")
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
    """Push intervention reports to HuggingFace Hub.

    Args:
        env: Push only this environment (e.g. 'catalyst', 'spectra').
        dry_run: Collect and print summary without pushing.
    """
    if not RUNS_ROOT.is_dir():
        raise SystemExit(f"Runs root not found: {RUNS_ROOT}")

    configs = _collect(RUNS_ROOT, env_filter=env)
    if not configs:
        raise SystemExit("No intervention reports found.")

    total = sum(len(v) for v in configs.values())
    logger.info(f"Collected {total} records across {len(configs)} configs:")
    for cname in sorted(configs):
        recs = configs[cname]
        conditions = sorted({r["condition"] for r in recs})
        logger.info(f"  {cname:<20} {len(recs):>3} records  conditions: {conditions}")

    if dry_run:
        logger.info("Dry run — nothing pushed.")
        return

    _push_configs(configs)
    logger.success(f"All done: {len(configs)} configs pushed to {HF_REPO}")


if __name__ == "__main__":
    fire.Fire(main)
