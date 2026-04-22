"""
Push corral benchmark reports to HuggingFace Hub.

Collects report JSONs from reports/ and reports_v2/, normalises all
metric keys to Title-Case (the "newer" convention used by gpt-oss-120b
reports), flattens token usage, and pushes one HF config per
model x environment pair.

The mode for pushing a single config expects a directory structured like
`<root>/level_N/{tasks,subtasks}/*.json`.

Usage
-----
# Push every report from all hardcoded directories
python scripts/push_reports_to_hf.py

# Push (or overwrite) a single model/env combination
python scripts/push_reports_to_hf.py \
    --reports_path=reports_v2/gpt-4o/spectra \
    --model=gpt-4o --env=spectra
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import fire
from constants import (
    ALL_COLUMNS,
    ENV_MAP,
    METRICS_KEY_MAP,
    MODEL_DIR_MAP,
    MODEL_DISPLAY,
    TASK_KEY_MAP,
)
from constants import (
    HF_REPO_REPORTS as HF_REPO,
)
from constants import (
    METRICS_SKIP_KEYS as _METRICS_SKIP_KEYS,
)
from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
from utils import (
    infer_agent_type as _infer_agent_type,
)
from utils import (
    infer_model_from_str as _infer_model_from_str,
)
from utils import (
    is_report_json as _is_report_json,
)
from utils import (
    load_report as _load_report,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")


def _normalise_metrics(metrics: dict) -> dict[str, Any]:
    """Return a flat dict with Title-Case keys, flattened tokens, merged tool-calls."""
    out: dict[str, Any] = {}

    # Token usage is stored as a nested dict in older reports; flatten to scalar columns for easy dataset filtering
    tu = metrics.get("total_token_usage") or metrics.get("Total Token Usage") or {}
    out["prompt_tokens"] = tu.get("prompt_tokens", 0)
    out["completion_tokens"] = tu.get("completion_tokens", 0)
    out["total_tokens"] = tu.get("total_tokens", 0)

    # Older reports split counts into successful/failed/total scalars; newer ones provide a single dict — unify both into the same JSON string
    if "Total Tool Calls" in metrics and isinstance(metrics["Total Tool Calls"], dict):
        out["Total Tool Calls"] = json.dumps(metrics["Total Tool Calls"])
    else:
        out["Total Tool Calls"] = json.dumps(
            {
                "successful": metrics.get("successful_tool_calls", 0),
                "failed": metrics.get("failed_tool_calls", 0),
                "total": (
                    metrics.get("total_tool_calls")
                    or metrics.get("Total Tool Calls")
                    or 0
                ),
            }
        )

    # Apply the Title-Case mapping table; pass through any key not present in the map unchanged so future metrics are not silently dropped
    for key, value in metrics.items():
        if key in _METRICS_SKIP_KEYS:
            continue
        norm = METRICS_KEY_MAP.get(key, key)
        out[norm] = value

    return out


def _normalise_task_results(task_results: dict) -> dict:
    """
    Normalise per-task aggregate keys to Title Case (`Task Success Rate` etc.).
    Keeps trial-level data untouched.
    """
    normalised: dict[str, Any] = {}
    for task_name, task_data in task_results.items():
        new_task: dict[str, Any] = {}
        for key, value in task_data.items():
            new_task[TASK_KEY_MAP.get(key, key)] = value
        normalised[task_name] = new_task
    return normalised


def _ensure_columns(record: dict) -> dict:
    """Fill missing columns with `None` so every record has the same schema."""
    for col in ALL_COLUMNS:
        record.setdefault(col, None)
    return record


_VERBOSITY_WORDS = ("brief", "comprehensive", "workflow")


def _extract_verbosity(filename: str) -> str | None:
    """Return the first verbosity word found in *filename*, or `None`."""
    fl = filename.lower()
    for v in _VERBOSITY_WORDS:
        if v in fl:
            return v
    return None


def _find_agent_logs_dir(
    report_path: Path, agent_type: str, verbosity: str
) -> Path | None:
    """
    Find the `agent_logs-*` directory in the same folder as *report_path*
    that matches *agent_type* ("react"/"tool_calling") and *verbosity*.

    Dir names look like:
      agent_logs-ReActAgent-claude-sonnet-4-5-20250929-brief
      agent_logs-ReActAgent-claude-sonnet-4-5-20250929-brief_brief  (md variant)
      agent_logs-ToolCallingAgent-gpt_oss_120b-workflow
    The last hyphen-separated segment carries the verbosity token.
    """
    agent_class = "ReActAgent" if agent_type == "react" else "ToolCallingAgent"
    for d in report_path.parent.iterdir():
        if not d.is_dir() or not d.name.startswith("agent_logs-"):
            continue
        if agent_class not in d.name:
            continue
        last_seg = d.name.rsplit("-", 1)[-1]  # e.g. "brief" or "brief_brief"
        if (
            last_seg == verbosity
            or last_seg.startswith(f"{verbosity}_")
            or last_seg.endswith(f"_{verbosity}")
        ):
            return d
    return None


def _load_trial_messages(agent_logs_dir: Path, task_id: str) -> list[list | None]:
    """
    Return a list of message-lists for *task_id* sorted oldest-first (trial 1 first).

    Files are named `{task_id}_{YYYYMMDD_HHMMSS}.json`; sorting by the
    datetime suffix gives chronological order which matches report trial order.
    """
    prefix = f"{task_id}_"
    candidates = sorted(
        (
            f
            for f in agent_logs_dir.iterdir()
            if f.name.startswith(prefix) and f.suffix == ".json"
        ),
        key=lambda p: p.stem[len(task_id) + 1 :],  # "YYYYMMDD_HHMMSS" suffix
    )
    result: list[list | None] = []
    for fp in candidates:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            result.append(d.get("messages"))
        except Exception as exc:
            logger.debug("Failed to read agent log {}: {}", fp.name, exc)
            result.append(None)
    return result


def _enrich_task_results_with_messages(
    task_results: dict, report_path: Path, agent_type: str
) -> dict:
    """
    Attach `messages` from the matching `agent_logs-*` directory to every trial.

    - If messages are already embedded in the first trial (gpt-oss-120b style)
      the function returns *task_results* unchanged.
    - If no matching `agent_logs` directory is found a debug message is logged
      and *task_results* is returned unchanged.
    - Older trials (smaller trial_id) correspond to earlier datetime stamps in
      the agent-log filenames, so files are matched by chronological sort order.
    """
    first_trials = next(iter(task_results.values()), {}).get("trials", [])
    if first_trials and "messages" in first_trials[0]:
        return task_results

    verbosity = _extract_verbosity(report_path.name)
    if verbosity is None:
        logger.warning(
            "Cannot extract verbosity from '{}'; skipping message enrichment.",
            report_path.name,
        )
        return task_results

    logs_dir = _find_agent_logs_dir(report_path, agent_type, verbosity)
    if logs_dir is None:
        logger.debug(
            "No agent_logs dir found for '{}' (agent={}, verbosity={}).",
            report_path.name,
            agent_type,
            verbosity,
        )
        return task_results

    enriched: dict = {}
    for task_id, task_data in task_results.items():
        trial_messages = _load_trial_messages(logs_dir, task_id)
        trials = task_data.get("trials", [])
        new_trials = []
        for i, trial in enumerate(trials):
            t = dict(trial)
            if i < len(trial_messages) and trial_messages[i] is not None:
                t["messages"] = trial_messages[i]
            new_trials.append(t)
        new_task = dict(task_data)
        new_task["trials"] = new_trials
        enriched[task_id] = new_task
    return enriched


def _build_record(
    data: dict,
    model: str,
    env: str,
    level: int,
    category: str,
    agent_type: str,
    filepath: Path | None = None,
) -> dict:
    # Guard against callers passing a raw dir fragment instead of a resolved canonical key,
    # which would store an unrecognised string in the 'model' column.
    assert model in MODEL_DISPLAY, (
        f"_build_record received unregistered canonical model '{model}'. "
        f"Add it to MODEL_CANONICAL, MODEL_DISPLAY, and (for v2 dirs) MODEL_DIR_MAP."
    )
    # Guard against an env fragment that was never mapped, which would create a one-off
    # config name and make the record unfindable under the canonical env grouping.
    _canonical_envs = set(ENV_MAP.values())
    assert env in _canonical_envs, (
        f"_build_record received unregistered env '{env}'. "
        f"Add it to ENV_MAP. Known canonical envs: {sorted(_canonical_envs)}"
    )

    # Enrich task_results with messages from agent_logs before normalisation so
    # that the messages end up inside the serialised trial objects.
    task_results = data["task_results"]
    if filepath is not None:
        task_results = _enrich_task_results_with_messages(
            task_results, filepath, agent_type
        )

    norm = _normalise_metrics(data["metrics"])
    norm_tr = _normalise_task_results(task_results)

    record: dict[str, Any] = {
        "model": MODEL_DISPLAY.get(model, model),
        "agent_type": agent_type,
        "environment": env,
        "level": level,
        "category": category,
    }
    record.update(norm)
    record["Task Results"] = json.dumps(norm_tr)
    return _ensure_columns(record)


def _process_afm(base: Path) -> list[dict]:
    """
    `reports/afm/{Task,Subtask}/task_N_model_Agent/*.json`
    """
    records: list[dict] = []
    for category_dir_name in ("Task", "Subtask"):
        cat_path = base / category_dir_name
        if not cat_path.is_dir():
            continue
        category = "task" if category_dir_name == "Task" else "subtask"

        for subdir in sorted(cat_path.iterdir()):
            if not subdir.is_dir():
                continue
            # Dir names encode level, model, and agent type
            m = re.match(
                r"(?:task|subtasks?)_(\d+)_(.+?)_(ReAct|ToolCall(?:ing|ling))",
                subdir.name,
                re.IGNORECASE,
            )
            if not m:
                logger.debug(f"Skipped unmatched AFM dir: {subdir.name}")
                continue
            level = int(m.group(1))
            model = _infer_model_from_str(m.group(2))
            if model is None:
                logger.warning(f"Cannot infer model from AFM dir {subdir.name}")
                continue

            for f in sorted(subdir.iterdir()):
                if not _is_report_json(f):
                    continue
                data = _load_report(f)
                if data is None:
                    continue
                agent = _infer_agent_type(f.name)
                records.append(
                    _build_record(
                        data, model, "afm", level, category, agent, filepath=f
                    )
                )
    return records


def _process_flat(base: Path, env: str) -> list[dict]:
    """
    `reports/{catalyst,ml,resistor_networks}/{model}_{single|chained}/*.json`

    single → task,  chained → subtask.  Level defaults to 1.
    """
    records: list[dict] = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        dirname = subdir.name.lower()

        model = _infer_model_from_str(dirname)
        if model is None:
            logger.debug(f"Skipped flat dir (no model): {subdir.name}")
            continue

        category = "subtask" if "chained" in dirname else "task"

        for f in sorted(subdir.iterdir()):
            if not _is_report_json(f):
                continue
            data = _load_report(f)
            if data is None:
                continue
            agent = _infer_agent_type(f.name)
            records.append(
                _build_record(data, model, env, 1, category, agent, filepath=f)
            )
    return records


def _process_leveled(base: Path, model: str, env: str) -> list[dict]:
    """
    Generic processor for `base/level_N/{tasks,subtasks}/*.json`.
    """
    records: list[dict] = []
    for level_dir in sorted(base.iterdir()):
        if not level_dir.is_dir():
            continue
        lm = re.match(r"level_(\d+)", level_dir.name)
        if not lm:
            continue
        level = int(lm.group(1))

        for cat_dir in sorted(level_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cn = cat_dir.name.lower()
            if cn.startswith("subtask"):
                category = "subtask"
            elif cn.startswith("task"):
                category = "task"
            else:
                continue

            for f in sorted(cat_dir.iterdir()):
                if not _is_report_json(f):
                    continue
                data = _load_report(f)
                if data is None:
                    continue
                agent = _infer_agent_type(f.name)
                records.append(
                    _build_record(data, model, env, level, category, agent, filepath=f)
                )
    return records


def _process_gpt_oss_v1(base: Path) -> list[dict]:
    """
    `reports/gpt-oss-120b/{env}/level_N/{task,subtasks}/*.json`
    """
    records: list[dict] = []
    for env_dir in sorted(base.iterdir()):
        if not env_dir.is_dir():
            continue
        env = ENV_MAP.get(env_dir.name)
        if env is None:
            logger.debug(f"Skipped gpt-oss env dir: {env_dir.name}")
            continue
        records.extend(_process_leveled(env_dir, "gpt_oss_120b", env))
    return records


def _process_v2(base: Path) -> list[dict]:
    """
    `reports_v2/{model}/{env}[/{sub_env}]/level_N/{tasks,subtasks}/*.json`
    """
    records: list[dict] = []
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model = MODEL_DIR_MAP.get(model_dir.name)
        if model is None:
            logger.warning(f"Unknown model dir in reports_v2: {model_dir.name}")
            continue

        for env_dir in sorted(model_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            env = ENV_MAP.get(env_dir.name)
            if env is None:
                logger.debug(f"Skipped v2 env dir: {env_dir.name}")
                continue

            # Without this check, _process_leveled would recurse into sub-env dirs and silently produce no records
            has_levels = any(
                d.name.startswith("level_") for d in env_dir.iterdir() if d.is_dir()
            )
            if has_levels:
                records.extend(_process_leveled(env_dir, model, env))
            else:
                for sub_dir in sorted(env_dir.iterdir()):
                    if sub_dir.is_dir():
                        records.extend(_process_leveled(sub_dir, model, env))
    return records


def _config_name(model_canonical: str, env: str) -> str:
    """Build `{model_underscored}-{env}` config name."""
    display = MODEL_DISPLAY.get(model_canonical, model_canonical)
    model_part = display.replace("-", "_").replace(".", "_")
    return f"{model_part}-{env}"


def _count_report_jsons(directory: Path) -> int:
    """Count JSON files that pass the pre-filter under *directory* (recursive)."""
    return sum(1 for f in directory.rglob("*.json") if _is_report_json(f))


def _collect_all(workspace: Path) -> dict[str, list[dict]]:
    """Walk every hardcoded source directory and return {config: [records]}."""
    all_records: list[dict] = []

    src = workspace / "reports" / "afm"
    if src.is_dir():
        logger.info(f"Processing {src} …")
        before = len(all_records)
        all_records.extend(_process_afm(src))
        # A non-zero JSON count with zero records means every file was dropped by a
        # missing MODEL_CANONICAL entry or a regex mismatch in _process_afm.
        json_count = _count_report_jsons(src)
        assert len(all_records) - before > 0 or json_count == 0, (
            f"{src} contains {json_count} report JSON(s) but produced 0 records — "
            "check MODEL_CANONICAL and the AFM dir-name regex."
        )

    for name, env in [
        ("catalyst", "catalyst"),
        ("ml", "ml"),
        ("resistor_networks", "resistor"),
    ]:
        src = workspace / "reports" / name
        if src.is_dir():
            logger.info(f"Processing {src} (env={env}) …")
            before = len(all_records)
            all_records.extend(_process_flat(src, env))
            json_count = _count_report_jsons(src)
            assert len(all_records) - before > 0 or json_count == 0, (
                f"{src} contains {json_count} report JSON(s) but produced 0 records — "
                f"check MODEL_CANONICAL for '{name}' subdirectory names."
            )

    src = workspace / "reports" / "gpt-oss-120b"
    if src.is_dir():
        logger.info(f"Processing {src} …")
        before = len(all_records)
        all_records.extend(_process_gpt_oss_v1(src))
        json_count = _count_report_jsons(src)
        assert len(all_records) - before > 0 or json_count == 0, (
            f"{src} contains {json_count} report JSON(s) but produced 0 records — "
            "check ENV_MAP for subdirectory names under gpt-oss-120b/."
        )

    src = workspace / "reports_v2"
    if src.is_dir():
        logger.info(f"Processing {src} …")
        before = len(all_records)
        all_records.extend(_process_v2(src))
        json_count = _count_report_jsons(src)
        assert len(all_records) - before > 0 or json_count == 0, (
            f"{src} contains {json_count} report JSON(s) but produced 0 records — "
            "check MODEL_DIR_MAP and ENV_MAP for new subdirectory names under reports_v2/."
        )

    configs: dict[str, list[dict]] = defaultdict(list)
    for rec in all_records:
        model_key = rec["model"].replace("-", "_").replace(".", "_")
        cname = f"{model_key}-{rec['environment']}"
        configs[cname].append(rec)
    return dict(configs)


def _push_configs(configs: dict[str, list[dict]]) -> None:
    for cname in sorted(configs):
        recs = configs[cname]
        logger.info(f"Pushing config '{cname}' ({len(recs)} records) …")
        ds = Dataset.from_list(recs)
        ds.push_to_hub(HF_REPO, config_name=cname, token=HF_TOKEN)
        logger.success(f"  ✓ {cname}")


def main(
    reports_path: str | None = None,
    model: str | None = None,
    env: str | None = None,
) -> None:
    """Push corral benchmark reports to HuggingFace Hub.

    Without arguments every report under `reports/` and `reports_v2/`
    is collected, grouped by model x environment, and pushed as separate
    configs to `jablonkagroup/corral_runs_reports`.

    With `--reports_path` (plus `--model` and `--env`) only that
    single config is pushed (overwriting any existing one).

    Args:
        reports_path: Path to a directory structured as
            `<root>/level_N/{tasks,subtasks}/*.json` (like
            `reports_v2/gpt-4o/spectra`).  Sub-environment directories
            are detected automatically.
        model: Model name  (e.g. gpt-4o, claude-4.5, gpt-oss-120b).
            Required when `reports_path` is set.
        env: Environment name  (e.g. afm, catalyst, md, ml, retro,
            resistor, spectra).  Required when `reports_path` is set.
    """
    workspace = Path(__file__).resolve().parent.parent

    if reports_path is not None:
        if model is None or env is None:
            raise SystemExit(
                "ERROR: --model and --env are required when using --reports_path"
            )

        # Validate and normalise user-supplied strings before any I/O to give early, actionable error messages
        model_canon = _infer_model_from_str(model)
        if model_canon is None:
            raise SystemExit(f"ERROR: Unknown model '{model}'")
        env_canon = ENV_MAP.get(env.lower())
        if env_canon is None:
            raise SystemExit(f"ERROR: Unknown environment '{env}'")

        rpath = Path(reports_path)
        if not rpath.is_absolute():
            rpath = workspace / rpath
        if not rpath.is_dir():
            raise SystemExit(f"ERROR: Directory not found: {rpath}")

        logger.info(
            f"Processing {rpath} (model={MODEL_DISPLAY[model_canon]}, env={env_canon}) …"
        )

        # Same heuristic as in _process_v2: avoids an empty result when the path points to a multi-sub-env root
        has_levels = any(
            d.name.startswith("level_") for d in rpath.iterdir() if d.is_dir()
        )
        records: list[dict] = []
        if has_levels:
            records = _process_leveled(rpath, model_canon, env_canon)
        else:
            for sub in sorted(rpath.iterdir()):
                if sub.is_dir():
                    records.extend(_process_leveled(sub, model_canon, env_canon))

        if not records:
            raise SystemExit("No report files found in the given path.")

        cname = _config_name(model_canon, env_canon)
        logger.info(f"Pushing config '{cname}' ({len(records)} records) …")
        ds = Dataset.from_list(records)
        ds.push_to_hub(HF_REPO, config_name=cname, token=HF_TOKEN)
        logger.success(f"Done: pushed config '{cname}'")

    else:
        # Full mode (all hardcoded dirs)
        configs = _collect_all(workspace)
        if not configs:
            raise SystemExit("No reports found in the workspace.")

        total = sum(len(v) for v in configs.values())
        logger.info(f"Collected {total} reports across {len(configs)} configs:")
        for cname in sorted(configs):
            logger.info(f"  {cname:<30}  {len(configs[cname])} reports")

        _push_configs(configs)
        logger.success(f"All done: {len(configs)} configs pushed.")


if __name__ == "__main__":
    fire.Fire(main)
