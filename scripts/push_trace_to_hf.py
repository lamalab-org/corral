"""
Collect benchmark trial + message + logprobs into a dataset.

Output schema (one row per assistant message):
- task
- trial
- environment
- verbosity
- agent_type
- message_number
- message_id
- level (int)
- per_token_entropy (list[float])
- per_token_logprob (list[float])
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value, load_dataset
from loguru import logger
from utils import entropy_from_top_logprobs, read_json, safe_float

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def infer_agent_and_verbosity_from_dir(
    dirname: str,
) -> tuple[str | None, str | None]:
    """
    Examples:
      logprobs_brief_react -> verbosity=brief, agent_type=react
      logprobs_comprehensive_tool_calling -> verbosity=comprehensive, agent_type=tool_calling
    """
    m = re.match(
        r"^logprobs_(brief|workflow|comprehensive)_(react|tool_calling)$", dirname
    )
    if not m:
        return None, None
    return m.group(2), m.group(1)


def infer_environment_from_path(p: Path) -> str:
    """
    "environment" in your description seems to be the environment directory like catalyst_single, ml_chained, etc.
    We infer it as the nearest ancestor that contains agent_logs-* OR logprobs_* folders.
    """
    cur = p.resolve()
    for parent in [cur, *list(cur.parents)]:
        if any(
            (parent / x).is_dir()
            for x in [
                "agent_logs-ReActAgent-openai",
                "agent_logs-ToolCallingAgent-openai",
                "logprobs_brief_react",
                "logprobs_brief_tool_calling",
                "logprobs_workflow_react",
                "logprobs_workflow_tool_calling",
                "logprobs_comprehensive_react",
                "logprobs_comprehensive_tool_calling",
            ]
        ):
            return parent.name
    return p.parent.name


@dataclass
class UpdateFilter:
    """Filter specifying which rows to replace during incremental update."""

    environment: str  # "tasks" or "subtasks"
    agent_type: str  # "react" or "tool_calling"
    verbosity: list[str]  # e.g. ["brief", "workflow"]
    level: int  # 1 or 2


def parse_update_filters(raw: str | list) -> list[UpdateFilter]:
    """Parse JSON string (or already-parsed list) into a list of UpdateFilter objects.

    Expected format:
    [
      {"environment": "subtasks", "agent_type": "react", "verbosity": ["brief", "workflow"], "level": 1},
      {"environment": "tasks", "agent_type": "react", "verbosity": ["brief", "comprehensive"], "level": 2}
    ]
    """
    data = raw if isinstance(raw, list) else json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("--update must be a JSON list of filter objects")
    filters: list[UpdateFilter] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Each filter must be a dict, got: {type(item)}")
        for key in ("environment", "agent_type", "verbosity", "level"):
            if key not in item:
                raise ValueError(f"Filter missing required key: {key}")
        verbosity = item["verbosity"]
        if isinstance(verbosity, str):
            verbosity = [verbosity]
        filters.append(
            UpdateFilter(
                environment=item["environment"],
                agent_type=item["agent_type"],
                verbosity=verbosity,
                level=int(item["level"]),
            )
        )
    return filters


def row_matches_filter(row: dict[str, Any], f: UpdateFilter) -> bool:
    """Check if a dataset row matches the given filter criteria."""
    return (
        str(row.get("environment", "")) == f.environment
        and str(row.get("agent_type", "")) == f.agent_type
        and str(row.get("verbosity", "")) in f.verbosity
        and int(row.get("level", -1)) == f.level
    )


def filters_for_config(
    filters: list[UpdateFilter], level_num: int, task_type: str
) -> list[UpdateFilter]:
    """Return only filters that apply to this config (level + environment/task_type)."""
    return [f for f in filters if f.level == level_num and f.environment == task_type]


def download_existing_config(
    dataset_name: str, config_name: str
) -> list[dict[str, Any]]:
    """Download existing HF dataset config and return rows as list of dicts."""
    try:
        ds = load_dataset(dataset_name, config_name)
        rows = ds["train"].to_list()
        logger.info(
            f"Downloaded {len(rows)} existing rows from {dataset_name}/{config_name}"
        )
        return rows
    except Exception as e:
        logger.info(
            f"Could not load existing config '{config_name}' from {dataset_name}: {e}"
        )
        logger.info("Starting with empty dataset for this config.")
        return []


@dataclass
class TrialMessageRef:
    task: str
    trial: str
    score: float | None
    environment: str
    level: int | None  # Changed to int to match schema
    verbosity: str
    agent_type: str
    message_number: int
    message_id: str


def extract_trial_message_refs(
    report_json: dict[str, Any], environment: str, level: int | None
) -> list[TrialMessageRef]:
    """
    From your report example:
      task_results -> task -> trials -> trial -> messages -> assistant messages have "id"
    Report file name encodes verbosity+agent_type but we also read metrics.tool_verbosity.
    """
    out: list[TrialMessageRef] = []

    metrics = report_json.get("metrics", {}) if isinstance(report_json, dict) else {}
    verbosity = metrics.get("tool_verbosity")  # "brief" etc.
    # The 'level' is now passed in as an argument, overriding any value from metrics if it exists
    # level = metrics.get("level", 1)  # for now setting to one since i only have level 1

    task_results = report_json.get("task_results", {})
    if not isinstance(task_results, dict):
        return out

    # Try to infer agent_type from the report filename later
    for task_name, task_blob in task_results.items():
        if not isinstance(task_blob, dict):
            continue
        trials = task_blob.get("trials", [])
        if not isinstance(trials, list):
            continue

        for t in trials:
            if not isinstance(t, dict):
                continue
            trial_id = str(t.get("trial_id", ""))
            score = safe_float(t.get("score"))
            messages = t.get("messages", [])
            if not isinstance(messages, list):
                continue

            msg_num = 0
            for m in messages:
                if not isinstance(m, dict):
                    continue
                if m.get("role") != "assistant":
                    continue
                msg_id = m.get("id")
                if not isinstance(msg_id, str) or not msg_id:
                    continue
                msg_num += 1

                # We don't know agent_type from the report json itself; fill later from filename parsing.
                out.append(
                    TrialMessageRef(
                        task=task_name,
                        trial=trial_id,
                        score=score,
                        environment=environment,
                        level=level,  # Use the passed-in level
                        verbosity=str(verbosity) if verbosity else "unknown",
                        agent_type="unknown",
                        message_number=msg_num,
                        message_id=msg_id,
                    )
                )

    return out


def infer_agent_type_from_report_filename(name: str) -> str | None:
    """
    Examples:
      gpt_oss_120-react-catalyst-brief_verbosity-single_try.json -> react
      gpt_oss_120-tool_calling-catalyst-brief_verbosity-single_try.json -> tool_calling
    """
    if "-react-" in name:
        return "react"
    if "-tool_calling-" in name or "-tool-calling-" in name:
        return "tool_calling"
    # your older ones might contain "ToolCallingAgent" in filenames:
    if "ToolCalling" in name:
        return "tool_calling"
    if "ReAct" in name:
        return "react"
    return None


def load_logprob_files(root: Path) -> dict[str, dict[str, Any]]:
    """
    Return dict keyed by message_id ("chatcmpl-...") with:
      {
        "per_token_logprob": [...],
        "per_token_entropy": [...],
      }
    computed from logprobs.content tokens list.
    """
    by_id: dict[str, dict[str, Any]] = {}

    for lp_path in root.rglob("logprobs_*/*.json"):
        try:
            data = read_json(lp_path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        msg_id = data.get("id")
        if not isinstance(msg_id, str) or not msg_id:
            continue

        logprobs = data.get("logprobs", {})
        content = None
        if isinstance(logprobs, dict):
            content = logprobs.get("content")
        if not isinstance(content, list):
            continue

        token_logps: list[float] = []
        token_ents: list[float] = []

        for tok in content:
            if not isinstance(tok, dict):
                continue
            lp = safe_float(tok.get("logprob"))
            if lp is None:
                continue
            token_logps.append(lp)

            top = tok.get("top_logprobs")
            H = entropy_from_top_logprobs(top) if isinstance(top, list) else None
            token_ents.append(H if H is not None else float("nan"))

        by_id[msg_id] = {
            "per_token_logprob": token_logps,
            "per_token_entropy": token_ents,
            "source_path": str(lp_path),
            "task_id": data.get("task_id"),
            "iteration": data.get("iteration"),
        }

    return by_id


def push_environment_to_hf(
    rows: list[dict[str, Any]],
    dataset_name: str,
    environment_subset: str,
    private: bool = False,
    max_shard_size: str = "1GB",
):
    """
    Push one environment as one HF dataset config (subset).
    - agent_type + verbosity remain columns in the same dataset.
    - Strong typing for size + consistency.
    - Sharding for better streaming/perf.

    Parameters
    ----------
    rows : list[dict]
        Your rows (one row per assistant message).
    dataset_name : str
        Repo id, e.g. "n0w0f/oss-trace-logprobs"
    environment_subset : str
        HF config name, e.g. "catalyst_single"
    private : bool
        Make repo private.
    max_shard_size : str
        e.g. "200MB", "500MB", "1GB"
    """
    if not rows:
        logger.info(f"Skipping push for {environment_subset}: no rows to push.")
        return

    # Superset schema: keep it stable across environments.
    features = Features(
        {
            "task": Value("string"),
            "trial": Value("string"),
            "score": Value("float32"),
            "environment": Value("string"),
            "level": Value("int32"),
            "verbosity": Value("string"),
            "agent_type": Value("string"),
            "message_number": Value("int32"),
            "step_in_trial": Value("int32"),
            "message_id": Value("string"),
            # Big columns: force float32 for size
            "per_token_entropy": Sequence(Value("float32")),
            "per_token_logprob": Sequence(Value("float32")),
            # Optional provenance/debug (keep as string/int, nullable ok)
            "logprobs_source_path": Value("string"),
            "logprobs_task_id": Value("string"),
            "logprobs_iteration": Value("int32"),
            "report_source_path": Value("string"),
        }
    )

    # Normalize rows to match schema (ensure keys exist)
    def normalize(r: dict[str, Any]) -> dict[str, Any]:
        out = {k: r.get(k) for k in features}

        # Ensure sequences are lists (or empty list). Avoid None for Sequence columns.
        out["per_token_entropy"] = out["per_token_entropy"] or []
        out["per_token_logprob"] = out["per_token_logprob"] or []

        # Ensure required scalars exist
        out["environment"] = out["environment"] or environment_subset
        out["verbosity"] = out["verbosity"] or "unknown"
        out["agent_type"] = out["agent_type"] or "unknown"
        out["trial"] = str(out["trial"] or "")
        out["task"] = str(out["task"] or "")
        out["message_id"] = str(out["message_id"] or "")

        # step_in_trial fallback
        if out["step_in_trial"] is None:
            out["step_in_trial"] = (
                out["message_number"] if out["message_number"] is not None else 0
            )

        # Ensure 'level' is an int, default to 0 if None
        if out["level"] is None:
            out["level"] = 0
        else:
            out["level"] = int(out["level"])

        return out

    rows_norm = [normalize(r) for r in rows]

    ds = Dataset.from_list(rows_norm)
    ds = ds.cast(features)  # enforce typing + consistency

    ds.push_to_hub(
        repo_id=dataset_name,
        config_name=environment_subset,
        private=private,
        max_shard_size=max_shard_size,
    )

    logger.info(
        f"✅ Pushed {len(ds)} rows to {dataset_name} (subset/config='{environment_subset}')"
    )


def main() -> None:
    # Set Hugging Face Hub timeout to a higher value to avoid ReadTimeout errors
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "240")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing environment runs. If 'spectra' is given, it will process subdirectories.",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL path (template for multiple files).",
    )
    ap.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Optional parquet output path (template for multiple files).",
    )
    ap.add_argument(
        "--push_to_hub", action="store_true", help="Whether to push to HF datasets."
    )
    ap.add_argument(
        "--hf_dataset_name",
        type=str,
        default="jablonkagroup/corral-oss-trace-logprobs",
        help="HF dataset name.",
    )
    ap.add_argument(
        "--update",
        type=str,
        default=None,
        help=(
            "JSON list of filter sets for incremental update. Each filter is a dict with "
            'keys: environment ("tasks"|"subtasks"), agent_type ("react"|"tool_calling"), '
            "verbosity (list of strings), level (int). "
            "When provided, the existing HF config is downloaded, rows matching the filters "
            "are removed, new local rows matching the filters are added, and the merged "
            "result is pushed. "
            'Example: \'[{"environment": "subtasks", "agent_type": "react", '
            '"verbosity": ["brief", "workflow"], "level": 1}]\''
        ),
    )
    # hf_subset_name is now dynamically generated, so it's not a direct arg
    args = ap.parse_args()

    # Parse update filters if provided
    update_filters: list[UpdateFilter] | None = None
    if args.update:
        update_filters = parse_update_filters(args.update)
        logger.info(f"Update mode: {len(update_filters)} filter(s) provided")
        for i, uf in enumerate(update_filters):
            logger.info(
                f"  Filter {i + 1}: environment={uf.environment}, agent_type={uf.agent_type}, "
                f"verbosity={uf.verbosity}, level={uf.level}"
            )

    base_root = Path(args.root).resolve()

    # Define the structure to iterate
    # This assumes a structure like: base_root/level_1/tasks, base_root/level_1/subtasks, etc.
    levels = {1: ["level_1"], 2: ["level_2"], 3: ["level_3"]}
    task_types = ["tasks", "subtasks"]

    all_rows: list[dict[str, Any]] = []

    for level_num, level_dirs in levels.items():
        for level_dir_name in level_dirs:
            for task_type in task_types:
                current_root = base_root / level_dir_name / task_type

                if not current_root.is_dir():
                    logger.info(f"Skipping {current_root}: Directory not found.")
                    continue

                logger.info(
                    f"Processing directory: {current_root} for level {level_num}, task type {task_type}"
                )

                # 1) Load all logprob files for the current root
                logprob_by_id = load_logprob_files(current_root)

                # 2) Find report JSON files within the current root
                report_paths: list[Path] = []
                for p in current_root.rglob("*.json"):
                    if "logprobs_" in str(p.parent):
                        continue
                    if p.name.startswith("chatcmpl-"):
                        continue
                    # matches your report style names
                    if "verbosity" in p.name and "_try" in p.name:
                        report_paths.append(p)

                if not report_paths:
                    logger.info(
                        f"No report JSON files found in {current_root}. Skipping."
                    )
                    continue

                current_rows: list[dict[str, Any]] = []

                for rp in sorted(report_paths):
                    try:
                        report = read_json(rp)
                    except Exception:
                        logger.info(f"Failed to read report JSON: {rp}")
                        continue
                    if not isinstance(report, dict):
                        logger.info(f"Report JSON is not a dictionary: {rp}")
                        continue

                    # The 'environment' here could be the parent of level_1/level_2, or just the current root name
                    # For clarity, let's use the parent of the current_root (e.g., 'spectra') as the environment
                    # And the specific 'level_1/tasks' as the subset name
                    _environment = (
                        current_root.parent.parent.name
                    )  # This would be 'spectra'
                    agent_from_name = infer_agent_type_from_report_filename(rp.name)

                    # Pass the inferred level to the extraction function
                    refs = extract_trial_message_refs(
                        report, environment=current_root.name, level=level_num
                    )
                    for r in refs:
                        if agent_from_name:
                            r.agent_type = agent_from_name

                        lp = logprob_by_id.get(r.message_id)
                        if lp is None:
                            # no matching logprobs file found for this message_id
                            # logger.info(f"Warning: No logprobs found for message_id {r.message_id} in {rp}")
                            continue

                        current_rows.append(
                            {
                                "task": r.task,
                                "trial": r.trial,
                                "score": r.score,
                                "level": r.level,
                                "environment": r.environment,  # This will be like 'tasks' or 'subtasks'
                                "verbosity": r.verbosity,
                                "agent_type": r.agent_type,
                                "message_number": r.message_number,
                                "message_id": r.message_id,
                                "per_token_entropy": lp["per_token_entropy"],
                                "per_token_logprob": lp["per_token_logprob"],
                                "logprobs_source_path": lp.get("source_path"),
                                "logprobs_task_id": lp.get("task_id"),
                                "logprobs_iteration": lp.get("iteration"),
                                "report_source_path": str(rp),
                                "step_in_trial": r.message_number,  # Added step_in_trial for consistency with schema
                            }
                        )

                # Append to overall rows if you want a single file, or process immediately for separate files/pushes
                all_rows.extend(current_rows)

                # --- Process each subset individually for file output and HF push ---
                subset_name = f"{base_root.name}_{level_dir_name}_{task_type}"

                if update_filters:
                    # ---- Update mode: merge new local rows into existing HF data ----
                    applicable_filters = filters_for_config(
                        update_filters, level_num, task_type
                    )
                    if not applicable_filters:
                        logger.info(
                            f"No update filters match config '{subset_name}', skipping."
                        )
                        continue

                    # Keep only new local rows that match the filter criteria
                    filtered_new_rows = [
                        r
                        for r in current_rows
                        if any(row_matches_filter(r, f) for f in applicable_filters)
                    ]

                    # Download existing dataset and remove entries matching the filters
                    existing_rows = download_existing_config(
                        args.hf_dataset_name, subset_name
                    )
                    kept_rows = [
                        r
                        for r in existing_rows
                        if not any(row_matches_filter(r, f) for f in applicable_filters)
                    ]

                    removed_count = len(existing_rows) - len(kept_rows)
                    logger.info(
                        f"Update mode for '{subset_name}': removed {removed_count} "
                        f"existing rows matching filters, "
                        f"adding {len(filtered_new_rows)} new rows"
                    )

                    rows_to_output = kept_rows + filtered_new_rows
                    if not rows_to_output:
                        logger.info(
                            f"No rows after merge for '{subset_name}', skipping."
                        )
                        continue

                elif current_rows:
                    rows_to_output = current_rows
                else:
                    logger.info(f"No data collected for {level_dir_name}/{task_type}.")
                    continue

                # 3) Write JSONL for the current subset
                out_path = Path(args.out.replace(".jsonl", f"_{subset_name}.jsonl"))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    for row in rows_to_output:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                logger.info(f"Wrote {len(rows_to_output)} rows -> {out_path}")

                # 4) Optional parquet for the current subset
                if args.parquet:
                    try:
                        if not _HAS_PANDAS:
                            raise ImportError
                        _df = pd.DataFrame(rows_to_output)
                        pq_path = Path(
                            args.parquet.replace(".parquet", f"_{subset_name}.parquet")
                        )
                        pq_path.parent.mkdir(parents=True, exist_ok=True)
                        _df.to_parquet(pq_path, index=False)
                        logger.info(f"Wrote parquet -> {pq_path}")
                    except ImportError:
                        logger.info("pandas/pyarrow not installed; skipping parquet.")
                    except Exception as e:
                        logger.info(f"Failed parquet write for {subset_name}: {e}")

                # 5) Optional push to HF for the current subset
                if args.push_to_hub:
                    push_environment_to_hf(
                        rows_to_output,
                        dataset_name=args.hf_dataset_name,
                        environment_subset=subset_name,
                        private=False,
                        max_shard_size="1GB",
                    )

    logger.info("\nFinished processing all specified directories.")


if __name__ == "__main__":
    main()
