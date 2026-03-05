"""
Push QA and Reasoning-QA benchmark results to Hugging Face datasets.

Structure:
  tasks/qa/<env>_qa/<model>/reports/
      <timestamp>/          -> individual JSON reports
      topic_reports/        -> aggregated topic JSON reports

  tasks/reasoning_qa/<env>/<model>/reports/
      <timestamp>/          -> individual JSON reports
      topic_reports/        -> aggregated topic JSON reports

Models recognized: claude, gpt, gpt_oss
Skipped directories: tasks_json, tasks

Destinations:
  - topic_reports/  -> jablonkagroup/corral-QAs-topic_reports
  - other reports   -> jablonkagroup/corral-QAs-reports

Config names:
  - tasks/qa       : {env}_qa_{model}         (e.g. afm_qa_claude)
  - reasoning_qa   : {env}_reasoning_qa_{model} (e.g. afm_reasoning_qa_claude)
"""

import argparse
import copy
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger

# Load HF_TOKEN (and any other secrets) from .env
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

_hf_api = HfApi()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_REPORTS = "jablonkagroup/corral-QAs-reports"
REPO_TOPIC = "jablonkagroup/corral-QAs-topic_reports"

KNOWN_MODELS = {"claude", "gpt", "gpt_oss"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def record_from_json(
    data: dict, source_file: str, env: str, model: str, qa_type: str
) -> dict:
    """Convert a loaded JSON dict into a record for the HF dataset.

    Each top-level key in the JSON becomes a column with its value passed
    through completely unchanged.  Nested dicts/lists stay nested.
    Three metadata columns are prepended: source_file, env, model, qa_type.
    """
    return {
        "source_file": source_file,
        "env": env,
        "model": model,
        "qa_type": qa_type,
        **data,
    }


def collect_topic_record(
    topic_dir: Path, env: str, model: str, qa_type: str
) -> dict | None:
    """
    Build ONE record from a topic_reports/ directory.

    summary.json is the base row.  Every other *_report.json file maps to a
    topic name (filename minus '_report.json') and its full content is
    injected as a 'file_report' key inside the matching
    summary['topics_summary'][topic] dict.
    """
    summary_path = topic_dir / "summary.json"
    if not summary_path.exists():
        logger.warning(f"No summary.json in {topic_dir} - skipping topic record.")
        return None

    try:
        summary = read_json(summary_path)
    except Exception as exc:
        logger.warning(f"Could not read {summary_path}: {exc}")
        return None

    # Work on a shallow copy so we don't mutate the original; deep-copy
    # topics_summary so we can safely add file_report keys.
    summary = copy.deepcopy(summary)
    topics_summary: dict = summary.get("topics_summary", {})

    for json_file in sorted(topic_dir.glob("*_report.json")):
        topic_name = json_file.name[: -len("_report.json")]  # strip suffix
        try:
            file_data = read_json(json_file)
        except Exception as exc:
            logger.warning(f"Could not read {json_file}: {exc}")
            continue

        if topic_name in topics_summary:
            topics_summary[topic_name]["file_report"] = file_data
        else:
            # Topic not in summary (shouldn't normally happen) - add it anyway
            topics_summary[topic_name] = {"file_report": file_data}
            logger.debug(f"Topic '{topic_name}' not in summary, added from file.")

    summary["topics_summary"] = topics_summary
    return record_from_json(summary, "topic_reports/summary.json", env, model, qa_type)


def collect_reports(reports_dir: Path, env: str, model: str, qa_type: str):
    """
    Walk reports_dir and return two lists of records:
      - regular_records  (from timestamped subdirs - one row per JSON file)
      - topic_records    (from topic_reports/ - ONE row for the whole directory)
    """
    regular_records = []
    topic_records = []

    if not reports_dir.is_dir():
        logger.warning(f"Reports dir does not exist: {reports_dir}")
        return regular_records, topic_records

    for child in sorted(reports_dir.iterdir()):
        if not child.is_dir():
            continue

        if child.name == "topic_reports":
            rec = collect_topic_record(child, env, model, qa_type)
            if rec is not None:
                topic_records.append(rec)
        else:
            for json_file in sorted(child.glob("*.json")):
                try:
                    data = read_json(json_file)
                except Exception as exc:
                    logger.warning(f"Could not read {json_file}: {exc}")
                    continue
                rel = json_file.relative_to(reports_dir)
                regular_records.append(
                    record_from_json(data, str(rel), env, model, qa_type)
                )

    return regular_records, topic_records


def _mixed_type_columns(records: list[dict], all_keys: set[str]) -> set[str]:
    """
    Return column names where some non-null rows have a list value and others
    have a non-list value at the TOP level.  pyarrow can't infer a schema for
    those without explicit features.
    """
    mixed = set()
    for k in all_keys:
        has_list = False
        has_nonlist = False
        for r in records:
            v = r.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                has_list = True
            else:
                has_nonlist = True
            if has_list and has_nonlist:
                mixed.add(k)
                break
    return mixed


def _serialise_complex(records: list[dict], all_keys: set[str]) -> list[dict]:
    """Serialise every dict/list value to a JSON string (fallback path)."""
    out = []
    for r in records:
        row = {}
        for k in all_keys:
            v = r.get(k)
            if isinstance(v, (dict | list)):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        out.append(row)
    return out


def save_local(records: list[dict], output_dir: Path, subfolder: str, config_name: str):
    """Save a list of records as a JSON file for local inspection."""
    if not records:
        logger.info(
            f"  No records for config '{config_name}' [{subfolder}] - skipping local save."
        )
        return

    dest = output_dir / subfolder
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{config_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    logger.success(f"  Saved {len(records)} records → {out_path}")


def push_config(records: list[dict], repo: str, config_name: str, dry_run: bool):
    """Push a list of records as a named configuration of an HF dataset."""
    if not records:
        logger.info(f"  No records for config '{config_name}' in {repo} - skipping.")
        return

    logger.info(f"  Pushing {len(records)} records → {repo} [config={config_name}]")

    if dry_run:
        logger.info("  [dry-run] skipping actual push.")
        return

    # Union of all keys so every row has the same columns (missing → None)
    all_keys: set[str] = set()
    for r in records:
        all_keys.update(r.keys())

    # Fix top-level mixed list/non-list columns first
    mixed = _mixed_type_columns(records, all_keys)
    if mixed:
        logger.debug(
            f"  Serialising mixed-type columns to JSON string: {sorted(mixed)}"
        )

    normalised = []
    for r in records:
        row = {}
        for k in all_keys:
            v = r.get(k)
            if k in mixed and v is not None:
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        normalised.append(row)

    try:
        ds = Dataset.from_list(normalised)
    except Exception as exc:
        # Nested schema inconsistencies (e.g. list-of-dicts with varying
        # sub-fields, NaN mixed with None, etc.) can still trip up pyarrow.
        # Fall back to serialising all dict/list columns to JSON strings.
        logger.warning(
            f"  pyarrow schema inference failed ({exc}); "
            f"retrying with full dict/list serialisation."
        )
        normalised = _serialise_complex(records, all_keys)
        ds = Dataset.from_list(normalised)

    # Push parquet directly via huggingface_hub to avoid the auto-generated
    # README growing unboundedly (413 Payload Too Large on /api/validate-yaml).
    _hf_api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=False)

    buf = io.BytesIO()
    pq.write_table(ds.data.table, buf)
    buf.seek(0)
    _hf_api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"{config_name}/train-00000-of-00001.parquet",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Add config {config_name}",
    )
    logger.success(f"  Done → {repo} [{config_name}]")


# ---------------------------------------------------------------------------
# Main scan logic
# ---------------------------------------------------------------------------


def scan_qa_dir(base: Path, qa_type: str):
    """
    Yield (env, model, regular_records, topic_records) for every
    <env>/<model>/reports directory under *base*.

    qa_type: "qa" or "reasoning_qa"
    """
    for env_dir in sorted(base.iterdir()):
        if not env_dir.is_dir():
            continue

        # Strip trailing _qa from directory name to get bare env
        raw_name = env_dir.name  # e.g. "afm_qa" or "afm"
        if qa_type == "qa" and raw_name.endswith("_qa"):
            env = raw_name[: -len("_qa")]  # "afm"
        else:
            env = raw_name  # already bare, e.g. "afm"

        for model_dir in sorted(env_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name not in KNOWN_MODELS:
                logger.debug(f"Skipping non-model directory: {model_dir}")
                continue

            model = model_dir.name
            reports_dir = model_dir / "reports"
            regular, topic = collect_reports(reports_dir, env, model, qa_type)
            yield env, model, regular, topic


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Push QA benchmark results to Hugging Face datasets."
    )
    parser.add_argument(
        "--tasks-root",
        default=str(Path(__file__).parent.parent / "tasks"),
        help="Root of the tasks directory (default: ../tasks relative to this script)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push results to Hugging Face datasets (disabled by default).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "output" / "qa_datasets"),
        help="Directory for local JSON output (default: ../output/qa_datasets).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and log without actually pushing to HF.",
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Only process tasks/qa (skip reasoning_qa).",
    )
    parser.add_argument(
        "--reasoning-only",
        action="store_true",
        help="Only process tasks/reasoning_qa (skip qa).",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root)
    output_dir = Path(args.output_dir)
    qa_base = tasks_root / "qa"
    reasoning_base = tasks_root / "reasoning_qa"

    # Accumulate: (repo, config_name) -> list[records]
    all_regular: dict[str, list[dict]] = {}
    all_topic: dict[str, list[dict]] = {}

    sources = []
    if not args.reasoning_only:
        sources.append((qa_base, "qa"))
    if not args.qa_only:
        sources.append((reasoning_base, "reasoning_qa"))

    for base, qa_type in sources:
        if not base.is_dir():
            logger.warning(f"Directory not found: {base}")
            continue
        logger.info(f"Scanning {base} (qa_type={qa_type})")

        for env, model, regular, topic in scan_qa_dir(base, qa_type):
            config_name = f"{env}_{qa_type}_{model}"  # e.g. afm_qa_claude
            logger.info(
                f"  {env}/{model}: {len(regular)} regular, {len(topic)} topic records  [config={config_name}]"
            )
            all_regular.setdefault(config_name, []).extend(regular)
            all_topic.setdefault(config_name, []).extend(topic)

    # Always save locally for inspection
    logger.info(f"\nSaving regular reports locally → {output_dir / 'reports'}")
    for config_name, records in sorted(all_regular.items()):
        save_local(records, output_dir, "reports", config_name)

    logger.info(f"\nSaving topic reports locally → {output_dir / 'topic_reports'}")
    for config_name, records in sorted(all_topic.items()):
        save_local(records, output_dir, "topic_reports", config_name)

    if not args.push:
        logger.info(
            "\n[no --push] Skipping Hugging Face upload. Pass --push to enable."
        )
    else:
        # Wipe the data/ folder on both repos before uploading so stale configs
        # from previous runs don't linger.
        for repo in [REPO_REPORTS, REPO_TOPIC]:
            if args.dry_run:
                logger.info(f"[dry-run] would delete data/ on {repo}")
                continue
            try:
                _hf_api.create_repo(
                    repo_id=repo, repo_type="dataset", exist_ok=True, private=False
                )
                _hf_api.delete_folder(
                    path_in_repo="data",
                    repo_id=repo,
                    repo_type="dataset",
                    commit_message="Reset: remove stale data/ before fresh upload",
                )
                logger.info(f"Deleted data/ on {repo}")
            except Exception as exc:
                # Folder may not exist yet on the first run - that's fine.
                logger.info(
                    f"Could not delete data/ on {repo} (probably doesn't exist yet): {exc}"
                )

        # Push regular reports
        logger.info(f"\nPushing regular reports → {REPO_REPORTS}")
        for config_name, records in sorted(all_regular.items()):
            push_config(records, REPO_REPORTS, config_name, args.dry_run)

        # Push topic reports
        logger.info(f"\nPushing topic reports → {REPO_TOPIC}")
        for config_name, records in sorted(all_topic.items()):
            push_config(records, REPO_TOPIC, config_name, args.dry_run)

    logger.success("All done.")


if __name__ == "__main__":
    main()
