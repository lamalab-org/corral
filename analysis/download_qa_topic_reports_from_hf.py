"""
Download QA topic-report configs from HuggingFace Hub into a local dataset.

This script downloads QA evaluation scores for each model across different environments.
The QA topic reports contain aggregated question-answering performance metrics.

Without arguments every config is merged into one `qa_topic_reports.jsonl` file,
which can be filtered by the `model`, `env`, and `qa_type` columns:

    import pandas as pd
    df = pd.read_json("analysis/results/data/qa_topic_reports.jsonl", lines=True)
    df[df["model"] == "claude"]

    from analysis.download_qa_topic_reports_from_hf import load_local_dataset
    df = load_local_dataset()

When specific configs are requested each one is saved under its own name,
making it easy to share or inspect a single model/environment slice:

    df = pd.read_json("analysis/results/data/afm_qa_claude.jsonl", lines=True)

Usage:

```
# Download all configs → analysis/results/data/qa_topic_reports.jsonl
python analysis/download_qa_topic_reports_from_hf.py

# Download specific configs → one file per config
python analysis/download_qa_topic_reports_from_hf.py --configs=afm_qa_claude --configs=catalyst_reasoning_qa_gpt

# Custom output directory
python analysis/download_qa_topic_reports_from_hf.py --output_dir=my_local_data

# Skip configs / pairs already on disk
python analysis/download_qa_topic_reports_from_hf.py --skip_existing
```
"""

import json
import os
import sys
from pathlib import Path

import fire
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from constants import HF_REPO_QA_TOPIC as HF_REPO  # noqa: E402

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

HF_TOKEN: str | None = os.getenv("HF_TOKEN")

_OUTPUT_FILENAME = "qa_topic_reports.jsonl"


def load_local_dataset(output_dir: str | Path = "results/data") -> pd.DataFrame:
    """Load the local combined QA topic reports dataset written by :func:`main`.

    Args:
        output_dir: Directory previously populated by :func:`main`.

    Returns:
        DataFrame with all downloaded QA topic reports, or an empty DataFrame
        when the file does not exist yet.
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    reports_file = output_path / _OUTPUT_FILENAME
    if not reports_file.exists():
        logger.warning(f"No dataset found at '{reports_file}'.")
        return pd.DataFrame()
    reports_df = pd.read_json(reports_file, lines=True)
    logger.info(f"Loaded {len(reports_df):,} rows from '{reports_file}'.")
    return reports_df


def _list_configs(configs: list[str] | None) -> list[str]:
    """Return configs to download, validating any user-supplied names against
    what the Hub actually has to catch typos early."""
    from huggingface_hub import HfApi

    logger.info(f"Fetching config list from '{HF_REPO}' …")

    # List all files in the repo and extract config names from parquet file paths
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type="dataset", token=HF_TOKEN)

    # Extract config names from parquet file paths like "afm_qa_claude/train-00000-of-00001.parquet"
    available = []
    for f in files:
        if f.endswith("/train-00000-of-00001.parquet"):
            config_name = f.split("/")[0]
            available.append(config_name)

    available = sorted(available)
    logger.info(f"  → {len(available)} config(s) found")
    logger.debug(f"  Available configs: {available}")

    if not configs:
        return available

    available_set = set(available)
    unknown = [c for c in configs if c not in available_set]
    if unknown:
        raise SystemExit(
            f"ERROR: the following configs are not present in '{HF_REPO}': "
            f"{unknown}\n"
            f"Available configs: {sorted(available_set)}"
        )
    return sorted(configs)


def _fetch_config(config_name: str) -> pd.DataFrame:
    """Download one Hub config and return it as a DataFrame with nested JSON decoded."""
    logger.info(f"  ↳ Downloading config '{config_name}' …")

    # Download parquet file directly - the configs are stored as parquet files
    # with paths like: afm_qa_claude/train-00000-of-00001.parquet
    parquet_file = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"{config_name}/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    logger.info("  ↳ Downloaded parquet file")

    # Read the parquet file directly with pandas
    config_df = pd.read_parquet(parquet_file)
    logger.info(f"  ↳ Loaded {len(config_df)} rows from parquet")

    # Decode JSON string columns that contain complex nested structures
    # Common columns that might be JSON strings: topics_summary, file_report
    json_columns = ["topics_summary"]
    for col in json_columns:
        if col in config_df.columns:
            config_df[col] = config_df[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )

    return config_df


def main(
    output_dir: str = "results/data",
    skip_existing: bool = False,
    configs: list[str] | None = None,
) -> None:
    """Download HF QA topic report configs and save them locally.

    Args:
        output_dir:
            Directory where output files are written.  Created if absent.
            Default: `results/data/` (relative to `analysis/`).
        skip_existing:
            Full-download mode: skip model x environment x qa_type triples already in
            `qa_topic_reports.jsonl`.  Per-config mode: skip configs whose file
            already exists.
        configs:
            Config names to download.  When omitted all available configs
            are merged into `qa_topic_reports.jsonl`.  When provided each config is
            written to its own `<config_name>.jsonl`.

            Example::

                python analysis/download_qa_topic_reports_from_hf.py --configs=afm_qa_claude --configs=catalyst_reasoning_qa_gpt
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # fire passes a single --configs=value as a bare string; wrap it so the
    # rest of the code always works with a list.
    if isinstance(configs, str):
        configs = [configs]

    config_list = _list_configs(configs or None)
    if not config_list:
        raise SystemExit("No configs to download.")

    per_config_mode = bool(configs)

    if per_config_mode:
        _download_per_config(config_list, output_path, skip_existing)
    else:
        _download_combined(config_list, output_path, skip_existing)


def _download_per_config(
    config_list: list[str], output_path: Path, skip_existing: bool
) -> None:
    """Write each config to its own `<config_name>.jsonl` file."""
    failed: list[str] = []
    for cname in config_list:
        dest = output_path / f"{cname}.jsonl"
        if skip_existing and dest.exists():
            logger.info(f"  ↳ '{cname}' already on disk - skipping.")
            continue
        try:
            config_df = _fetch_config(cname)
            config_df.to_json(dest, orient="records", lines=True, force_ascii=False)
            logger.success(f"  ✓ '{cname}'  {len(config_df):,} rows  →  {dest}")
        except Exception as exc:
            logger.error(f"  ✗ '{cname}' failed: {exc}")
            failed.append(cname)
    if failed:
        logger.warning(f"Failed configs: {failed}")
        raise SystemExit(1)


def _download_combined(
    config_list: list[str], output_path: Path, skip_existing: bool
) -> None:
    """Merge all configs into a single `qa_topic_reports.jsonl`, appending only triples
    not yet present when `skip_existing` is set."""
    reports_file = output_path / _OUTPUT_FILENAME

    existing_triples: set[tuple[str, str, str]] = set()
    if skip_existing and reports_file.exists():
        existing_df = pd.read_json(reports_file, lines=True)
        existing_triples = set(
            zip(
                existing_df["model"],
                existing_df["env"],
                existing_df["qa_type"],
                strict=False,
            )
        )
        logger.info(
            f"Found {len(existing_triples)} existing model x environment x qa_type triple(s) on disk."
        )

    logger.info(f"Will download {len(config_list)} config(s) → '{reports_file}'")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for cname in config_list:
        try:
            config_df = _fetch_config(cname)
            if existing_triples:
                config_df = config_df[
                    ~config_df.apply(
                        lambda r: (r["model"], r["env"], r["qa_type"])
                        in existing_triples,
                        axis=1,
                    )
                ]
                if config_df.empty:
                    logger.info(f"  ↳ '{cname}' already fully present - skipping.")
                    continue
            frames.append(config_df)
            logger.success(f"  ✓ '{cname}'  {len(config_df):,} rows")
        except Exception as exc:
            logger.error(f"  ✗ '{cname}' failed: {exc}")
            failed.append(cname)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        write_mode = "a" if (skip_existing and reports_file.exists()) else "w"
        combined.to_json(
            reports_file,
            orient="records",
            lines=True,
            force_ascii=False,
            mode=write_mode,
        )
        size_kb = reports_file.stat().st_size // 1024
        logger.success(
            f"Dataset written → '{reports_file}'  "
            f"({len(combined):,} new rows, {size_kb} KB)"
        )
    else:
        logger.info("No new rows to write.")

    n_skip = len(config_list) - len(frames) - len(failed)
    logger.info(
        f"\nSummary: {len(frames)} downloaded, {n_skip} skipped, {len(failed)} failed."
    )
    if failed:
        logger.warning(f"Failed configs: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
