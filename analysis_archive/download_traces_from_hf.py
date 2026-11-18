"""
Download trace configs from HuggingFace Hub into a local dataset.

Without arguments every config is merged into one ``traces.jsonl`` file:

    import pandas as pd
    df = pd.read_json("analysis/results/data/traces.jsonl", lines=True)
    df[df["env"] == "afm"]

    from analysis.download_traces_from_hf import load_local_dataset
    df = load_local_dataset()

When specific configs are requested each one is saved under its own name:

    df = pd.read_json("analysis/results/data/claude_sonnet_45-afm-level_1-tasks-ReActAgent-brief-traces.jsonl", lines=True)

Usage:

```
# Download all configs → analysis/results/data/traces.jsonl
python analysis/download_traces_from_hf.py

# Download only task configs (skip subtasks)
python analysis/download_traces_from_hf.py --tasks_only

# Download specific configs → one file per config
python analysis/download_traces_from_hf.py --configs=claude_sonnet_45-afm-level_1-tasks-ReActAgent-brief-traces

# Custom output directory
python analysis/download_traces_from_hf.py --output_dir=my_local_data

# Skip configs already on disk
python analysis/download_traces_from_hf.py --skip_existing
```
"""

import json
import os
import re
import sys
from pathlib import Path

import fire
import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from constants import HF_REPO_TRACES as HF_REPO  # noqa: E402

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

HF_TOKEN: str | None = os.getenv("HF_TOKEN")

_OUTPUT_FILENAME = "traces.jsonl"

# Config names have the shape:
#   {model}-{env}-{level}-{category}-{agent}-{verbosity}-traces
# where category is one of: tasks, task, subtasks, subtask
_SUBTASK_PATTERN = re.compile(r"-subtasks?-")


def load_local_dataset(output_dir: str | Path = "results/data") -> pd.DataFrame:
    """Load the local combined traces dataset written by :func:`main`.

    Args:
        output_dir: Directory previously populated by :func:`main`.

    Returns:
        DataFrame with all downloaded traces, or an empty DataFrame
        when the file does not exist yet.
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    traces_file = output_path / _OUTPUT_FILENAME
    if not traces_file.exists():
        logger.warning(f"No dataset found at '{traces_file}'.")
        return pd.DataFrame()
    traces_df = pd.read_json(traces_file, lines=True)
    logger.info(f"Loaded {len(traces_df):,} rows from '{traces_file}'.")
    return traces_df


def _list_configs(
    configs: list[str] | None,
    tasks_only: bool = False,
) -> list[str]:
    """Return configs to download, validating any user-supplied names."""
    logger.info(f"Fetching config list from '{HF_REPO}' …")
    available: list[str] = get_dataset_config_names(HF_REPO, token=HF_TOKEN)
    logger.info(f"  → {len(available)} config(s) found")

    if configs:
        available_set = set(available)
        unknown = [c for c in configs if c not in available_set]
        if unknown:
            raise SystemExit(
                f"ERROR: the following configs are not present in '{HF_REPO}': "
                f"{unknown}\n"
                f"Available configs: {sorted(available_set)}"
            )
        selected = sorted(configs)
    else:
        selected = sorted(available)

    if tasks_only:
        selected = [c for c in selected if not _SUBTASK_PATTERN.search(c)]
        logger.info(f"  → {len(selected)} task-only config(s) after filtering subtasks")

    return selected


def _fetch_config(config_name: str) -> pd.DataFrame:
    """Download one Hub config and return it as a DataFrame."""
    logger.info(f"  ↳ Downloading config '{config_name}' …")
    ds = load_dataset(HF_REPO, name=config_name, token=HF_TOKEN, split="train")
    config_df = ds.to_pandas()

    # Decode JSON-encoded columns if present.
    for col in ("messages", "token_usage", "tools_used", "error_types"):
        if col in config_df.columns:
            config_df[col] = config_df[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )
    return config_df


def main(
    output_dir: str = "results/data",
    skip_existing: bool = False,
    tasks_only: bool = False,
    configs: list[str] | None = None,
) -> None:
    """Download HF trace configs and save them locally.

    Args:
        output_dir:
            Directory where output files are written.  Created if absent.
            Default: ``results/data/`` (relative to ``analysis/``).
        skip_existing:
            Full-download mode: skip (model, env, level, category) tuples
            already in ``traces.jsonl``.  Per-config mode: skip configs
            whose file already exists.
        tasks_only:
            When True, skip subtask configs and download only task-level
            traces.
        configs:
            Config names to download.  When omitted all available configs
            are merged into ``traces.jsonl``.  When provided each config
            is written to its own ``<config_name>.jsonl``.
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(configs, str):
        configs = [configs]

    config_list = _list_configs(configs or None, tasks_only=tasks_only)
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
    """Write each config to its own ``<config_name>.jsonl`` file."""
    failed: list[str] = []
    for cname in config_list:
        dest = output_path / f"{cname}.jsonl"
        if skip_existing and dest.exists():
            logger.info(f"  ↳ '{cname}' already on disk — skipping.")
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
    """Merge all configs into a single ``traces.jsonl``."""
    traces_file = output_path / _OUTPUT_FILENAME

    existing_keys: set[tuple[str, str, str, str]] = set()
    if skip_existing and traces_file.exists():
        existing_df = pd.read_json(traces_file, lines=True)
        existing_keys = set(
            zip(
                existing_df["model"],
                existing_df["env"],
                existing_df["level"],
                existing_df["category"],
                strict=False,
            )
        )
        logger.info(
            f"Found {len(existing_keys)} existing (model, env, level, category) "
            f"tuple(s) on disk."
        )

    logger.info(f"Will download {len(config_list)} config(s) → '{traces_file}'")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for cname in config_list:
        try:
            config_df = _fetch_config(cname)
            if existing_keys:
                config_df = config_df[
                    ~config_df.apply(
                        lambda r: (
                            r.get("model", ""),
                            r.get("env", ""),
                            r.get("level", ""),
                            r.get("category", ""),
                        )
                        in existing_keys,
                        axis=1,
                    )
                ]
                if config_df.empty:
                    logger.info(f"  ↳ '{cname}' already fully present — skipping.")
                    continue
            frames.append(config_df)
            logger.success(f"  ✓ '{cname}'  {len(config_df):,} rows")
        except Exception as exc:
            logger.error(f"  ✗ '{cname}' failed: {exc}")
            failed.append(cname)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        write_mode = "a" if (skip_existing and traces_file.exists()) else "w"
        combined.to_json(
            traces_file,
            orient="records",
            lines=True,
            force_ascii=False,
            mode=write_mode,
        )
        size_kb = traces_file.stat().st_size // 1024
        logger.success(
            f"Dataset written → '{traces_file}'  "
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
