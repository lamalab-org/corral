"""
Download per-message log-probability traces from HuggingFace Hub.

Dataset: jablonkagroup/corral-oss-trace-logprobs

Each HF config is a slice keyed by environment / level / item-type, e.g.:
    catalyst_tasks          → env=catalyst, level=1, type=tasks
    melting_level_1_tasks   → env=melting  → (md), level=1, type=tasks
    retrosynthesis_level_2_subtasks → env=retrosynthesis → (retro), level=2

NOTE: the ``environment`` column stored in HF is unreliable — for levelled
configs it contains only "tasks" or "subtasks".  The canonical environment
name is always derived from the *config name* using ``ENV_MAP`` from
``scripts/constants.py``.

Output file: ``results/data/logprobs.jsonl``

Added / overwritten columns in the output:
    environment   - canonical name (md, retro, catalyst, ...)
    hf_config     - original HF config string for traceability
    item_type     - "tasks" or "subtasks"

Usage:

```
# Download all configs → analysis/results/data/logprobs.jsonl
python analysis/download_logprobs_from_hf.py

# Custom output directory
python analysis/download_logprobs_from_hf.py --output_dir=my_local_data

# Skip if output already exists
python analysis/download_logprobs_from_hf.py --skip_existing
```
"""

import os
import sys
from pathlib import Path

import fire
import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from constants import ENV_MAP  # noqa: E402
from constants import HF_REPO_TRACE as HF_REPO  # noqa: E402

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

HF_TOKEN: str | None = os.getenv("HF_TOKEN")

_OUTPUT_FILENAME = "logprobs.jsonl"

# Environments expected in our analysis (from plot_config.py)
_EXPECTED_ENVS = {
    "afm",
    "catalyst",
    "md",
    "ml",
    "resistor",
    "retro",
    "spectra",
    "wetlab",
}


def load_local_dataset(output_dir: str | Path = "results/data") -> pd.DataFrame:
    """Load the local log-probability dataset written by :func:`main`.

    Args:
        output_dir: Directory previously populated by :func:`main`.

    Returns:
        DataFrame with all downloaded log-probability traces, or an empty
        DataFrame when the file does not exist yet.
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    logprobs_file = output_path / _OUTPUT_FILENAME
    if not logprobs_file.exists():
        logger.warning(f"No dataset found at '{logprobs_file}'.")
        return pd.DataFrame()
    logprobs_df = pd.read_json(logprobs_file, lines=True)
    logger.info(f"Loaded {len(logprobs_df):,} rows from '{logprobs_file}'.")
    return logprobs_df


def _parse_config_name(config_name: str) -> tuple[str, int | None, str]:
    """Parse a HF config name into ``(raw_env, level, item_type)``.

    Config name patterns:
        ``{env}_tasks``                 → level=None
        ``{env}_subtasks``              → level=None
        ``{env}_level_{N}_tasks``       → level=N
        ``{env}_level_{N}_subtasks``    → level=N

    ``env`` may itself contain underscores (e.g. ``surface_energy``).

    Examples
    --------
    >>> _parse_config_name("catalyst_tasks")
    ('catalyst', None, 'tasks')
    >>> _parse_config_name("ml_subtasks")
    ('ml', None, 'subtasks')
    >>> _parse_config_name("melting_level_1_tasks")
    ('melting', 1, 'tasks')
    >>> _parse_config_name("surface_energy_level_2_subtasks")
    ('surface_energy', 2, 'subtasks')
    >>> _parse_config_name("retrosynthesis_level_3_tasks")
    ('retrosynthesis', 3, 'tasks')
    """
    if "_level_" in config_name:
        env_part, rest = config_name.split("_level_", 1)
        # rest is like "1_tasks" or "2_subtasks"
        level_str, item_type = rest.split("_", 1)
        return env_part, int(level_str), item_type

    if config_name.endswith("_subtasks"):
        return config_name[: -len("_subtasks")], None, "subtasks"

    if config_name.endswith("_tasks"):
        return config_name[: -len("_tasks")], None, "tasks"

    return config_name, None, "unknown"


def _canonical_env(raw_env: str) -> str:
    """Map raw HF env name to canonical corral name via ``ENV_MAP``.

    Unknown names are kept as-is with a warning.
    """
    canonical = ENV_MAP.get(raw_env)
    if canonical is None:
        logger.warning(f"  ⚠ '{raw_env}' not in ENV_MAP — keeping as-is.")
        return raw_env
    return canonical


def _fetch_config(config_name: str) -> pd.DataFrame:
    """Download one HF config, fix the environment column, add metadata."""
    logger.info(f"  ↳ Downloading '{config_name}' …")
    ds = load_dataset(HF_REPO, name=config_name, token=HF_TOKEN, split="train")
    config_df = ds.to_pandas()

    raw_env, level, item_type = _parse_config_name(config_name)
    env = _canonical_env(raw_env)

    # The HF 'environment' column is unreliable (stores 'tasks'/'subtasks' for
    # levelled configs).  Overwrite it with the canonical name from the config.
    config_df["environment"] = env
    config_df["hf_config"] = config_name
    config_df["item_type"] = item_type

    # Back-fill level from config name when the column is missing / all-zero
    if level is not None and "level" in config_df.columns:
        config_df["level"] = config_df["level"].where(config_df["level"] != 0, level)

    return config_df


def main(
    output_dir: str = "results/data",
    skip_existing: bool = False,
) -> None:
    """Download all log-probability trace configs and consolidate locally.

    Downloads every config from ``jablonkagroup/corral-oss-trace-logprobs``,
    normalises environment names (melting/quenching/surface_energy → md,
    retrosynthesis → retro, …), and writes a single JSONL file.

    Args:
        output_dir:
            Directory where the output file is written.  Created if absent.
            Default: ``results/data/`` (relative to ``analysis/``).
        skip_existing:
            When *True* and the output file already exists the script exits
            immediately without re-downloading.
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    logprobs_file = output_path / _OUTPUT_FILENAME

    if skip_existing and logprobs_file.exists():
        logger.info(f"Output '{logprobs_file}' already exists — skipping download.")
        return

    logger.info(f"Fetching config list from '{HF_REPO}' …")
    available: list[str] = get_dataset_config_names(HF_REPO, token=HF_TOKEN)
    logger.info(f"  → {len(available)} config(s): {sorted(available)}")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for config_name in sorted(available):
        raw_env, level, item_type = _parse_config_name(config_name)
        env = _canonical_env(raw_env)
        try:
            config_df = _fetch_config(config_name)
            frames.append(config_df)
            logger.success(
                f"  ✓ '{config_name}'  env={env}  level={level}"
                f"  type={item_type}  {len(config_df):,} rows"
            )
        except Exception as exc:
            logger.error(f"  ✗ '{config_name}' failed: {exc}")
            failed.append(config_name)

    if not frames:
        logger.warning("No data downloaded.")
        if failed:
            raise SystemExit(1)
        return

    combined = pd.concat(frames, ignore_index=True)

    envs_present = sorted(combined["environment"].unique())
    missing_envs = _EXPECTED_ENVS - set(envs_present)
    logger.info(f"Environments in dataset : {envs_present}")
    if missing_envs:
        logger.warning(f"Environments NOT in HF dataset: {sorted(missing_envs)}")

    combined.to_json(
        logprobs_file,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    size_kb = logprobs_file.stat().st_size // 1024
    logger.success(
        f"Dataset written → '{logprobs_file}'  ({len(combined):,} rows, {size_kb} KB)"
    )

    summary = (
        combined.groupby(["environment", "item_type"])
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["environment", "item_type"])
    )
    logger.info(
        f"\nRow counts per environment / item_type:\n{summary.to_string(index=False)}"
    )

    if failed:
        logger.warning(f"Failed configs: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
