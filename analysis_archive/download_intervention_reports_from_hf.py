"""Download intervention benchmark reports from HuggingFace Hub.

Fetches all environment configs from the intervention reports dataset
and merges them into a single ``intervention_reports.jsonl`` file.

Downloads parquet files directly via HfApi to avoid schema-merge issues
when configs have slightly different dtypes.

Usage::

    python download_intervention_reports_from_hf.py
    python download_intervention_reports_from_hf.py --skip_existing
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import fire
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from constants import HF_REPO_INTERVENTION_REPORTS as HF_REPO  # noqa: E402

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

HF_TOKEN: str | None = os.getenv("HF_TOKEN")

_OUTPUT_FILENAME = "intervention_reports.jsonl"

CONFIGS = ["catalyst", "md", "ml", "resistor", "retrosynthesis", "spectra", "wetlab"]


def _fetch_config(api: HfApi, config_name: str) -> pd.DataFrame:
    """Download one config's parquet file and return as DataFrame."""
    logger.info(f"  -> Downloading config '{config_name}' ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = api.hf_hub_download(
            HF_REPO,
            f"{config_name}/train-00000-of-00001.parquet",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=tmpdir,
        )
        config_df = pd.read_parquet(local_path)

    if "Task Results" in config_df.columns:
        config_df["Task Results"] = config_df["Task Results"].apply(
            lambda v: json.loads(v) if isinstance(v, str) else v
        )
    return config_df


def main(
    output_dir: str = "results/data",
    skip_existing: bool = False,
) -> None:
    """Download intervention reports from HF and save locally.

    Args:
        output_dir: Directory for output files (relative to analysis/).
        skip_existing: Skip download if output file already exists.
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    reports_file = output_path / _OUTPUT_FILENAME
    if skip_existing and reports_file.exists():
        logger.info(f"'{reports_file}' already exists -- skipping.")
        return

    api = HfApi(token=HF_TOKEN)
    logger.info(f"Will download {len(CONFIGS)} config(s) -> '{reports_file}'")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for cname in CONFIGS:
        try:
            config_df = _fetch_config(api, cname)
            frames.append(config_df)
            logger.success(f"  OK '{cname}'  {len(config_df):,} rows")
        except Exception as exc:
            logger.error(f"  FAIL '{cname}': {exc}")
            failed.append(cname)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_json(reports_file, orient="records", lines=True, force_ascii=False)
        size_kb = reports_file.stat().st_size // 1024
        logger.success(
            f"Dataset written -> '{reports_file}'  "
            f"({len(combined):,} rows, {size_kb} KB)"
        )
    else:
        logger.info("No data downloaded.")

    if failed:
        logger.warning(f"Failed configs: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
