"""
Push OpenAlex AI-scientists CSV results to HuggingFace Hub.

Uploads every CSV file found in
`analysis/openalex/openalex_ai_scientists_output/` (excluding the
`checkpoints/` subdirectory) as a separate config to the HF dataset
`jablonkagroup/rise_ai_scientists`.  Each CSV becomes a named config
whose name is derived from the file stem (e.g. `yearly_counts`).

Usage
-----
    python scripts/push_openalex_to_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

HF_REPO = "jablonkagroup/rise_ai_scientists"
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "analysis"
    / "openalex"
    / "openalex_ai_scientists_output"
)


def push_csvs() -> None:
    """Read every CSV in *DATA_DIR* and push it as a config to *HF_REPO*."""
    hf_token = os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)

    # Ensure the repo exists (dataset type)
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in {}", DATA_DIR)
        return

    for csv_path in csv_files:
        config_name = csv_path.stem  # e.g. "yearly_counts"
        logger.info("Pushing {} → {}/{}", csv_path.name, HF_REPO, config_name)

        ds = Dataset.from_csv(str(csv_path))
        ds.push_to_hub(
            HF_REPO,
            config_name=config_name,
            token=hf_token,
        )
        logger.success("  ✓ {} ({} rows)", config_name, len(ds))

    logger.info("All CSVs pushed to {}", HF_REPO)


if __name__ == "__main__":
    push_csvs()
