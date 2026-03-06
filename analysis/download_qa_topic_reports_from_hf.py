"""
Download QA topic-report configs from HuggingFace Hub into a local dataset.

This script downloads QA evaluation scores for each model across different environments.
The QA topic reports contain aggregated question-answering performance metrics.

The merged dataset is saved as `qa_topic_reports.jsonl` file with `model`, `env`, and
`qa_type` columns for filtering:

    import pandas as pd
    df = pd.read_json("analysis/results/data/qa_topic_reports.jsonl", lines=True)
    df[df["model"] == "claude"]

    from analysis.download_qa_topic_reports_from_hf import load_local_dataset
    df = load_local_dataset()

Usage:

```
# Download all configs → analysis/results/data/qa_topic_reports.jsonl
python analysis/download_qa_topic_reports_from_hf.py

# Custom output directory
python analysis/download_qa_topic_reports_from_hf.py --output_dir=my_local_data
```
"""

import os
import sys
from pathlib import Path

import fire
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from constants import HF_REPO_QA_TOPIC as HF_REPO  # noqa: E402

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

HF_TOKEN: str | None = os.getenv("HF_TOKEN")

_OUTPUT_FILENAME = "qa_topic_reports.jsonl"

# Environment name normalization to match reports dataset
# QA data has some names that differ from the main reports
ENV_NAME_NORMALIZATION = {
    "corral_md": "md",  # QA has "corral_md", reports has "md"
}


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


def parse_config_name(dirname: str) -> tuple[str, str, str] | None:
    """Parse env_qa_model from directory name.

    Examples:
        "md_qa_claude" -> ("md", "qa", "claude")
        "corral_md_reasoning_qa_gpt" -> ("corral_md", "reasoning_qa", "gpt")
        "afm_qa_gpt_oss" -> ("afm", "qa", "gpt_oss")
    """
    parts = dirname.split("_")

    # Find where qa_type starts (either "qa" or "reasoning")
    if "reasoning" in parts:
        qa_idx = parts.index("reasoning")
        qa_type = "reasoning_qa"
    elif "qa" in parts:
        qa_idx = parts.index("qa")
        qa_type = "qa"
    else:
        return None

    # Everything before qa_type is env
    env = "_".join(parts[:qa_idx])
    # Everything after qa_type is model
    model = "_".join(parts[qa_idx + (2 if qa_type == "reasoning_qa" else 1) :])

    # Apply environment name normalization
    env = ENV_NAME_NORMALIZATION.get(env, env)

    return env, qa_type, model


def main(
    output_dir: str = "results/data",
) -> None:
    """Download HF QA topic report configs and save them locally.

    Downloads all parquet files from the dataset repository and combines them into
    a single JSONL file with normalized environment names.

    Args:
        output_dir:
            Directory where output files are written. Created if absent.
            Default: `results/data/` (relative to `analysis/`).
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching file list from '{HF_REPO}' …")
    api = HfApi()
    files = api.list_repo_files(HF_REPO, token=HF_TOKEN, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    logger.info(f"  → {len(parquet_files)} parquet file(s) found")

    dfs = []
    failed = []

    for pf in parquet_files:
        # Extract directory name (config name)
        dirname = pf.split("/")[0]
        parsed = parse_config_name(dirname)

        if parsed is None:
            logger.warning(f"  ⚠ Skipping '{pf}' - cannot parse config name")
            continue

        env, qa_type, model = parsed

        try:
            logger.info(
                f"  ↳ Downloading '{dirname}' (env={env}, qa_type={qa_type}, model={model}) …"
            )
            local_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=pf,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            df = pd.read_parquet(local_path)  # noqa: PD901

            # Add metadata columns
            df["env"] = env
            df["qa_type"] = qa_type
            df["model"] = model

            dfs.append(df)
            logger.success(f"  ✓ '{dirname}'  {len(df):,} rows")
        except Exception as exc:
            logger.error(f"  ✗ '{dirname}' failed: {exc}")
            failed.append(dirname)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        output_file = output_path / _OUTPUT_FILENAME
        combined.to_json(
            output_file,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        size_kb = output_file.stat().st_size // 1024
        logger.success(
            f"Dataset written → '{output_file}'  "
            f"({len(combined):,} rows, {size_kb} KB)"
        )
    else:
        logger.warning("No data downloaded")

    if failed:
        logger.warning(f"Failed configs: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
