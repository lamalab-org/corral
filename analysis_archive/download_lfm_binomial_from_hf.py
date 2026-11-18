"""
Download LFM-Binomial results from HuggingFace Hub.

Downloads pre-fitted model results (traces, thetas, comparisons) from
jablonkagroup/corral_lfm_binomial_results into results/lfm-binomial/.

Usage:
    # Download only files needed for plotting (default)
    python download_lfm_binomial_from_hf.py

    # Download everything including all traces (~2 GB)
    python download_lfm_binomial_from_hf.py --include_traces
"""

import os
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

HF_TOKEN: str | None = os.getenv("HF_TOKEN")
HF_REPO = "jablonkagroup/corral_lfm_binomial_results"

# Files needed for plotting (small, ~1 MB total)
PLOT_FILES = [
    "irt_baseline/knowledge_theta.csv",
    "irt_baseline/reasoning_theta.csv",
    "model_comparison.csv",
    "prepared_data.csv",
]


def main(
    output_dir: str = "results/lfm-binomial",
    include_traces: bool = False,
    skip_existing: bool = True,
) -> None:
    """Download LFM-Binomial results from HuggingFace Hub.

    Args:
        output_dir: Local directory for downloaded files.
        include_traces: If True, download all files including large .nc traces (~2 GB).
                       If False (default), download only files needed for plotting.
        skip_existing: Skip files that already exist locally.
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = _SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    if include_traces:
        files_to_download = api.list_repo_files(
            HF_REPO, token=HF_TOKEN, repo_type="dataset"
        )
        files_to_download = [f for f in files_to_download if f != ".gitattributes"]
        logger.info(f"Downloading all {len(files_to_download)} files from '{HF_REPO}'")
    else:
        files_to_download = PLOT_FILES
        logger.info(
            f"Downloading {len(files_to_download)} plot-essential files from '{HF_REPO}'"
        )

    downloaded = 0
    skipped = 0
    failed = []

    for filename in files_to_download:
        dest = output_path / filename
        if skip_existing and dest.exists():
            logger.debug(f"  Skipping (exists): {filename}")
            skipped += 1
            continue

        try:
            logger.info(f"  Downloading: {filename}")
            local_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=output_path,
            )
            downloaded += 1
            logger.success(f"  -> {local_path}")
        except Exception as exc:
            logger.error(f"  Failed: {filename}: {exc}")
            failed.append(filename)

    logger.info(
        f"\nSummary: {downloaded} downloaded, {skipped} skipped, {len(failed)} failed."
    )
    if failed:
        logger.warning(f"Failed files: {failed}")
        raise SystemExit(1)

    logger.success(f"LFM-Binomial results ready at {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
