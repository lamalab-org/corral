"""
Download individual QA reports from HuggingFace Hub for IRT analysis.

Downloads question-level data from jablonkagroup/corral-QAs-reports and transforms
it into IRT-ready format: knowledge_qa.csv and reasoning_qa.csv.

This is separate from download_qa_topic_reports_from_hf.py which downloads
aggregated topic-level summaries.

Output format:
    model_id: model name (Claude-4.5, gpt-4o, oss)
    env_id: environment name
    item_id: unique question UUID
    correct: 0 or 1 binary score
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = "jablonkagroup/corral-QAs-reports"

# Model name mapping to match IRT script expectations
MODEL_NAME_MAP = {
    "claude": "Claude-4.5",
    "gpt": "gpt-4o",
    "gpt_oss": "oss",
}

# Environment name mapping
ENV_NAME_MAP = {
    "afm": "afm",
    "catalyst": "catalyst",
    "md": "corral_md",
    "ml": "ml",
    "resistor": "resistor",
    "retro": "retro",
    "spectra": "spectra",
    "wetlab": "wetlab",
}


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

    # Normalize environment names
    if env == "corral_md":
        env = "md"  # Normalize to "md"

    return env, qa_type, model


def extract_score_from_row(row: pd.Series) -> int:
    """Extract binary score (0 or 1) from a question row.

    Args:
        row: DataFrame row with 'results' column

    Returns:
        1 if correct (all_correct == 1), 0 otherwise
    """
    import numpy as np

    try:
        results = row["results"]

        # Results can be a list or numpy array - check for length
        if not hasattr(results, "__len__") or len(results) == 0:
            return 0

        result = results[0]

        # Handle cases where result is not a dict
        if not isinstance(result, dict):
            return 0

        metrics = result.get("metrics", {})

        # Use all_correct metric (binary indicator that all parts are correct)
        score = metrics.get("all_correct")
        if score is None or (isinstance(score, float) and np.isnan(score)):
            return 0

        # Convert to binary (all_correct is already 0 or 1)
        return int(score)
    except (TypeError, ValueError, KeyError, AttributeError):
        return 0


def download_and_process_config(
    dirname: str, parquet_file: str
) -> pd.DataFrame | None:
    """Download and process a single config file.

    Args:
        dirname: Config directory name (e.g., "afm_qa_claude")
        parquet_file: Full path to parquet file in repo

    Returns:
        DataFrame with columns: model_id, env_id, item_id, correct
        or None if parsing fails
    """
    parsed = parse_config_name(dirname)
    if parsed is None:
        logger.warning(f"Cannot parse config name: {dirname}")
        return None

    env, qa_type, model = parsed

    # Map names
    model_mapped = MODEL_NAME_MAP.get(model, model)
    env_mapped = ENV_NAME_MAP.get(env, env)

    try:
        logger.info(
            f"  Downloading {dirname} (env={env_mapped}, qa_type={qa_type}, model={model_mapped})"
        )

        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=parquet_file,
            repo_type="dataset",
            token=HF_TOKEN,
        )

        df = pd.read_parquet(local_path)

        # Extract question-level data
        records = []
        for _, row in df.iterrows():
            # Use UUID as item_id for stable unique identification
            item_id = row.get("uuid", row.get("name", "unknown"))
            correct = extract_score_from_row(row)

            records.append(
                {
                    "model_id": model_mapped,
                    "env_id": env_mapped,
                    "item_id": item_id,
                    "correct": correct,
                }
            )

        result_df = pd.DataFrame(records)
        logger.success(
            f"  ✓ {dirname}: {len(result_df)} questions, "
            f"avg score: {result_df['correct'].mean():.3f}"
        )
        return result_df

    except Exception as exc:
        logger.error(f"  ✗ {dirname} failed: {exc}")
        return None


def main():
    """Download QA reports and create IRT-ready datasets."""

    output_dir = Path("results/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching file list from '{HF_REPO}'")
    api = HfApi()
    files = api.list_repo_files(HF_REPO, token=HF_TOKEN, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    logger.info(f"Found {len(parquet_files)} parquet files")

    knowledge_dfs = []
    reasoning_dfs = []
    failed = []

    for pf in parquet_files:
        dirname = pf.split("/")[0]
        result_df = download_and_process_config(dirname, pf)

        if result_df is None:
            failed.append(dirname)
            continue

        # Determine QA type from dirname
        if "reasoning_qa" in dirname:
            reasoning_dfs.append(result_df)
        else:
            knowledge_dfs.append(result_df)

    # Combine and save
    if knowledge_dfs:
        knowledge_qa = pd.concat(knowledge_dfs, ignore_index=True)
        output_path = output_dir / "knowledge_qa.csv"
        knowledge_qa.to_csv(output_path, index=False)
        logger.success(
            f"\nSaved Knowledge QA to {output_path} ({len(knowledge_qa)} rows)"
        )

        logger.info("\nKnowledge QA Summary:")
        logger.info(f"  Total responses: {len(knowledge_qa)}")
        logger.info(f"  Unique questions: {knowledge_qa['item_id'].nunique()}")
        logger.info(f"  Overall accuracy: {knowledge_qa['correct'].mean():.3f}")
        logger.info("  By model:")
        for model in sorted(knowledge_qa["model_id"].unique()):
            model_acc = knowledge_qa[knowledge_qa["model_id"] == model]["correct"].mean()
            logger.info(f"    {model}: {model_acc:.3f}")

    if reasoning_dfs:
        reasoning_qa = pd.concat(reasoning_dfs, ignore_index=True)
        output_path = output_dir / "reasoning_qa.csv"
        reasoning_qa.to_csv(output_path, index=False)
        logger.success(
            f"\nSaved Reasoning QA to {output_path} ({len(reasoning_qa)} rows)"
        )

        logger.info("\nReasoning QA Summary:")
        logger.info(f"  Total responses: {len(reasoning_qa)}")
        logger.info(f"  Unique questions: {reasoning_qa['item_id'].nunique()}")
        logger.info(f"  Overall accuracy: {reasoning_qa['correct'].mean():.3f}")
        logger.info("  By model:")
        for model in sorted(reasoning_qa["model_id"].unique()):
            model_acc = reasoning_qa[reasoning_qa["model_id"] == model]["correct"].mean()
            logger.info(f"    {model}: {model_acc:.3f}")

    if failed:
        logger.warning(f"\nFailed configs: {failed}")


if __name__ == "__main__":
    main()
