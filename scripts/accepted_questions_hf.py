"""Build and publish accepted-question traces to the Hugging Face Hub.

This script walks the `accepted_questions/` directory tree, reads every
trace JSON file it finds, assembles a tabular dataset, and uploads grouped
subsets to the Hub using the same config/naming policy as `trace_hf.py`.
"""

import json
import tempfile
from pathlib import Path

import fire
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
from loguru import logger

DATASET_NAME = "jablonkagroup/questions4manual_annotation"


def load_json(path):
    path = Path(path)
    with path.open() as f:
        return json.load(f)


def build_rows(root: str):
    """Walk *root* and yield one dataset row per trace JSON file.

    Expected layout::

        {root}/{model_dir}/{env}/{level}/tasks/{agent_logs_dir}/*.json

    Metadata is extracted both from the directory hierarchy and from the
    JSON payload itself.
    """

    root = Path(root)
    rows = []

    for json_path in sorted(root.rglob("*.json")):
        try:
            rel = json_path.relative_to(root)
        except ValueError:
            continue

        # Expected relative parts:
        #   model_dir / env / level / "tasks" / agent_logs_dir / file.json
        parts = rel.parts
        if len(parts) < 6:
            logger.warning(
                f"Skipping {json_path}: unexpected depth ({len(parts)} parts)"
            )
            continue

        model_dir = parts[0]
        env = parts[1]
        level = parts[2]
        category = parts[3]  # typically "tasks"

        # Normalise model name (e.g. gpt-4o -> gpt_4o)
        model = model_dir.replace("-", "_")

        trace = load_json(json_path)

        task_id = trace.get("task_id")
        agent = trace.get("agent")
        verbosity = trace.get("tool_verbosity")
        messages = trace.get("messages")
        model_version = trace.get("model")
        timestamp = trace.get("timestamp")

        rows.append(
            {
                "model": model,
                "env": env,
                "level": level,
                "category": category,
                "agent": agent,
                "verbosity": verbosity,
                "task_name": task_id,
                "messages": messages,
                "trace_file": str(json_path),
                "model_version": model_version,
                "timestamp": timestamp,
            }
        )

    return rows


def push_to_hub(trials_df: pd.DataFrame, dataset_name: str):
    """Upload grouped trace subsets to the Hugging Face Hub.

    Uses the same config-naming policy as `trace_hf.push_trials_to_hub`:
    one parquet file per (model, env, level, category, agent, verbosity)
    combination.
    """

    api = HfApi()

    api.create_repo(
        repo_id=dataset_name,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    group_cols = ["model", "env", "level", "category", "agent", "verbosity"]

    for subset_keys, subset_df in trials_df.groupby(group_cols):
        model, env, level, category, agent, verbosity = subset_keys

        subset_name = f"{model}-{env}-{level}-{category}-{agent}-{verbosity}-traces"

        logger.info(f"\nUploading subset: {subset_name}")

        dataset = Dataset.from_pandas(subset_df.reset_index(drop=True))

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(str(parquet_path))
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=f"{subset_name}/train-00000-of-00001.parquet",
                repo_id=dataset_name,
                repo_type="dataset",
            )


def main(root: str = "accepted_questions"):
    """Build and push accepted-question traces.

    Args:
        root: Path to the `accepted_questions` directory.
    """

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    rows = build_rows(root)

    if not rows:
        logger.warning("No trace files found - nothing to upload.")
        return

    trials_df = pd.DataFrame(rows)

    if trials_df.isna().any().any():
        nan_counts = trials_df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        raise ValueError(f"NaNs detected in trials dataframe:\n{nan_cols}")

    logger.info(
        f"Collected {len(trials_df)} traces across "
        f"{trials_df.groupby(['model', 'env', 'level', 'category', 'agent', 'verbosity']).ngroups} configs"
    )

    push_to_hub(trials_df, DATASET_NAME)


if __name__ == "__main__":
    fire.Fire(main)
