"""
Data preparation for Binomial models: balance + aggregate + feature engineering.

Key difference from Bernoulli (lfm/): after balancing, trials are aggregated
by (model, environment, scaffold, level, verbosity, category, task_uid) into
k_success/n_trials counts.

Usage:
    python prepare_data.py --input data/overall_trace.csv --irt-dir data/irt_baseline --output results/prepared_data.csv
    python prepare_data.py --help
"""

from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler


def create_task_uid(df):
    """Create task_uid = task + level (same task at different levels = different UIDs)."""
    df["task_uid"] = df["task"] + "__" + df["level"].astype(str)
    logger.info(
        f"Created {df['task_uid'].nunique()} unique task_uids from {df['task'].nunique()} tasks"
    )
    return df


def balance_by_tasks(df, random_state=42):
    """Sample equal number of unique task_uids per environment."""
    uid_counts = df.groupby("environment")["task_uid"].nunique()
    min_uids = uid_counts.min()
    min_env = uid_counts.idxmin()

    logger.info(f"Balancing: {min_uids} task_uids per environment (min from {min_env})")

    rng = np.random.RandomState(random_state)
    parts = []

    for env in sorted(df["environment"].unique()):
        env_df = df[df["environment"] == env]
        env_uids = env_df["task_uid"].unique()

        if len(env_uids) <= min_uids:
            chosen_uids = env_uids
        else:
            chosen_uids = rng.choice(env_uids, size=min_uids, replace=False)

        sampled = env_df[env_df["task_uid"].isin(chosen_uids)]
        parts.append(sampled)
        logger.info(
            f"  {env:12s}: {len(chosen_uids):4d} task_uids, {len(sampled):5d} rows"
        )

    balanced = pd.concat(parts, ignore_index=True)
    logger.info(
        f"Balanced total: {balanced['task_uid'].nunique()} task_uids, {len(balanced):,} rows"
    )
    return balanced


def aggregate_trials(df):
    """
    Aggregate repeated trials into Binomial observations.

    Groups by (model, environment, scaffold, level, verbosity, category, task, task_uid)
    and computes k_success (sum of successes) and n_trials (count).
    """
    group_cols = [
        "model",
        "environment",
        "scaffold",
        "level",
        "verbosity",
        "category",
        "task",
        "task_uid",
    ]

    agg = (
        df.groupby(group_cols)
        .agg(
            k_success=("success", "sum"),
            n_trials=("success", "count"),
        )
        .reset_index()
    )

    agg["success_rate"] = agg["k_success"] / agg["n_trials"]

    logger.info("\n=== Trial Aggregation (Bernoulli -> Binomial) ===")
    logger.info(f"Rows before: {len(df):,} (one per trial)")
    logger.info(f"Rows after:  {len(agg):,} (one per combination)")
    logger.info(
        f"Trials per combination: min={agg['n_trials'].min()}, "
        f"median={agg['n_trials'].median():.0f}, max={agg['n_trials'].max()}"
    )
    logger.info(
        f"Overall success rate: {agg['k_success'].sum() / agg['n_trials'].sum():.1%}"
    )

    return agg


def merge_irt_abilities(df, irt_dir):
    """Merge IRT theta estimates from knowledge and reasoning QA files."""
    irt_dir = Path(irt_dir)
    knowledge_path = irt_dir / "knowledge_theta.csv"
    reasoning_path = irt_dir / "reasoning_theta.csv"

    if not knowledge_path.exists() or not reasoning_path.exists():
        logger.warning(f"IRT files not found in {irt_dir}, skipping ability merge")
        return df

    knowledge = pd.read_csv(knowledge_path).rename(
        columns={"theta_mean": "knowledge_theta", "theta_sd": "knowledge_sd"}
    )
    reasoning = pd.read_csv(reasoning_path).rename(
        columns={"theta_mean": "reasoning_theta", "theta_sd": "reasoning_sd"}
    )

    data_df = df.merge(
        knowledge[["model", "environment", "knowledge_theta", "knowledge_sd"]],
        on=["model", "environment"],
        how="left",
    )
    data_df = data_df.merge(
        reasoning[["model", "environment", "reasoning_theta", "reasoning_sd"]],
        on=["model", "environment"],
        how="left",
    )

    n_complete = (
        data_df[["knowledge_theta", "reasoning_theta"]].notna().all(axis=1).sum()
    )
    logger.info(f"Merged IRT abilities: {n_complete:,}/{len(data_df):,} complete")
    return data_df


def standardize_abilities(df):
    """Standardize IRT abilities to zero mean, unit variance."""
    if "knowledge_theta" not in df.columns:
        logger.warning("IRT abilities not available, skipping standardization")
        return df

    scaler = StandardScaler()
    df["knowledge_z"] = scaler.fit_transform(df[["knowledge_theta"]])
    df["reasoning_z"] = scaler.fit_transform(df[["reasoning_theta"]])
    df["level_num"] = df["level"].str.extract(r"(\d+)").astype(int)

    logger.info(
        f"Standardized: knowledge_z mean={df['knowledge_z'].mean():.3f}, "
        f"reasoning_z mean={df['reasoning_z'].mean():.3f}"
    )
    return df


def create_indices(df):
    """Create all integer indices needed by the models."""
    # Task ID
    task_uids = sorted(df["task_uid"].unique())
    df["task_id"] = df["task_uid"].map({t: i for i, t in enumerate(task_uids)})

    # Level
    if "level_num" not in df.columns:
        df["level_num"] = df["level"].str.extract(r"(\d+)").astype(int)
    df["level_id"] = df["level_num"] - 1

    # Categoricals
    for col, id_col in [
        ("model", "model_id"),
        ("scaffold", "scaffold_id"),
        ("environment", "environment_id"),
        ("verbosity", "verbosity_id"),
        ("category", "category_id"),
    ]:
        vals = sorted(df[col].unique())
        df[id_col] = df[col].map({v: i for i, v in enumerate(vals)})

    # Environment x Level
    df["env_level_combo"] = df["environment"] + "_L" + df["level_num"].astype(str)
    combos = sorted(df["env_level_combo"].unique())
    df["env_level_id"] = df["env_level_combo"].map({c: i for i, c in enumerate(combos)})

    # Environment x Scaffold
    df["env_scaffold_combo"] = df["environment"] + "_" + df["scaffold"]
    combos = sorted(df["env_scaffold_combo"].unique())
    df["env_scaffold_id"] = df["env_scaffold_combo"].map(
        {c: i for i, c in enumerate(combos)}
    )

    # Scaffold x Level
    df["scaffold_level_combo"] = df["scaffold"] + "_" + df["level"]
    combos = sorted(df["scaffold_level_combo"].unique())
    df["scaffold_level_id"] = df["scaffold_level_combo"].map(
        {c: i for i, c in enumerate(combos)}
    )

    logger.info(
        f"Indices created: {df['task_id'].nunique()} tasks, "
        f"{df['environment_id'].nunique()} envs, "
        f"{df['env_level_id'].nunique()} env-level combos"
    )
    return df


def main(
    input_path="data/overall_trace.csv",
    irt_dir="data/irt_baseline",
    output="results/prepared_data.csv",
    balance=True,
    seed=42,
):
    """
    Prepare data for Binomial model fitting.

    Args:
        input_path: Path to raw agent trial data (overall_trace.csv)
        irt_dir: Directory containing knowledge_theta.csv and reasoning_theta.csv
        output: Output path for prepared data
        balance: Whether to balance by task_uids across environments
        seed: Random seed for balancing
    """
    logger.info("Starting data preparation (Binomial)...")

    data_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(data_df):,} rows from {input_path}")

    data_df = create_task_uid(data_df)

    if balance:
        data_df = balance_by_tasks(data_df, random_state=seed)

    # Aggregate trials BEFORE merging IRT abilities
    data_df = aggregate_trials(data_df)

    data_df = merge_irt_abilities(data_df, irt_dir)
    data_df = standardize_abilities(data_df)
    data_df = create_indices(data_df)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(output_path, index=False)

    logger.info(f"\nPrepared data saved: {output_path}")
    logger.info(f"Shape: {data_df.shape}")
    logger.info(
        f"Success rate: {data_df['k_success'].sum() / data_df['n_trials'].sum():.1%}"
    )


if __name__ == "__main__":
    fire.Fire(main)
