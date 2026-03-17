"""
Data Balancing Script

Creates balanced dataset with equal representation across:
- All 7 environments
- Both task categories (task/subtask) within each environment

Addresses severe imbalance in raw data (Spectra 65%, smallest envs ~2%)
"""

from pathlib import Path

import pandas as pd
from loguru import logger


def load_data():
    """Load agent trials and IRT abilities"""
    data_dir = Path("results/data")
    irt_dir = Path("results/irt_baseline")

    # Load agent trials
    df = pd.read_csv(data_dir / "overall_trace.csv")
    logger.info(f"Loaded {len(df):,} total agent trials")

    # Load IRT abilities (organized by model AND environment)
    knowledge = pd.read_csv(irt_dir / "knowledge_theta.csv")
    reasoning = pd.read_csv(irt_dir / "reasoning_theta.csv")

    # Rename theta_mean to knowledge_theta/reasoning_theta
    knowledge = knowledge.rename(
        columns={"theta_mean": "knowledge_theta", "theta_sd": "knowledge_sd"}
    )
    reasoning = reasoning.rename(
        columns={"theta_mean": "reasoning_theta", "theta_sd": "reasoning_sd"}
    )

    # Merge abilities (on both model AND environment)
    df = df.merge(
        knowledge[["model", "environment", "knowledge_theta", "knowledge_sd"]],
        on=["model", "environment"],
        how="left",
    )
    df = df.merge(
        reasoning[["model", "environment", "reasoning_theta", "reasoning_sd"]],
        on=["model", "environment"],
        how="left",
    )

    logger.info(
        f"Merged IRT abilities: {df[['knowledge_theta', 'reasoning_theta']].notna().all(axis=1).sum():,} complete"
    )

    return df


def analyze_imbalance(df):
    """Report current data distribution"""
    logger.info("\n=== Current Data Distribution ===")

    # By environment
    env_counts = df["environment"].value_counts().sort_index()
    logger.info("\nEnvironment counts:")
    for env, count in env_counts.items():
        pct = 100 * count / len(df)
        logger.info(f"  {env:12s}: {count:6,} ({pct:5.1f}%)")

    # By environment × category
    logger.info("\nEnvironment × Category:")
    crosstab = pd.crosstab(df["environment"], df["category"], margins=True)
    logger.info(f"\n{crosstab}")

    # By environment × level
    logger.info("\nEnvironment × Level:")
    level_crosstab = pd.crosstab(df["environment"], df["level"], margins=True)
    logger.info(f"\n{level_crosstab}")

    return env_counts, crosstab


def determine_sample_size(df):
    """
    Determine balanced sample size per (environment, category) combination

    Strategy:
    - Find minimum available samples across all (env, category) groups
    - Target: At least 800-1000 per group, but respect data limits
    - Ensure sufficient representation of all environments
    """
    group_sizes = df.groupby(["environment", "category"]).size()

    logger.info("\n=== Available samples per (Environment, Category) ===")
    for (env, cat), count in group_sizes.sort_index().items():
        logger.info(f"  {env:12s} × {cat:7s}: {count:6,}")

    min_available = group_sizes.min()
    logger.info(f"\nMinimum available: {min_available:,}")

    # Target 1000 per group, but cap at minimum available
    target_per_group = min(1000, min_available)

    # Ensure we get at least 800 if possible
    if min_available >= 800:
        target_per_group = min(1000, min_available)
    else:
        target_per_group = min_available
        logger.warning(f"Some groups have < 800 samples. Using {target_per_group}")

    n_groups = len(group_sizes)
    total_expected = target_per_group * n_groups

    logger.info("\n=== Balanced Sampling Strategy ===")
    logger.info(f"Target per group: {target_per_group:,}")
    logger.info(f"Number of groups: {n_groups}")
    logger.info(f"Expected total:   {total_expected:,}")

    return target_per_group


def balance_dataset(df, samples_per_group):
    """
    Create balanced dataset with equal samples per (environment, category)

    Args:
        df: Full dataset
        samples_per_group: Number of samples to draw from each group

    Returns:
        Balanced DataFrame
    """
    balanced_data = []

    for (env, cat), group_df in df.groupby(["environment", "category"]):
        if len(group_df) < samples_per_group:
            logger.warning(f"{env} × {cat}: Only {len(group_df)} available, using all")
            sampled = group_df
        else:
            sampled = group_df.sample(n=samples_per_group, random_state=42)

        balanced_data.append(sampled)
        logger.debug(f"  {env:12s} × {cat:7s}: sampled {len(sampled):,}")

    balanced_df = pd.concat(balanced_data, ignore_index=True)

    logger.info("\n=== Balanced Dataset Created ===")
    logger.info(f"Total samples: {len(balanced_df):,}")

    return balanced_df


def verify_balance(df):
    """Verify balanced distribution"""
    logger.info("\n=== Balanced Distribution Verification ===")

    # By environment
    env_counts = df["environment"].value_counts().sort_index()
    logger.info("\nEnvironment counts (should be equal):")
    for env, count in env_counts.items():
        pct = 100 * count / len(df)
        logger.info(f"  {env:12s}: {count:6,} ({pct:5.1f}%)")

    # By environment × category (should be exactly equal)
    logger.info("\nEnvironment × Category (should be equal per cell):")
    crosstab = pd.crosstab(df["environment"], df["category"])
    logger.info(f"\n{crosstab}")

    # Success rates should still vary (this is real signal, not imbalance artifact)
    logger.info("\nSuccess rates by environment (real signal, should vary):")
    success_rates = (
        df.groupby("environment")["success"].mean().sort_values(ascending=False)
    )
    for env, rate in success_rates.items():
        logger.info(f"  {env:12s}: {rate:.1%}")


def save_balanced_data(df, output_path="results/balanced_data.csv"):
    """Save balanced dataset"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"\nBalanced data saved to: {output_path}")
    logger.info(f"Shape: {df.shape}")

    return output_path


def main():
    """Main execution"""
    logger.info("Starting data balancing...")

    # Load data
    df = load_data()

    # Analyze current imbalance
    analyze_imbalance(df)

    # Determine balanced sample size
    samples_per_group = determine_sample_size(df)

    # Create balanced dataset
    balanced_df = balance_dataset(df, samples_per_group)

    # Verify balance
    verify_balance(balanced_df)

    # Save
    output_path = save_balanced_data(balanced_df)

    logger.success(f"✓ Balanced dataset created: {output_path}")

    return balanced_df


if __name__ == "__main__":
    main()
