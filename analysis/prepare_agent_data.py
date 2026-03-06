"""
Transform benchmark reports into agent trace data for IRT analysis.

Reads reports.jsonl and extracts trial-level success data to create
overall_trace.csv needed for the latent factor modeling.

Output format:
    model: model name (Claude 4.5, GPT-4o, OSS)
    environment: environment name
    scaffold: agent type (ReAct, ToolCalling)
    level: difficulty level
    verbosity: tool verbosity
    task: task ID
    success: binary success (0 or 1)
"""

import pandas as pd
from loguru import logger
from pathlib import Path

# Normalization mappings - standardize variants to canonical IDs
# Canonical IDs match reports.jsonl exactly
ENV_NORMALIZATION = {
    "corral_md": "md",
    "md_melting": "md",
    "md_quenching": "md",
    "md_surface_energy": "md",
    "retrosynthesis": "retro",
}

# No model normalization needed - reports.jsonl already uses canonical IDs
MODEL_NORMALIZATION = {}


def extract_trial_data(reports_df: pd.DataFrame) -> pd.DataFrame:
    """Extract individual trial data from benchmark reports.

    Args:
        reports_df: DataFrame from reports.jsonl

    Returns:
        DataFrame with trial-level data for IRT analysis
    """
    records = []

    for _, row in reports_df.iterrows():
        model = row["model"]
        environment = row["environment"]
        agent_type = row["agent_type"]
        level = row["level"]
        category = row.get("category", "unknown")  # task or subtask
        verbosity = row.get("Tool Verbosity", "unknown")
        task_results = row.get("Task Results", {})

        # Normalize variants to canonical IDs (keep raw IDs otherwise)
        model_normalized = MODEL_NORMALIZATION.get(model, model)
        env_normalized = ENV_NORMALIZATION.get(environment, environment)

        # Extract trials from each task
        for task_id, task_data in task_results.items():
            trials = task_data.get("trials", [])

            # Get task-level Pass@k and Pass^k metrics
            pass_at_k = {
                "pass@1": task_data.get("Task Pass@1", None),
                "pass@2": task_data.get("Task Pass@2", None),
                "pass@3": task_data.get("Task Pass@3", None),
                "pass@4": task_data.get("Task Pass@4", None),
                "pass@5": task_data.get("Task Pass@5", None),
            }
            pass_hat_k = {
                "pass^1": task_data.get("Task Pass^1", None),
                "pass^2": task_data.get("Task Pass^2", None),
                "pass^3": task_data.get("Task Pass^3", None),
                "pass^4": task_data.get("Task Pass^4", None),
                "pass^5": task_data.get("Task Pass^5", None),
            }

            for trial in trials:
                success = trial.get("success", False)
                score = trial.get("score", 0)

                records.append({
                    "model": model_normalized,
                    "environment": env_normalized,
                    "scaffold": agent_type,  # Keep raw ID: react, tool_calling
                    "level": f"level_{level}",
                    "category": category,
                    "verbosity": verbosity,
                    "task": task_id,
                    "success": 1 if success else 0,
                    "score": score,
                    **pass_at_k,
                    **pass_hat_k,
                })

    result_df = pd.DataFrame(records)
    return result_df


def main():
    """Process benchmark reports and create agent trace data."""

    # Load reports
    reports_path = Path("results/data/reports.jsonl")
    if not reports_path.exists():
        raise FileNotFoundError(f"Reports not found: {reports_path}")

    logger.info(f"Loading reports from {reports_path}")
    reports_df = pd.read_json(reports_path, lines=True)
    logger.info(f"Loaded {len(reports_df)} report entries")

    # Extract trial data
    logger.info("\nExtracting trial-level data...")
    agent_data = extract_trial_data(reports_df)

    # Save
    output_path = Path("results/data/overall_trace.csv")
    agent_data.to_csv(output_path, index=False)
    logger.success(f"\nSaved agent trace data to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Agent Trace Data Summary")
    logger.info("=" * 60)
    logger.info(f"Total trials: {len(agent_data)}")
    logger.info(f"Unique tasks: {agent_data['task'].nunique()}")
    logger.info(f"Overall success rate: {agent_data['success'].mean():.3f}")
    logger.info(f"\nModels: {sorted(agent_data['model'].unique())}")
    logger.info(f"Environments: {sorted(agent_data['environment'].unique())}")
    logger.info(f"Scaffolds: {sorted(agent_data['scaffold'].unique())}")
    logger.info(f"Categories: {sorted(agent_data['category'].unique())}")
    logger.info(f"Verbosity levels: {sorted(agent_data['verbosity'].unique())}")

    logger.info("\nSuccess rate by model:")
    for model in sorted(agent_data["model"].unique()):
        model_success = agent_data[agent_data["model"] == model]["success"].mean()
        logger.info(f"  {model}: {model_success:.3f}")

    logger.info("\nSuccess rate by environment:")
    for env in sorted(agent_data["environment"].unique()):
        env_success = agent_data[agent_data["environment"] == env]["success"].mean()
        logger.info(f"  {env}: {env_success:.3f}")


if __name__ == "__main__":
    main()
