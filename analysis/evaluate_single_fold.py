"""
Run a single K-fold or Group LOO-CV iteration.

This script is designed to be called multiple times in parallel (via Snakemake, SLURM, etc.)
Each invocation runs ONE fold with full MCMC parallelism (4 chains).

Usage:
    # K-fold: Leave out one environment
    python evaluate_single_fold.py kfold --holdout_env=md --output_file=results/eval/kfold/md.json

    # Group LOO: Leave out one (model, env, scaffold) group
    python evaluate_single_fold.py group --holdout_model=claude-4.5 --holdout_env=md --holdout_scaffold=react --output_file=results/eval/group/0.json
"""

import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from latent_factor_modeling import (
    QA_ENV_NORMALIZATION,
    QA_MODEL_NORMALIZATION,
    filter_extreme_groups,
    fit_agent_model_with_task_effects,
    fit_irt_model,
    get_theta_estimates,
)
from loguru import logger
from scipy.special import expit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_metrics(y_true, y_pred_proba):
    """Compute classification metrics."""
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred_proba),
        "n_samples": len(y_true),
        "mean_pred_prob": float(np.mean(y_pred_proba)),
        "mean_true_rate": float(np.mean(y_true)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def fit_and_predict(
    train_agent_data,
    test_agent_data,
    knowledge_qa,
    reasoning_qa,
    fold_name,
):
    """Fit IRT models on train data and predict test data."""
    logger.info(f"[{fold_name}] Starting...")
    logger.info(
        f"[{fold_name}] Train: {len(train_agent_data)} trials, Test: {len(test_agent_data)} trials"
    )

    # Stage 1a: IRT on Knowledge QA
    # IMPORTANT: Use FULL QA data (not filtered by train environments)
    # Theta estimates are fixed capability measurements, independent of agent trials
    knowledge_qa_full = knowledge_qa.copy()
    knowledge_qa_full["model_id"] = knowledge_qa_full["model_id"].replace(
        QA_MODEL_NORMALIZATION
    )
    knowledge_qa_full["env_id"] = knowledge_qa_full["env_id"].replace(
        QA_ENV_NORMALIZATION
    )

    logger.info(
        f"[{fold_name}] Fitting Knowledge IRT on {len(knowledge_qa_full)} QA items (all environments)"
    )
    knowledge_irt_trace, knowledge_qa_df = fit_irt_model(knowledge_qa_full)
    knowledge_theta_df = get_theta_estimates(knowledge_irt_trace, knowledge_qa_df)

    # Stage 1b: IRT on Reasoning QA
    reasoning_qa_full = reasoning_qa.copy()
    reasoning_qa_full["model_id"] = reasoning_qa_full["model_id"].replace(
        QA_MODEL_NORMALIZATION
    )
    reasoning_qa_full["env_id"] = reasoning_qa_full["env_id"].replace(
        QA_ENV_NORMALIZATION
    )

    logger.info(
        f"[{fold_name}] Fitting Reasoning IRT on {len(reasoning_qa_full)} QA items (all environments)"
    )
    reasoning_irt_trace, reasoning_qa_df = fit_irt_model(reasoning_qa_full)
    reasoning_theta_df = get_theta_estimates(reasoning_irt_trace, reasoning_qa_df)

    # Stage 2: Agent Model with Task + Category Effects
    train_filtered = filter_extreme_groups(train_agent_data)
    logger.info(f"[{fold_name}] Fitting Agent Model on {len(train_filtered)} trials")

    agent_trace, agent_df = fit_agent_model_with_task_effects(
        agent_df=train_filtered,
        knowledge_theta_df=knowledge_theta_df,
        reasoning_theta_df=reasoning_theta_df,
        include_category=True,
    )

    # Stage 3: Predict Test Data
    logger.info(f"[{fold_name}] Generating predictions for test data...")

    test_with_theta = test_agent_data.merge(
        knowledge_theta_df[["model", "environment", "theta_mean"]].rename(
            columns={"theta_mean": "theta_knowledge"}
        ),
        on=["model", "environment"],
        how="left",
    )
    test_with_theta = test_with_theta.merge(
        reasoning_theta_df[["model", "environment", "theta_mean"]].rename(
            columns={"theta_mean": "theta_reasoning"}
        ),
        on=["model", "environment"],
        how="left",
    )

    test_with_theta = test_with_theta.dropna(
        subset=["theta_knowledge", "theta_reasoning"]
    )
    logger.info(
        f"[{fold_name}] Test samples after theta merge: {len(test_with_theta)}/{len(test_agent_data)}"
    )

    if len(test_with_theta) == 0:
        logger.warning(f"[{fold_name}] No test samples with valid theta estimates!")
        return {
            "fold": fold_name,
            "metrics": None,
            "n_train": len(train_agent_data),
            "n_test": len(test_agent_data),
            "n_test_with_theta": 0,
        }

    # Get posterior means for prediction
    beta0 = agent_trace.posterior["beta0"].mean().to_numpy()
    lambda_vals = agent_trace.posterior["lambda"].mean(dim=["chain", "draw"]).to_numpy()
    psi_vals = agent_trace.posterior["psi"].mean(dim=["chain", "draw"]).to_numpy()
    gamma = agent_trace.posterior["gamma"].mean(dim=["chain", "draw"]).to_numpy()
    delta_vals = (
        agent_trace.posterior["delta_inc"].mean(dim=["chain", "draw"]).to_numpy()
    )
    kappa = agent_trace.posterior["kappa"].mean(dim=["chain", "draw"]).to_numpy()

    # Preprocess test data to match training data format
    # Extract numeric level from "level_1" → 1
    test_with_theta["level_numeric"] = (
        test_with_theta["level"].str.extract(r"(\d+)").astype(int)
    )
    test_with_theta["level_idx"] = test_with_theta["level_numeric"] - 1

    # Map test data to indices
    env_map = {env: i for i, env in enumerate(sorted(agent_df["environment"].unique()))}
    scaffold_map = {s: i for i, s in enumerate(sorted(agent_df["scaffold"].unique()))}
    verbosity_map = {v: i for i, v in enumerate(sorted(agent_df["verbosity"].unique()))}
    category_map = {"task": 0, "subtask": 1}

    # Predict
    predictions = []
    for _, row in test_with_theta.iterrows():
        env_idx = env_map.get(row["environment"])
        scaffold_idx = scaffold_map.get(row["scaffold"])
        level_idx = int(row["level_idx"])  # Already computed above
        verbosity_idx = verbosity_map.get(row["verbosity"])
        category_idx = category_map.get(row["category"])

        if any(
            idx is None
            for idx in [env_idx, scaffold_idx, level_idx, verbosity_idx, category_idx]
        ):
            continue

        logit = (
            beta0
            + lambda_vals[env_idx] * row["theta_knowledge"]
            + psi_vals[env_idx] * row["theta_reasoning"]
            + gamma[scaffold_idx]
            + delta_vals[level_idx]
            + delta_vals[3 + verbosity_idx]
            + kappa[category_idx]
        )

        pred_prob = expit(logit)
        predictions.append(pred_prob)

    if len(predictions) == 0:
        logger.warning(f"[{fold_name}] No valid predictions generated!")
        return {
            "fold": fold_name,
            "metrics": None,
            "n_train": len(train_agent_data),
            "n_test": len(test_agent_data),
            "n_test_with_theta": len(test_with_theta),
        }

    y_true = test_with_theta.iloc[: len(predictions)]["success"].to_numpy()
    y_pred_proba = np.array(predictions)

    metrics = compute_metrics(y_true, y_pred_proba)
    logger.success(
        f"[{fold_name}] Accuracy: {metrics['accuracy']:.3f}, Log Loss: {metrics['log_loss']:.3f}"
    )

    return {
        "fold": fold_name,
        "metrics": metrics,
        "n_train": len(train_agent_data),
        "n_test": len(test_agent_data),
        "n_test_with_theta": len(test_with_theta),
        "predictions": predictions,
        "y_true": y_true.tolist(),
    }


def kfold(
    holdout_env: str,
    output_file: str,
    data_dir: str = "results/data",
):
    """Run one K-fold CV iteration (leave out one environment).

    Args:
        holdout_env: Environment to hold out (e.g., 'md', 'afm')
        output_file: Path to save results JSON
        data_dir: Directory with input data
    """
    logger.info(f"K-FOLD: Holding out environment '{holdout_env}'")

    # Load data
    knowledge_qa = pd.read_csv(f"{data_dir}/knowledge_qa.csv")
    reasoning_qa = pd.read_csv(f"{data_dir}/reasoning_qa.csv")
    agent_data = pd.read_csv(f"{data_dir}/overall_trace.csv")

    # Remove wetlab
    agent_data = agent_data[agent_data["environment"] != "wetlab"].copy()

    # Split train/test
    train_data = agent_data[agent_data["environment"] != holdout_env]
    test_data = agent_data[agent_data["environment"] == holdout_env]

    # Run evaluation
    result = fit_and_predict(
        train_data,
        test_data,
        knowledge_qa,
        reasoning_qa,
        fold_name=f"LOEO-{holdout_env}",
    )

    # Save result
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    logger.success(f"Saved results to {output_file}")


def group(
    holdout_model: str,
    holdout_env: str,
    holdout_scaffold: str,
    output_file: str,
    data_dir: str = "results/data",
):
    """Run one Group LOO-CV iteration (leave out one model/env/scaffold combo).

    Args:
        holdout_model: Model to hold out (e.g., 'claude-4.5')
        holdout_env: Environment to hold out (e.g., 'md')
        holdout_scaffold: Scaffold to hold out (e.g., 'react')
        output_file: Path to save results JSON
        data_dir: Directory with input data
    """
    logger.info(
        f"GROUP LOO: Holding out ({holdout_model}, {holdout_env}, {holdout_scaffold})"
    )

    # Load data
    knowledge_qa = pd.read_csv(f"{data_dir}/knowledge_qa.csv")
    reasoning_qa = pd.read_csv(f"{data_dir}/reasoning_qa.csv")
    agent_data = pd.read_csv(f"{data_dir}/overall_trace.csv")

    # Remove wetlab
    agent_data = agent_data[agent_data["environment"] != "wetlab"].copy()

    # Split train/test
    train_data = agent_data[
        ~(
            (agent_data["model"] == holdout_model)
            & (agent_data["environment"] == holdout_env)
            & (agent_data["scaffold"] == holdout_scaffold)
        )
    ]
    test_data = agent_data[
        (agent_data["model"] == holdout_model)
        & (agent_data["environment"] == holdout_env)
        & (agent_data["scaffold"] == holdout_scaffold)
    ]

    # Run evaluation
    result = fit_and_predict(
        train_data,
        test_data,
        knowledge_qa,
        reasoning_qa,
        fold_name=f"Group-{holdout_model}-{holdout_env}-{holdout_scaffold}",
    )

    # Save result
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    logger.success(f"Saved results to {output_file}")


if __name__ == "__main__":
    fire.Fire({"kfold": kfold, "group": group})
