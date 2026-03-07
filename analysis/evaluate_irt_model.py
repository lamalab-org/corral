"""
Evaluate IRT Model Predictive Performance

Two evaluation strategies:
1. K-Fold CV (Leave-One-Environment-Out): Tests generalization to new domains
2. Group LOO-CV: Tests generalization to new (model, environment, scaffold) combinations

Both approaches refit the entire pipeline (IRT + Agent model) for proper evaluation.

Usage:
    # Leave-one-environment-out (7 folds, parallelized)
    python evaluate_irt_model.py kfold --n_jobs=7

    # Group LOO-CV (42 groups, parallelized)
    python evaluate_irt_model.py group_loo --n_jobs=42

    # Both evaluations
    python evaluate_irt_model.py all --n_jobs=20
"""

import json

# Import fitting functions from main script
import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy.special import expit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from latent_factor_modeling import (
    QA_ENV_NORMALIZATION,
    QA_MODEL_NORMALIZATION,
    filter_extreme_groups,
    fit_agent_model_with_task_effects,
    fit_irt_model,
    get_theta_estimates,
)


def compute_metrics(y_true, y_pred_proba):
    """Compute classification metrics.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        dict of metrics
    """
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred_proba),
        "n_samples": len(y_true),
        "mean_pred_prob": float(np.mean(y_pred_proba)),
        "mean_true_rate": float(np.mean(y_true)),
    }

    # ROC-AUC only if we have both classes
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
    """Fit IRT models on train data and predict test data.

    Args:
        train_agent_data: Training agent trials
        test_agent_data: Test agent trials
        knowledge_qa: Full knowledge QA data
        reasoning_qa: Full reasoning QA data
        fold_name: Name of this fold for logging

    Returns:
        dict with predictions and metrics
    """
    logger.info(f"[{fold_name}] Starting...")
    logger.info(
        f"[{fold_name}] Train: {len(train_agent_data)} trials, Test: {len(test_agent_data)} trials"
    )

    # ==========================================================================
    # Stage 1a: IRT on Knowledge QA (only train environments/models)
    # ==========================================================================
    train_envs = set(train_agent_data["environment"].unique())
    train_models = set(train_agent_data["model"].unique())

    # Filter QA data to train environments/models
    knowledge_qa_train = knowledge_qa[
        knowledge_qa["env_id"].isin(train_envs)
        & knowledge_qa["model_id"].isin(train_models)
    ].copy()

    knowledge_qa_train["model_id"] = knowledge_qa_train["model_id"].replace(
        QA_MODEL_NORMALIZATION
    )
    knowledge_qa_train["env_id"] = knowledge_qa_train["env_id"].replace(
        QA_ENV_NORMALIZATION
    )

    logger.info(
        f"[{fold_name}] Fitting Knowledge IRT on {len(knowledge_qa_train)} QA items"
    )
    knowledge_irt_trace, knowledge_qa_df = fit_irt_model(knowledge_qa_train)
    knowledge_theta_df = get_theta_estimates(knowledge_irt_trace, knowledge_qa_df)

    # ==========================================================================
    # Stage 1b: IRT on Reasoning QA (only train environments/models)
    # ==========================================================================
    reasoning_qa_train = reasoning_qa[
        reasoning_qa["env_id"].isin(train_envs)
        & reasoning_qa["model_id"].isin(train_models)
    ].copy()

    reasoning_qa_train["model_id"] = reasoning_qa_train["model_id"].replace(
        QA_MODEL_NORMALIZATION
    )
    reasoning_qa_train["env_id"] = reasoning_qa_train["env_id"].replace(
        QA_ENV_NORMALIZATION
    )

    logger.info(
        f"[{fold_name}] Fitting Reasoning IRT on {len(reasoning_qa_train)} QA items"
    )
    reasoning_irt_trace, reasoning_qa_df = fit_irt_model(reasoning_qa_train)
    reasoning_theta_df = get_theta_estimates(reasoning_irt_trace, reasoning_qa_df)

    # ==========================================================================
    # Stage 2: Agent Model with Task + Category Effects
    # ==========================================================================
    train_filtered = filter_extreme_groups(train_agent_data)
    logger.info(f"[{fold_name}] Fitting Agent Model on {len(train_filtered)} trials")

    agent_trace, agent_df = fit_agent_model_with_task_effects(
        agent_df=train_filtered,
        knowledge_theta_df=knowledge_theta_df,
        reasoning_theta_df=reasoning_theta_df,
        include_category=True,
    )

    # ==========================================================================
    # Stage 3: Predict Test Data
    # ==========================================================================
    logger.info(f"[{fold_name}] Generating predictions for test data...")

    # Merge theta estimates into test data
    test_with_theta = test_agent_data.merge(
        knowledge_theta_df[["model", "environment", "theta"]].rename(
            columns={"theta": "theta_knowledge"}
        ),
        on=["model", "environment"],
        how="left",
    )
    test_with_theta = test_with_theta.merge(
        reasoning_theta_df[["model", "environment", "theta"]].rename(
            columns={"theta": "theta_reasoning"}
        ),
        on=["model", "environment"],
        how="left",
    )

    # Drop rows with missing theta (test combinations not in train)
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

    # Map test data to indices
    env_map = {env: i for i, env in enumerate(sorted(agent_df["environment"].unique()))}
    scaffold_map = {s: i for i, s in enumerate(sorted(agent_df["scaffold"].unique()))}
    level_map = {lv: i for i, lv in enumerate(sorted(agent_df["level"].unique()))}
    verbosity_map = {v: i for i, v in enumerate(sorted(agent_df["verbosity"].unique()))}
    category_map = {"task": 0, "subtask": 1}

    # Predict
    predictions = []
    for _, row in test_with_theta.iterrows():
        env_idx = env_map.get(row["environment"])
        scaffold_idx = scaffold_map.get(row["scaffold"])
        level_idx = level_map.get(row["level"])
        verbosity_idx = verbosity_map.get(row["verbosity"])
        category_idx = category_map.get(row["category"])

        # Skip if any mapping failed
        if any(
            idx is None
            for idx in [env_idx, scaffold_idx, level_idx, verbosity_idx, category_idx]
        ):
            continue

        # Linear predictor
        logit = (
            beta0
            + lambda_vals[env_idx] * row["theta_knowledge"]
            + psi_vals[env_idx] * row["theta_reasoning"]
            + gamma[scaffold_idx]
            + delta_vals[level_idx]
            + delta_vals[3 + verbosity_idx]  # verbosity after level
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


def kfold_cv(
    knowledge_qa, reasoning_qa, agent_data, n_jobs=7, output_dir="results/eval"
):
    """Leave-One-Environment-Out Cross-Validation.

    Args:
        knowledge_qa: Knowledge QA dataframe
        reasoning_qa: Reasoning QA dataframe
        agent_data: Agent trial dataframe
        n_jobs: Number of parallel jobs
        output_dir: Output directory

    Returns:
        dict of results
    """
    logger.info("=" * 80)
    logger.info("K-FOLD CV: Leave-One-Environment-Out")
    logger.info("=" * 80)

    # Remove wetlab (as in main script)
    agent_data = agent_data[agent_data["environment"] != "wetlab"].copy()

    environments = sorted(agent_data["environment"].unique())
    logger.info(f"Environments: {environments}")
    logger.info(f"Running {len(environments)}-fold CV with {n_jobs} parallel jobs")

    def run_fold(holdout_env):
        train_data = agent_data[agent_data["environment"] != holdout_env]
        test_data = agent_data[agent_data["environment"] == holdout_env]

        return fit_and_predict(
            train_data,
            test_data,
            knowledge_qa,
            reasoning_qa,
            fold_name=f"LOEO-{holdout_env}",
        )

    # Run folds in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_fold)(env) for env in environments
    )

    # Aggregate metrics
    valid_results = [r for r in results if r["metrics"] is not None]
    if len(valid_results) == 0:
        logger.error("No valid fold results!")
        return {"results": results, "summary": None}

    # Compute mean metrics
    all_metrics = [r["metrics"] for r in valid_results]
    summary = {
        metric: float(np.mean([m[metric] for m in all_metrics]))
        for metric in all_metrics[0]
    }
    summary["n_folds"] = len(valid_results)

    logger.info("\n" + "=" * 80)
    logger.info("K-FOLD CV SUMMARY")
    logger.info("=" * 80)
    for metric, value in summary.items():
        if metric != "n_folds":
            logger.info(f"  {metric:20s}: {value:.4f}")
    logger.info(f"  {'n_folds':20s}: {summary['n_folds']}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "kfold_results.json").open("w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    logger.success(f"Saved results to {output_path / 'kfold_results.json'}")

    return {"results": results, "summary": summary}


def group_loo_cv(
    knowledge_qa, reasoning_qa, agent_data, n_jobs=20, output_dir="results/eval"
):
    """Group-Level LOO-CV: Leave out (model, environment, scaffold) groups.

    Args:
        knowledge_qa: Knowledge QA dataframe
        reasoning_qa: Reasoning QA dataframe
        agent_data: Agent trial dataframe
        n_jobs: Number of parallel jobs
        output_dir: Output directory

    Returns:
        dict of results
    """
    logger.info("=" * 80)
    logger.info("GROUP LOO-CV: Leave-Out (model, environment, scaffold) combinations")
    logger.info("=" * 80)

    # Remove wetlab
    agent_data = agent_data[agent_data["environment"] != "wetlab"].copy()

    # Get all groups
    groups = (
        agent_data.groupby(["model", "environment", "scaffold"])
        .size()
        .reset_index()
        .rename(columns={0: "n_trials"})
    )
    logger.info(f"Total groups: {len(groups)}")
    logger.info(f"Running with {n_jobs} parallel jobs")

    def run_group(group_idx, model, env, scaffold):
        train_data = agent_data[
            ~(
                (agent_data["model"] == model)
                & (agent_data["environment"] == env)
                & (agent_data["scaffold"] == scaffold)
            )
        ]
        test_data = agent_data[
            (agent_data["model"] == model)
            & (agent_data["environment"] == env)
            & (agent_data["scaffold"] == scaffold)
        ]

        return fit_and_predict(
            train_data,
            test_data,
            knowledge_qa,
            reasoning_qa,
            fold_name=f"Group-{group_idx:02d}-{model}-{env}-{scaffold}",
        )

    # Run groups in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_group)(i, row["model"], row["environment"], row["scaffold"])
        for i, row in groups.iterrows()
    )

    # Aggregate metrics
    valid_results = [r for r in results if r["metrics"] is not None]
    if len(valid_results) == 0:
        logger.error("No valid group results!")
        return {"results": results, "summary": None}

    all_metrics = [r["metrics"] for r in valid_results]
    summary = {
        metric: float(np.mean([m[metric] for m in all_metrics]))
        for metric in all_metrics[0]
    }
    summary["n_groups"] = len(valid_results)

    logger.info("\n" + "=" * 80)
    logger.info("GROUP LOO-CV SUMMARY")
    logger.info("=" * 80)
    for metric, value in summary.items():
        if metric != "n_groups":
            logger.info(f"  {metric:20s}: {value:.4f}")
    logger.info(f"  {'n_groups':20s}: {summary['n_groups']}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "group_loo_results.json").open("w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    logger.success(f"Saved results to {output_path / 'group_loo_results.json'}")

    return {"results": results, "summary": summary}


def kfold(n_jobs: int = 7, output_dir: str = "results/eval"):
    """Run K-Fold (Leave-One-Environment-Out) Cross-Validation.

    Args:
        n_jobs: Number of parallel jobs (default: 7 for 7 environments)
        output_dir: Output directory for results
    """
    # Load data
    knowledge_qa = pd.read_csv("results/data/knowledge_qa.csv")
    reasoning_qa = pd.read_csv("results/data/reasoning_qa.csv")
    agent_data = pd.read_csv("results/data/overall_trace.csv")

    return kfold_cv(knowledge_qa, reasoning_qa, agent_data, n_jobs, output_dir)


def group_loo(n_jobs: int = 20, output_dir: str = "results/eval"):
    """Run Group-Level LOO Cross-Validation.

    Args:
        n_jobs: Number of parallel jobs (default: 20)
        output_dir: Output directory for results
    """
    # Load data
    knowledge_qa = pd.read_csv("results/data/knowledge_qa.csv")
    reasoning_qa = pd.read_csv("results/data/reasoning_qa.csv")
    agent_data = pd.read_csv("results/data/overall_trace.csv")

    return group_loo_cv(knowledge_qa, reasoning_qa, agent_data, n_jobs, output_dir)


def run_all(n_jobs: int = 20, output_dir: str = "results/eval"):
    """Run both K-Fold and Group LOO evaluations.

    Args:
        n_jobs: Number of parallel jobs
        output_dir: Output directory for results
    """
    # Load data once
    knowledge_qa = pd.read_csv("results/data/knowledge_qa.csv")
    reasoning_qa = pd.read_csv("results/data/reasoning_qa.csv")
    agent_data = pd.read_csv("results/data/overall_trace.csv")

    # Run both evaluations
    logger.info("Running both K-Fold and Group LOO evaluations")

    kfold_results = kfold_cv(
        knowledge_qa, reasoning_qa, agent_data, min(n_jobs, 7), output_dir
    )
    group_results = group_loo_cv(
        knowledge_qa, reasoning_qa, agent_data, n_jobs, output_dir
    )

    return {"kfold": kfold_results, "group_loo": group_results}


if __name__ == "__main__":
    fire.Fire({"kfold": kfold, "group_loo": group_loo, "all": run_all})
