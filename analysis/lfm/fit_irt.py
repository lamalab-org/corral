"""
Fit 2PL IRT models to knowledge and reasoning QA data.

Estimates per-(model, environment) ability parameters (theta).

Produces:
    data/irt_baseline/knowledge_theta.csv
    data/irt_baseline/reasoning_theta.csv

Usage:
    python fit_irt.py
    python fit_irt.py --data-dir data --samples 2000 --tune 1000
"""

from pathlib import Path

import arviz as az
import fire
import numpy as np
import pandas as pd
import pymc as pm
from loguru import logger

# Normalization mappings - QA data uses different names than overall_trace
QA_MODEL_NORMALIZATION = {
    "Claude-4.5": "claude-4.5",
    "oss": "gpt-oss-120b",
}

QA_ENV_NORMALIZATION = {
    "corral_md": "md",
}


def load_qa_data(qa_type, data_dir="data"):
    """Load and normalize QA data."""
    path = Path(data_dir) / f"{qa_type}_qa.csv"
    qa_df = pd.read_csv(path)
    logger.info(
        f"Loaded {qa_type} QA: {len(qa_df)} rows, {qa_df['item_id'].nunique()} items"
    )

    qa_df["model_id"] = qa_df["model_id"].replace(QA_MODEL_NORMALIZATION)
    qa_df["env_id"] = qa_df["env_id"].replace(QA_ENV_NORMALIZATION)

    logger.info(f"  Models: {sorted(qa_df['model_id'].unique())}")
    logger.info(f"  Envs:   {sorted(qa_df['env_id'].unique())}")
    return qa_df


def fit_irt_model(qa_df, samples=2000, tune=1000):
    """
    Fit 2PL IRT model.

    Model:
        theta[model,env] ~ N(mu[model] + nu[env], sigma_theta)
        a[item] = exp(log_a[item]),  log_a ~ N(0, 0.5)
        b[item] ~ N(0, 2)
        P(correct) = logistic(a[item] * theta[model,env] - b[item])
    """
    qa_df = qa_df.copy()

    qa_df["model_idx"] = pd.Categorical(qa_df["model_id"]).codes
    qa_df["env_idx"] = pd.Categorical(qa_df["env_id"]).codes
    qa_df["item_idx"] = pd.Categorical(qa_df["item_id"]).codes
    qa_df["theta_idx"] = qa_df.groupby(["model_idx", "env_idx"]).ngroup()

    N_theta = qa_df["theta_idx"].nunique()
    N_items = qa_df["item_idx"].nunique()
    N_models = qa_df["model_idx"].nunique()
    N_envs = qa_df["env_idx"].nunique()

    logger.info(
        f"IRT dimensions: {N_models} models, {N_envs} envs, {N_theta} thetas, {N_items} items"
    )

    theta_to_model = (
        qa_df.groupby("theta_idx")["model_idx"].first().to_numpy().astype(int)
    )
    theta_to_env = qa_df.groupby("theta_idx")["env_idx"].first().to_numpy().astype(int)

    with pm.Model():
        # Model effects (sum-to-zero)
        if N_models > 1:
            mu_raw = pm.Normal("mu_raw", 0, 1, shape=N_models - 1)
            mu = pm.Deterministic(
                "mu", pm.math.concatenate([mu_raw, [-pm.math.sum(mu_raw)]])
            )
        else:
            mu = pm.Deterministic("mu", pm.math.constant(np.array([0.0])))

        # Environment effects (sum-to-zero)
        if N_envs > 1:
            nu_raw = pm.Normal("nu_raw", 0, 1, shape=N_envs - 1)
            nu = pm.Deterministic(
                "nu", pm.math.concatenate([nu_raw, [-pm.math.sum(nu_raw)]])
            )
        else:
            nu = pm.Deterministic("nu", pm.math.constant(np.array([0.0])))

        # Ability parameters
        sigma_theta = pm.HalfNormal("sigma_theta", 0.5)
        theta_mean = mu[theta_to_model] + nu[theta_to_env]
        theta = pm.Normal("theta", mu=theta_mean, sigma=sigma_theta, shape=N_theta)

        # Item parameters (2PL)
        log_a = pm.Normal("log_a", 0, 0.5, shape=N_items)
        a = pm.Deterministic("a", pm.math.exp(log_a))
        b = pm.Normal("b", 0, 2, shape=N_items)

        # Likelihood
        item_indices = qa_df["item_idx"].to_numpy().astype(int)
        theta_indices = qa_df["theta_idx"].to_numpy().astype(int)

        eta = a[item_indices] * theta[theta_indices] - b[item_indices]
        pm.Bernoulli("y", logit_p=eta, observed=qa_df["correct"].to_numpy())

        trace = pm.sample(
            samples,
            tune=tune,
            chains=4,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
        )

    return trace, qa_df


def get_theta_estimates(trace, qa_df):
    """Extract theta posteriors as a dataframe with original model/env names."""
    summary = az.summary(trace, var_names=["theta"], hdi_prob=0.9)

    theta_to_model = qa_df.groupby("theta_idx")["model_id"].first()
    theta_to_env = qa_df.groupby("theta_idx")["env_id"].first()

    results = [
        {
            "model": theta_to_model[idx],
            "environment": theta_to_env[idx],
            "theta_mean": summary.iloc[idx]["mean"],
            "theta_sd": summary.iloc[idx]["sd"],
        }
        for idx in range(len(summary))
    ]
    return pd.DataFrame(results)


def main(data_dir="data", output_dir=None, samples=2000, tune=1000):
    """Fit IRT models for knowledge and reasoning QA data.

    Args:
        data_dir: Directory containing knowledge_qa.csv and reasoning_qa.csv
        output_dir: Directory to write theta CSVs (defaults to data_dir/irt_baseline)
        samples: Number of posterior samples per chain
        tune: Number of tuning samples per chain
    """
    if output_dir is None:
        output_dir = Path(data_dir) / "irt_baseline"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for qa_type in ["knowledge", "reasoning"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"Fitting {qa_type} IRT model...")
        logger.info(f"{'='*50}")

        qa_df = load_qa_data(qa_type, data_dir)
        trace, qa_df = fit_irt_model(qa_df, samples=samples, tune=tune)
        theta_df = get_theta_estimates(trace, qa_df)

        out_path = output_dir / f"{qa_type}_theta.csv"
        theta_df.to_csv(out_path, index=False)
        logger.info(f"Saved {qa_type} thetas to {out_path}")
        logger.info(f"\n{theta_df.to_string(index=False)}")

    logger.info("IRT fitting complete.")


if __name__ == "__main__":
    fire.Fire(main)
