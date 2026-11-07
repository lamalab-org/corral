"""
Model 5: Tasks + Scaffold x Level + Environment — Binomial likelihood

logit(P(success)) = b0 + lam*knowledge_z + psi*reasoning_z
                   + xi_verbosity + kappa_category
                   + gamma_scaffold_level[s, l] + alpha[environment_id] + tau[task_id]

PURPOSE: Does scaffold effectiveness vary by task difficulty?
"""

import pymc as pm
from loguru import logger


def build_model(df):
    logger.info(
        "\n=== Building Model 5: Tasks + Scaffold x Level + Environment [Binomial] ==="
    )

    k_success = df["k_success"].to_numpy()
    n_trials = df["n_trials"].to_numpy()
    knowledge_z = df["knowledge_z"].to_numpy()
    reasoning_z = df["reasoning_z"].to_numpy()
    category_id = df["category_id"].to_numpy()
    verbosity_id = df["verbosity_id"].to_numpy()
    scaffold_level_id = df["scaffold_level_id"].to_numpy()
    environment_id = df["environment_id"].to_numpy()
    task_id = df["task_id"].to_numpy()

    n_categories = df["category_id"].nunique()
    n_verbosity = df["verbosity_id"].nunique()
    n_scaffold_level = df["scaffold_level_id"].nunique()
    n_environments = df["environment_id"].nunique()
    n_tasks = df["task_id"].nunique()

    logger.info(
        f"Observations: {len(df):,}, Scaffold x Level: {n_scaffold_level}, Tasks: {n_tasks}"
    )

    with pm.Model() as model:
        b0 = pm.Normal("intercept", mu=0, sigma=2)
        lam = pm.Normal("knowledge_coef", mu=0, sigma=1)
        psi = pm.Normal("reasoning_coef", mu=0, sigma=1)

        g_raw = pm.Normal("scaffold_level_raw", mu=0, sigma=1, shape=n_scaffold_level)
        g = pm.Deterministic("scaffold_level_effect", g_raw - g_raw.mean())

        k_raw = pm.Normal("category_raw", mu=0, sigma=1, shape=n_categories)
        k = pm.Deterministic("category_effect", k_raw - k_raw.mean())

        x_raw = pm.Normal("verbosity_raw", mu=0, sigma=1, shape=n_verbosity)
        x = pm.Deterministic("verbosity_effect", x_raw - x_raw.mean())

        sigma_alpha = pm.HalfNormal("sigma_environment", sigma=0.5)
        alpha_raw = pm.Normal("environment_raw", mu=0, sigma=1, shape=n_environments)
        alpha = pm.Deterministic("environment_effect", sigma_alpha * alpha_raw)

        sigma_tau = pm.HalfNormal("sigma_task", sigma=0.5)
        tau_raw = pm.Normal("task_raw", mu=0, sigma=1, shape=n_tasks)
        tau = pm.Deterministic("task_effect", sigma_tau * tau_raw)

        logit_p = (
            b0
            + lam * knowledge_z
            + psi * reasoning_z
            + g[scaffold_level_id]
            + k[category_id]
            + x[verbosity_id]
            + alpha[environment_id]
            + tau[task_id]
        )

        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        pm.Binomial("y_obs", n=n_trials, p=p, observed=k_success)

    return model
