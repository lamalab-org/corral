"""
Model 4: Tasks + Scaffold x Environment Interactions — Binomial likelihood

logit(P(success)) = b0 + lam*knowledge_z + psi*reasoning_z
                   + delta_level + xi_verbosity + kappa_category
                   + gamma_scaffold_env[s, e] + tau[task_id]

PURPOSE: Does scaffold effectiveness vary by domain?
No separate environment or scaffold effects (absorbed into interaction).
"""

import pymc as pm
from loguru import logger


def build_model(df):
    logger.info("\n=== Building Model 4: Tasks + Scaffold x Environment [Binomial] ===")

    k_success = df["k_success"].to_numpy()
    n_trials = df["n_trials"].to_numpy()
    knowledge_z = df["knowledge_z"].to_numpy()
    reasoning_z = df["reasoning_z"].to_numpy()
    level_id = df["level_id"].to_numpy()
    category_id = df["category_id"].to_numpy()
    verbosity_id = df["verbosity_id"].to_numpy()
    env_scaffold_id = df["env_scaffold_id"].to_numpy()
    task_id = df["task_id"].to_numpy()

    n_levels = df["level_id"].nunique()
    n_categories = df["category_id"].nunique()
    n_verbosity = df["verbosity_id"].nunique()
    n_env_scaffold = df["env_scaffold_id"].nunique()
    n_tasks = df["task_id"].nunique()

    logger.info(
        f"Observations: {len(df):,}, Scaffold x Env: {n_env_scaffold}, Tasks: {n_tasks}"
    )

    with pm.Model() as model:
        b0 = pm.Normal("intercept", mu=0, sigma=2)
        lam = pm.Normal("knowledge_coef", mu=0, sigma=1)
        psi = pm.Normal("reasoning_coef", mu=0, sigma=1)

        g_raw = pm.Normal("scaffold_env_raw", mu=0, sigma=1, shape=n_env_scaffold)
        g = pm.Deterministic("scaffold_env_effect", g_raw - g_raw.mean())

        d_raw = pm.Normal("level_raw", mu=0, sigma=1, shape=n_levels)
        d = pm.Deterministic("level_effect", d_raw - d_raw.mean())

        k_raw = pm.Normal("category_raw", mu=0, sigma=1, shape=n_categories)
        k = pm.Deterministic("category_effect", k_raw - k_raw.mean())

        x_raw = pm.Normal("verbosity_raw", mu=0, sigma=1, shape=n_verbosity)
        x = pm.Deterministic("verbosity_effect", x_raw - x_raw.mean())

        sigma_tau = pm.HalfNormal("sigma_task", sigma=0.5)
        tau_raw = pm.Normal("task_raw", mu=0, sigma=1, shape=n_tasks)
        tau = pm.Deterministic("task_effect", sigma_tau * tau_raw)

        logit_p = (
            b0
            + lam * knowledge_z
            + psi * reasoning_z
            + g[env_scaffold_id]
            + d[level_id]
            + k[category_id]
            + x[verbosity_id]
            + tau[task_id]
        )

        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        pm.Binomial("y_obs", n=n_trials, p=p, observed=k_success)

    return model
