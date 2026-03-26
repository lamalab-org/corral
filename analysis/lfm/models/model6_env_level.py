"""
Model 6: Tasks + Environment x Level Interactions

logit(P(success)) = b0 + lam*knowledge_z + psi*reasoning_z
                   + gamma_scaffold + xi_verbosity + kappa_category
                   + delta_env_level[e, l] + tau[task_id]

PURPOSE: How does difficulty progression vary by environment?
Non-parametric: each (environment, level) combo gets a unique effect.
"""

import pymc as pm
from loguru import logger


def build_model(df):
    logger.info("\n=== Building Model 6: Tasks + Environment x Level ===")

    y = df["success"].to_numpy()
    knowledge_z = df["knowledge_z"].to_numpy()
    reasoning_z = df["reasoning_z"].to_numpy()
    scaffold_id = df["scaffold_id"].to_numpy()
    category_id = df["category_id"].to_numpy()
    verbosity_id = df["verbosity_id"].to_numpy()
    env_level_id = df["env_level_id"].to_numpy()
    task_id = df["task_id"].to_numpy()

    n_scaffolds = df["scaffold_id"].nunique()
    n_categories = df["category_id"].nunique()
    n_verbosity = df["verbosity_id"].nunique()
    n_env_level = df["env_level_id"].nunique()
    n_tasks = df["task_id"].nunique()

    logger.info(
        f"Observations: {len(df):,}, Env x Level: {n_env_level}, Tasks: {n_tasks}"
    )

    with pm.Model() as model:
        b0 = pm.Normal("intercept", mu=0, sigma=2)
        lam = pm.Normal("knowledge_coef", mu=0, sigma=1)
        psi = pm.Normal("reasoning_coef", mu=0, sigma=1)

        g_raw = pm.Normal("scaffold_raw", mu=0, sigma=1, shape=n_scaffolds)
        g = pm.Deterministic("scaffold_effect", g_raw - g_raw.mean())

        d_raw = pm.Normal("env_level_raw", mu=0, sigma=1, shape=n_env_level)
        d = pm.Deterministic("env_level_effect", d_raw - d_raw.mean())

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
            + g[scaffold_id]
            + d[env_level_id]
            + k[category_id]
            + x[verbosity_id]
            + tau[task_id]
        )

        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        pm.Bernoulli("y_obs", p=p, observed=y)

    return model
