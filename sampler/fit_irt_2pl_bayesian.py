"""Full Bayesian (MCMC) version of the 2PL fit.

Same model, same priors as fit_irt_2pl.py (theta ~ N(0,1), b ~ N(0,2),
log_a ~ N(0,0.5)) — the only thing that changes is HOW we fit it: instead of
a single penalized-MLE point (the posterior mode), we now recover the full
posterior distribution via NUTS/MCMC. This directly answers "how much
uncertainty was the MAP point estimate hiding" for theta_j, a_i, b_i.

Usage:
    python fit_irt_2pl_bayesian.py
"""

import sys
from pathlib import Path

import arviz as az
import fire
import numpy as np
import pandas as pd
import pymc as pm
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from fit_irt_2pl import build_trial_counts

OUT_DIR = Path(__file__).parent / "data"
LEGACY_MODELS = ["claude-4.5", "gpt-4o", "gpt-oss-120b"]


def fit_2pl_bayesian(
    K: np.ndarray,
    N: np.ndarray,
    theta_sd: float = 1.0,
    b_sd: float = 2.0,
    log_a_sd: float = 0.5,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 0,
) -> az.InferenceData:
    n_subj, n_item = K.shape
    with pm.Model():
        theta = pm.Normal("theta", mu=0, sigma=theta_sd, shape=n_subj)
        b = pm.Normal("b", mu=0, sigma=b_sd, shape=n_item)
        log_a = pm.Normal("log_a", mu=0, sigma=log_a_sd, shape=n_item)
        a = pm.Deterministic("a", pm.math.exp(log_a))

        logit_p = a[None, :] * (theta[:, None] - b[None, :])
        p = pm.math.sigmoid(logit_p)
        pm.Binomial("obs", n=N, p=p, observed=K)

        return pm.sample(
            draws=draws, tune=tune, chains=chains, random_seed=seed, target_accept=0.9
        )


def main(
    exclude_environments: str = "resistor",
    legacy_only: bool = True,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 0,
) -> None:
    excluded = [e.strip() for e in exclude_environments.split(",") if e.strip()]
    subjects, items, K, N, item_env = build_trial_counts(exclude_environments=excluded)

    if legacy_only:
        keep_idx = [
            i for i, s in enumerate(subjects) if s.split("__")[0] in LEGACY_MODELS
        ]
        subjects = [subjects[i] for i in keep_idx]
        K, N = K[keep_idx], N[keep_idx]
        logger.info(
            f"Fitting on {len(subjects)} legacy subjects only (matches stratified_irt's design fit)"
        )
    else:
        logger.info(f"Fitting on all {len(subjects)} subjects")

    logger.info(
        f"{len(subjects)} subjects x {len(items)} items, {int(N.sum())} trials — starting NUTS sampling"
    )
    trace = fit_2pl_bayesian(K, N, draws=draws, tune=tune, chains=chains, seed=seed)

    summary = az.summary(trace, var_names=["theta", "a", "b"])
    n_divergent = int(trace.sample_stats.diverging.sum())
    max_rhat = pd.to_numeric(summary["r_hat"], errors="coerce").max()
    min_ess = pd.to_numeric(summary["ess_bulk"], errors="coerce").min()
    logger.info(f"Divergences: {n_divergent} / {draws * chains}")
    logger.info(f"Max r_hat: {max_rhat:.4f} (want close to 1.00)")
    logger.info(f"Min ESS (bulk): {min_ess:.0f} (want > ~400 per chain-equivalent)")

    tag = "legacy" if legacy_only else "all"
    trace_path = OUT_DIR / f"irt_2pl_bayesian_trace_{tag}.nc"
    trace.to_netcdf(trace_path)
    logger.success(f"Saved trace -> {trace_path}")

    ci_cols = [c for c in summary.columns if c.startswith(("eti", "hdi"))]
    theta_summary = summary.loc[[f"theta[{i}]" for i in range(len(subjects))]].copy()
    theta_summary.index = subjects
    logger.info(
        "\nPosterior theta (subject ability):\n"
        + theta_summary[["mean", "sd", *ci_cols]].to_string()
    )

    summary.to_csv(OUT_DIR / f"irt_2pl_bayesian_summary_{tag}.csv")
    logger.success(
        f"Saved summary -> {OUT_DIR / f'irt_2pl_bayesian_summary_{tag}.csv'}"
    )


if __name__ == "__main__":
    fire.Fire(main)
