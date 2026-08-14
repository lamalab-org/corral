"""Fit a regularized (MAP) 2PL IRT model to the corral-mini response data.

P(success | subject j, item i) = sigmoid(a_i * (theta_j - b_i))
  theta_j: subject ability     b_i: item difficulty     a_i: item discrimination (>0)

Fit by penalized maximum likelihood (= MAP under independent Gaussian priors:
theta_j ~ N(0, theta_sd^2), b_i ~ N(0, b_sd^2), log(a_i) ~ N(0, log_a_sd^2)).
With only 12 subjects these priors are load-bearing regularization, not
decoration: an unregularized 2PL MLE blows up (a_i -> inf) on any item that
happens to perfectly split the 12 subjects into pass/fail (complete
separation), which is a real risk at this sample size.

Fit on trial-level Bernoulli counts (k successes / n=5 trials per cell), not
the pre-averaged 5-trial rate — see the earlier discussion on why collapsing
to a point estimate first throws away information the model should use.

Usage:
    python fit_irt_2pl.py
    python fit_irt_2pl.py --exclude_environments=resistor
"""

import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from subsampling_core import DATA_PATH, ITEM_KEY

OUT_DIR = Path(__file__).parent / "data"


def build_trial_counts(
    exclude_environments: list[str] | None = None,
    exclude_models: list[str] | None = None,
    data_path: Path | None = None,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, pd.Series]:
    df = pd.read_csv(data_path or DATA_PATH)
    if exclude_environments:
        df = df[~df["environment"].isin(exclude_environments)]
    if exclude_models:
        df = df[~df["model"].isin(exclude_models)]
    df = df.copy()
    df["subject"] = df["model"] + "__" + df["scaffold"]
    df["item"] = df[ITEM_KEY].astype(str).agg("|".join, axis=1)

    counts = (
        df.groupby(["subject", "item"])
        .agg(k=("success", "sum"), n=("success", "size"))
        .reset_index()
    )
    subjects = sorted(counts["subject"].unique())
    items = sorted(counts["item"].unique())
    s_idx = {s: i for i, s in enumerate(subjects)}
    i_idx = {it: i for i, it in enumerate(items)}

    K = np.zeros((len(subjects), len(items)))
    N = np.zeros((len(subjects), len(items)))
    for row in counts.itertuples():
        K[s_idx[row.subject], i_idx[row.item]] = row.k
        N[s_idx[row.subject], i_idx[row.item]] = row.n

    item_env = (
        df.drop_duplicates("item").set_index("item")["environment"].reindex(items)
    )
    return subjects, items, K, N, item_env


def fit_2pl(
    K: np.ndarray,
    N: np.ndarray,
    theta_sd: float = 1.0,
    b_sd: float = 2.0,
    log_a_sd: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_subj, n_item = K.shape
    rng = np.random.default_rng(seed)
    x0 = np.concatenate(
        [rng.normal(0, 0.1, n_subj), rng.normal(0, 0.1, n_item), np.zeros(n_item)]
    )

    def unpack(x):
        return x[:n_subj], x[n_subj : n_subj + n_item], x[n_subj + n_item :]

    def nll_and_grad(x):
        theta, b, log_a = unpack(x)
        a = np.exp(log_a)
        logits = a[None, :] * (theta[:, None] - b[None, :])
        p = np.clip(1 / (1 + np.exp(-logits)), 1e-9, 1 - 1e-9)

        ll = np.sum(K * np.log(p) + (N - K) * np.log(1 - p))
        resid = K - N * p  # d(loglik)/d(logit), shape (n_subj, n_item)

        grad_theta = -(a[None, :] * resid).sum(axis=1) + theta / theta_sd**2
        grad_b = (a[None, :] * resid).sum(axis=0) + b / b_sd**2
        grad_log_a = (
            -(a[None, :] * (theta[:, None] - b[None, :]) * resid).sum(axis=0)
            + log_a / log_a_sd**2
        )

        nll = (
            -ll
            + 0.5 * np.sum(theta**2) / theta_sd**2
            + 0.5 * np.sum(b**2) / b_sd**2
            + 0.5 * np.sum(log_a**2) / log_a_sd**2
        )
        return nll, np.concatenate([grad_theta, grad_b, grad_log_a])

    result = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B")
    theta, b, log_a = unpack(result.x)
    logger.info(
        f"2PL MAP fit: converged={result.success}, nll={result.fun:.1f}, "
        f"iters={result.nit}, msg={result.message}"
    )
    return theta, np.exp(log_a), b


def main(
    output: str | None = None,
    exclude_environments: str = "resistor",
    exclude_models: str = "",
    data_path: str | None = None,
) -> None:
    excluded_envs = [e.strip() for e in exclude_environments.split(",") if e.strip()]
    excluded_models = [m.strip() for m in exclude_models.split(",") if m.strip()]
    subjects, items, K, N, item_env = build_trial_counts(
        exclude_environments=excluded_envs,
        exclude_models=excluded_models,
        data_path=Path(data_path) if data_path else None,
    )
    logger.info(f"{len(subjects)} subjects x {len(items)} items, {int(N.sum())} trials")

    theta, a, b = fit_2pl(K, N)

    subj_df = pd.DataFrame({"subject": subjects, "theta": theta}).sort_values(
        "theta", ascending=False
    )
    item_df = pd.DataFrame(
        {"item": items, "a": a, "b": b, "environment": item_env.to_numpy()}
    )

    out_dir = Path(output) if output else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    subj_path, item_path = (
        out_dir / "irt_2pl_subjects.csv",
        out_dir / "irt_2pl_items.csv",
    )
    subj_df.to_csv(subj_path, index=False)
    item_df.to_csv(item_path, index=False)
    logger.success(f"Saved {subj_path}, {item_path}")

    logger.info("Ability (theta) by subject:\n" + subj_df.to_string(index=False))
    logger.info(
        f"Discrimination (a): mean={a.mean():.3f} std={a.std():.3f} "
        f"range=[{a.min():.3f}, {a.max():.3f}]"
    )
    logger.info(
        f"Difficulty (b): mean={b.mean():.3f} std={b.std():.3f} range=[{b.min():.3f}, {b.max():.3f}]"
    )


if __name__ == "__main__":
    fire.Fire(main)
