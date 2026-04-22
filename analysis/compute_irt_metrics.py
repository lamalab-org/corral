"""
Compute predictive performance metrics from fitted IRT models.

Usage:
    python compute_irt_metrics.py --model_dir=results/irt_category_task
"""

import json
from pathlib import Path

import arviz as az
import fire
import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import expit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def compute_metrics(model_dir: str = "results/irt_category_task"):
    """Compute various predictive performance metrics.

    Args:
        model_dir: Directory with IRT model results
    """
    model_dir = Path(model_dir)
    logger.info(f"Loading model from {model_dir}")

    # Load trace and data
    trace = az.from_netcdf(model_dir / "agent_trace.nc")
    agent_df = pd.read_csv(model_dir / "agent_df.csv")

    with (model_dir / "results.json").open() as f:
        results = json.load(f)

    logger.success("Loaded model data")

    # =========================================================================
    # 1. Variance Decomposition (already computed)
    # =========================================================================
    var_decomp = results["variance_decomposition"]

    logger.info("\n" + "=" * 80)
    logger.info("VARIANCE DECOMPOSITION")
    logger.info("=" * 80)
    for component, pct in sorted(var_decomp.items(), key=lambda x: -x[1]):
        logger.info(f"  {component:12s}: {pct:6.2f}%")

    # =========================================================================
    # 2. PSIS-LOO (Leave-One-Out Cross-Validation)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PSIS-LOO CROSS-VALIDATION")
    logger.info("=" * 80)

    try:
        loo = az.loo(trace, var_name="obs")

        logger.info(f"  ELPD LOO:     {loo.elpd_loo:.2f} ± {loo.se:.2f}")
        logger.info(f"  LOO IC:       {loo.loo:.2f}")
        logger.info(f"  p_loo:        {loo.p_loo:.2f} (effective # parameters)")

        # Check for problematic observations
        pareto_k = loo.pareto_k
        n_bad = np.sum(pareto_k > 0.7)
        n_very_bad = np.sum(pareto_k > 1.0)

        logger.info(
            f"  Pareto k > 0.7:  {n_bad}/{len(pareto_k)} ({100*n_bad/len(pareto_k):.1f}%)"
        )
        logger.info(
            f"  Pareto k > 1.0:  {n_very_bad}/{len(pareto_k)} ({100*n_very_bad/len(pareto_k):.1f}%)"
        )

        if n_very_bad > 0:
            logger.warning("  ⚠ Some observations have very high Pareto k (> 1.0)")
            logger.warning("    LOO estimates may be unreliable for these points")

        # Save LOO results
        loo_results = {
            "elpd_loo": float(loo.elpd_loo),
            "se": float(loo.se),
            "loo_ic": float(loo.loo),
            "p_loo": float(loo.p_loo),
            "n_pareto_k_high": int(n_bad),
            "n_pareto_k_very_high": int(n_very_bad),
        }

        with (model_dir / "loo_results.json").open("w") as f:
            json.dump(loo_results, f, indent=2)

        logger.success(f"Saved LOO results to {model_dir / 'loo_results.json'}")

    except Exception as e:
        logger.error(f"Failed to compute LOO: {e}")
        loo_results = None

    # =========================================================================
    # 3. In-Sample Prediction Accuracy
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("IN-SAMPLE PREDICTION ACCURACY")
    logger.info("=" * 80)

    # Get posterior predictive samples
    y_obs = agent_df["success"].to_numpy()

    # Get posterior mean predictions
    obs_samples = trace.posterior["obs"].to_numpy()  # shape: (chains, draws, N)
    obs_mean = obs_samples.mean(axis=(0, 1))  # mean across chains and draws

    # Convert logits to probabilities
    p_success = expit(obs_mean)

    # Compute metrics
    y_pred = (p_success > 0.5).astype(int)
    accuracy = accuracy_score(y_obs, y_pred)
    logloss = log_loss(y_obs, p_success)

    roc_auc = roc_auc_score(y_obs, p_success) if len(np.unique(y_obs)) > 1 else None

    logger.info(f"  Accuracy:     {accuracy:.4f}")
    logger.info(f"  Log Loss:     {logloss:.4f}")
    if roc_auc:
        logger.info(f"  ROC AUC:      {roc_auc:.4f}")

    # Calibration: bin predictions and compare to actual rates
    logger.info("\n  Calibration (binned):")
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(p_success, bins)

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            pred_prob = p_success[mask].mean()
            actual_rate = y_obs[mask].mean()
            n_samples = mask.sum()
            logger.info(
                f"    [{bins[i-1]:.1f}-{bins[i]:.1f}]: pred={pred_prob:.3f}, actual={actual_rate:.3f}, n={n_samples}"
            )

    # Save in-sample metrics
    in_sample_metrics = {
        "accuracy": float(accuracy),
        "log_loss": float(logloss),
        "roc_auc": float(roc_auc) if roc_auc else None,
        "n_samples": len(y_obs),
        "mean_pred_prob": float(p_success.mean()),
        "mean_true_rate": float(y_obs.mean()),
    }

    with (model_dir / "in_sample_metrics.json").open("w") as f:
        json.dump(in_sample_metrics, f, indent=2)

    logger.success(f"Saved in-sample metrics to {model_dir / 'in_sample_metrics.json'}")

    # =========================================================================
    # 4. Per-Environment Accuracy
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PER-ENVIRONMENT ACCURACY")
    logger.info("=" * 80)

    env_metrics = []
    for env in sorted(agent_df["environment"].unique()):
        mask = agent_df["environment"] == env
        env_accuracy = accuracy_score(y_obs[mask], y_pred[mask])
        env_n = mask.sum()
        env_metrics.append(
            {
                "environment": env,
                "accuracy": float(env_accuracy),
                "n_samples": int(env_n),
            }
        )
        logger.info(f"  {env:15s}: {env_accuracy:.4f} (n={env_n})")

    with (model_dir / "env_metrics.json").open("w") as f:
        json.dump(env_metrics, f, indent=2)

    logger.success(f"Saved per-environment metrics to {model_dir / 'env_metrics.json'}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model Directory: {model_dir}")
    logger.info(f"N Samples:       {len(y_obs)}")
    logger.info(f"In-Sample Acc:   {accuracy:.4f}")
    logger.info(f"In-Sample Loss:  {logloss:.4f}")
    if loo_results:
        logger.info(
            f"LOO ELPD:        {loo_results['elpd_loo']:.2f} ± {loo_results['se']:.2f}"
        )
    logger.info("\nKey Insight:")
    logger.info(
        f"  Dominant factor: {max(var_decomp.items(), key=lambda x: x[1])[0]} ({max(var_decomp.values()):.1f}%)"
    )


if __name__ == "__main__":
    fire.Fire(compute_metrics)
