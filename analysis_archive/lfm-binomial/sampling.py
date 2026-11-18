"""
Sampling configuration and utilities.

Common sampling settings, convergence checks, and result saving for all models.
"""

import json
import warnings
from pathlib import Path

import arviz as az
import pandas as pd
import pymc as pm
from loguru import logger

DEFAULT_SAMPLING_CONFIG = {
    "draws": 2000,
    "tune": 2000,
    "chains": 4,
    "cores": 8,
    "target_accept": 0.95,
    "init": "adapt_diag",
    "return_inferencedata": True,
    "progressbar": True,
    "idata_kwargs": {"log_likelihood": True},
}


def sample_model(model, **kwargs):
    """Sample from a PyMC model with robust settings."""
    config = DEFAULT_SAMPLING_CONFIG.copy()
    config.update(kwargs)

    logger.info("\n=== Sampling Configuration ===")
    for key, value in config.items():
        if key != "idata_kwargs":
            logger.info(f"  {key}: {value}")

    with model:
        return pm.sample(**config)


def check_convergence(idata, model_name="Model"):
    """Check convergence diagnostics (Rhat, ESS, divergences)."""

    logger.info(f"\n=== Convergence Diagnostics: {model_name} ===")

    diag_vars = [v for v in idata.posterior.data_vars if v != "p"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rhat = az.rhat(idata, var_names=diag_vars)
            rhat_vals = [float(rhat[v].max().values) for v in rhat.data_vars]
            rhat_max = max(rhat_vals) if rhat_vals else 1.0
            rhat_params_bad = sum(1 for v in rhat_vals if v > 1.01)
        logger.info(f"Rhat max:        {rhat_max:.4f} (target: < 1.01)")
        logger.info(f"Rhat > 1.01:     {rhat_params_bad} parameters")
    except Exception as e:
        logger.warning(f"Failed to compute Rhat: {e}")
        rhat_max = 1.0
        rhat_params_bad = 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ess_bulk = az.ess(idata, var_names=diag_vars, method="bulk")
            ess_vals = [float(ess_bulk[v].min().values) for v in ess_bulk.data_vars]
            ess_min = min(ess_vals) if ess_vals else 0.0
            ess_params_bad = sum(1 for v in ess_vals if v < 400)
        logger.info(f"ESS min:         {ess_min:.0f} (target: > 400)")
        logger.info(f"ESS < 400:       {ess_params_bad} parameters")
    except Exception as e:
        logger.warning(f"Failed to compute ESS: {e}")
        ess_min = 0.0
        ess_params_bad = 1

    divergences = idata.sample_stats.diverging.sum().item()
    logger.info(f"Divergences:     {divergences} (target: 0)")

    converged = (rhat_max < 1.01) and (ess_min > 400) and (divergences == 0)
    status = "CONVERGED" if converged else "FAILED"
    logger.info(f"Status:          {status}")

    if not converged:
        if rhat_max >= 1.01:
            logger.warning("  - Chains have not converged (Rhat >= 1.01)")
        if ess_min <= 400:
            logger.warning("  - Insufficient effective sample size (ESS <= 400)")
        if divergences > 0:
            logger.warning(f"  - {divergences} divergent transitions detected")

    return {
        "converged": converged,
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "divergences": divergences,
    }


def save_model_results(idata, model_name, output_dir="results", save_extras=True):
    """Save trace, summary, metadata, and optionally WAIC/LOO."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n=== Saving {model_name} Results ===")

    # Trace
    trace_path = output_dir / f"{model_name}_trace.nc"
    idata.to_netcdf(trace_path)
    logger.info(f"Trace saved: {trace_path} ({trace_path.stat().st_size / 1e6:.1f} MB)")

    # Summary
    summary_vars = [v for v in idata.posterior.data_vars if v != "p"]
    summary = az.summary(idata, var_names=summary_vars)
    summary_path = output_dir / f"{model_name}_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Summary saved: {summary_path}")

    # Metadata
    metadata = {
        "model_name": model_name,
        "n_chains": len(idata.posterior.chain),
        "n_draws": len(idata.posterior.draw),
        "parameters": list(idata.posterior.data_vars.keys()),
        "n_parameters": len(idata.posterior.data_vars),
    }
    metadata_path = output_dir / f"{model_name}_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {metadata_path}")

    if save_extras:
        # WAIC
        try:
            waic = az.waic(idata, pointwise=True)
            pd.DataFrame(
                {
                    "elpd_waic": [waic.elpd_waic],
                    "se": [waic.se],
                    "p_waic": [waic.p_waic],
                }
            ).to_csv(output_dir / f"{model_name}_waic.csv", index=False)
            logger.info(f"WAIC saved (ELPD={waic.elpd_waic:.1f})")
        except Exception as e:
            logger.warning(f"Failed to compute WAIC: {e}")

        # LOO
        try:
            loo = az.loo(idata, pointwise=True)
            pd.DataFrame(
                {
                    "elpd_loo": [loo.elpd_loo],
                    "se": [loo.se],
                    "p_loo": [loo.p_loo],
                }
            ).to_csv(output_dir / f"{model_name}_loo.csv", index=False)
            logger.info(f"LOO saved (ELPD={loo.elpd_loo:.1f})")

            n_high_k = (loo.pareto_k > 0.7).sum()
            if n_high_k > 0:
                logger.warning(f"{n_high_k} observations with Pareto k > 0.7")
        except Exception as e:
            logger.warning(f"Failed to compute LOO: {e}")

    logger.info(f"All results saved for {model_name}")
    return {
        "trace_path": trace_path,
        "summary_path": summary_path,
        "metadata_path": metadata_path,
    }
