"""
Evaluate fitted models (Binomial likelihood).

Usage:
    python evaluate.py --model 3
    python evaluate.py --all
    python evaluate.py --all --plots
    python evaluate.py --help
"""

import sys
import warnings
from pathlib import Path

import arviz as az
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from models import MODEL_NAMES
from sklearn.metrics import roc_auc_score


def load_trace(model_name, results_dir):
    trace_path = Path(results_dir) / f"{model_name}_trace.nc"
    if not trace_path.exists():
        return None
    return az.from_netcdf(trace_path)


def compute_classification_metrics(idata, df):
    """Compute classification metrics for Binomial data."""
    k_success = df["k_success"].to_numpy()
    n_trials = df["n_trials"].to_numpy()

    if "p" not in idata.posterior.data_vars:
        return {
            "auc": np.nan,
            "accuracy": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
        }

    p_mean = idata.posterior["p"].values.mean(axis=(0, 1))  # noqa: PD011

    if len(p_mean) != len(k_success):
        logger.warning(f"Shape mismatch: p={len(p_mean)}, data={len(k_success)}")
        return {
            "auc": np.nan,
            "accuracy": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
        }

    # Expand to trial-level for AUC/accuracy
    y_true_expanded = []
    p_expanded = []
    for i in range(len(k_success)):
        n = int(n_trials[i])
        k = int(k_success[i])
        y_true_expanded.extend([1] * k + [0] * (n - k))
        p_expanded.extend([p_mean[i]] * n)

    y_true_expanded = np.array(y_true_expanded)
    p_expanded = np.array(p_expanded)

    try:
        auc = roc_auc_score(y_true_expanded, p_expanded)
    except ValueError:
        auc = np.nan

    accuracy = ((p_expanded > 0.5).astype(int) == y_true_expanded).mean()

    # Brier score at aggregated level (weighted)
    observed_rate = k_success / n_trials
    brier = np.average((p_mean - observed_rate) ** 2, weights=n_trials)

    # Log-loss at aggregated level
    eps = 1e-15
    p_clip = np.clip(p_mean, eps, 1 - eps)
    log_loss = -np.average(
        observed_rate * np.log(p_clip) + (1 - observed_rate) * np.log(1 - p_clip),
        weights=n_trials,
    )

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "brier_score": float(brier),
        "log_loss": float(log_loss),
    }


def get_model_metrics(model_name, idata, df=None):
    metrics = {"model": model_name}

    diag_vars = [v for v in idata.posterior.data_vars if v != "p"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rhat = az.rhat(idata, var_names=diag_vars)
        rhat_vals = [float(rhat[v].max().values) for v in rhat.data_vars]
        rhat_max = max(rhat_vals) if rhat_vals else 1.0

        ess_bulk = az.ess(idata, var_names=diag_vars, method="bulk")
        ess_vals = [float(ess_bulk[v].min().values) for v in ess_bulk.data_vars]
        ess_min = min(ess_vals) if ess_vals else 0.0

    divergences = int(idata.sample_stats.diverging.sum().item())
    converged = (rhat_max < 1.01) and (ess_min > 400) and (divergences == 0)

    metrics.update(
        {
            "converged": converged,
            "rhat_max": rhat_max,
            "ess_min": ess_min,
            "divergences": divergences,
        }
    )

    try:
        loo = az.loo(idata, pointwise=True)
        metrics.update(
            {
                "loo": float(loo.elpd_loo),
                "loo_se": float(loo.se),
                "p_loo": float(loo.p_loo),
                "pareto_k_bad": int((loo.pareto_k > 0.7).sum()),
            }
        )
    except Exception as e:
        logger.warning(f"LOO failed for {model_name}: {e}")
        metrics.update(
            {"loo": np.nan, "loo_se": np.nan, "p_loo": np.nan, "pareto_k_bad": np.nan}
        )

    try:
        waic = az.waic(idata)
        metrics.update({"waic": float(waic.elpd_waic), "waic_se": float(waic.se)})
    except Exception as e:
        logger.warning(f"WAIC failed for {model_name}: {e}")
        metrics.update({"waic": np.nan, "waic_se": np.nan})

    if df is not None:
        metrics.update(compute_classification_metrics(idata, df))

    return metrics


def format_table(metrics_df):
    logger.info("\n" + "=" * 100)
    logger.info("MODEL COMPARISON (Binomial)")
    logger.info("=" * 100)

    t = metrics_df.copy()
    t["converged"] = t["converged"].map({True: "Y", False: "N"})

    cols = [
        "model",
        "converged",
        "loo",
        "loo_se",
        "p_loo",
        "auc",
        "accuracy",
        "brier_score",
        "rhat_max",
        "ess_min",
        "divergences",
    ]
    cols = [c for c in cols if c in t.columns]

    logger.info("\n" + t[cols].to_string(index=False, float_format="%.3f"))

    converged = metrics_df[metrics_df["converged"]]
    if len(converged) > 0:
        best = converged.sort_values("loo", ascending=False).iloc[0]
        logger.info(
            f"\nBest (highest LOO): {best['model']} — LOO={best['loo']:.1f}, AUC={best.get('auc', 'N/A')}"
        )


def make_plots(metrics_df, _az_compare_df, _traces_dict, _df, output_dir):
    """Generate comparison plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ELPD vs complexity
    valid = metrics_df.dropna(subset=["loo", "p_loo"])
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["green" if c else "red" for c in valid["converged"]]
        ax.scatter(
            valid["p_loo"],
            valid["loo"],
            c=colors,
            s=120,
            zorder=3,
            edgecolors="black",
            linewidth=0.5,
        )
        for _, row in valid.iterrows():
            ax.annotate(
                row["model"].replace("model", "M"),
                (row["p_loo"], row["loo"]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
            )
        ax.set_xlabel("Effective Parameters (p_loo)")
        ax.set_ylabel("ELPD-LOO")
        ax.set_title("Model Performance vs Complexity (Binomial)")
        ax.grid(alpha=0.3)
        plt.savefig(output_dir / "elpd_vs_complexity.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_dir / 'elpd_vs_complexity.png'}")


def evaluate_single(
    model_num, results_dir="results/", data="results/prepared_data.csv"
):
    """Evaluate a single fitted model."""
    model_name = MODEL_NAMES.get(int(model_num))
    if not model_name:
        logger.error(
            f"Unknown model {model_num}. Available: {sorted(MODEL_NAMES.keys())}"
        )
        sys.exit(1)

    idata = load_trace(model_name, results_dir)
    if idata is None:
        logger.error(f"No trace found for {model_name} in {results_dir}")
        sys.exit(1)

    data_df = pd.read_csv(data)
    metrics = get_model_metrics(model_name, idata, df=data_df)

    logger.info(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        if k == "model":
            continue
        if isinstance(v, float):
            logger.info(f"  {k:20s}: {v:.4f}")
        else:
            logger.info(f"  {k:20s}: {v}")

    return metrics


def evaluate_all(results_dir="results/", data="results/prepared_data.csv", plots=False):
    """Compare all fitted models."""
    results_dir = Path(results_dir)
    data_df = pd.read_csv(data)

    all_metrics = []
    traces_dict = {}

    for _num, name in sorted(MODEL_NAMES.items()):
        idata = load_trace(name, results_dir)
        if idata is None:
            continue
        logger.info(f"Processing {name}...")
        metrics = get_model_metrics(name, idata, df=data_df)
        all_metrics.append(metrics)
        traces_dict[name] = idata

    if not all_metrics:
        logger.error("No fitted models found")
        sys.exit(1)

    metrics_df = pd.DataFrame(all_metrics).sort_values("loo", ascending=False)
    format_table(metrics_df)

    # ArviZ comparison
    az_compare_df = None
    if len(traces_dict) >= 2:
        try:
            az_compare_df = az.compare(
                traces_dict, ic="loo", method="stacking", scale="log"
            )
            logger.info(f"\nArviZ comparison:\n{az_compare_df.to_string()}")
        except Exception as e:
            logger.warning(f"az.compare failed: {e}")

    # Save
    metrics_df.to_csv(results_dir / "model_comparison.csv", index=False)
    logger.info(f"Saved: {results_dir / 'model_comparison.csv'}")

    if plots:
        make_plots(
            metrics_df,
            az_compare_df,
            traces_dict,
            data_df,
            output_dir=results_dir / "plots",
        )

    return metrics_df


def main(
    model=None,
    compare_all=False,
    results_dir="results/",
    data="results/prepared_data.csv",
    plots=False,
):
    """
    Evaluate fitted models (Binomial).

    Args:
        model: Model number to evaluate (1-8)
        compare_all: Compare all fitted models
        results_dir: Directory containing model traces
        data: Path to prepared data
        plots: Generate comparison plots (only with --all)
    """
    if model is not None:
        return evaluate_single(model, results_dir=results_dir, data=data)
    elif compare_all:
        return evaluate_all(results_dir=results_dir, data=data, plots=plots)
    else:
        logger.error("Specify --model N or --all")
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)
