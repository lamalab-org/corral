"""
Final Report Generator (Binomial LFM)

Creates publication-quality visualizations using lama-aesthetics style:
1. Model comparison (LOO bar chart)
2. Variance decomposition
3. Posterior distributions
4. LOO predictions by outcome (scatter + jitter)
5. Task-averaged LOO predictions
6. Environment-level LOO predictions
7. Calibration curve

Usage:
    cd corral/plots/panel_4
    uv run python final_report.py
    uv run python final_report.py --best-model model3_abilities_env
"""

import importlib.util
from pathlib import Path

import arviz as az
import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, ONE_COL_WIDTH, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from scipy import stats

# Load plot_config from analysis/ via importlib
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLOT_CONFIG_PATH = REPO_ROOT / "analysis" / "plot_config.py"
_spec = importlib.util.spec_from_file_location("plot_config", PLOT_CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load plot config from {PLOT_CONFIG_PATH}")
plot_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_config)

ENVIRONMENT_NAMES = plot_config.ENVIRONMENT_NAMES
ENVIRONMENT_COLOURS = plot_config.ENVIRONMENT_COLOURS
FONT_SIZES = plot_config.FONT_SIZES

# Colours not in corral's plot_config — define locally
PRIMARY = "#5f59d0"
SECONDARY = "#e84ab5"
ACCENT = "#ff0677"
POSITIVE = "#0051ff"
NEGATIVE = "#ff0677"
OUTCOME_COLOURS = ["#7c3aed", "#2563eb", "#0d9488", "#ca8a04", "#dc2626", "#16a34a"]

# LFM model display names
LFM_MODEL_NAMES = {
    "model1_baseline_tasks": "M1: Baseline",
    "model2_tasks_environment": "M2: Tasks + Env",
    "model3_abilities_env": "M3: AbilitiesxEnv",
    "model4_scaffold_env": "M4: ScaffoldxEnv",
    "model5_scaffold_level": "M5: ScaffoldxLevel",
    "model6_env_level": "M6: EnvxLevel",
    "model7_abilities_env_level": "M7: AbilitiesxEnv-Level",
    "model8_abilities_env_envlevel_intercept": "M8: Hybrid",
}

# Default paths (relative to corral repo root)
RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "lfm-binomial"

# Apply lama-aesthetics style
lama_aesthetics.get_style("main")


def load_comparison_table(results_dir):
    return pd.read_csv(Path(results_dir) / "model_comparison.csv")


def load_trace(model_name, results_dir):
    trace_path = Path(results_dir) / f"{model_name}_trace.nc"
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    return az.from_netcdf(trace_path)


def load_data(results_dir):
    return pd.read_csv(Path(results_dir) / "prepared_data.csv")


def _save(fig, path):
    """Save as both PNG and PDF."""
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def plot_model_comparison(comparison_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comp_df = comparison_df[comparison_df["converged"]].copy()
    if len(comp_df) == 0:
        logger.warning("No converged models to plot")
        return

    comp_df = comp_df.sort_values("loo", ascending=True)
    comp_df["display"] = comp_df["model"].map(LFM_MODEL_NAMES).fillna(comp_df["model"])

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.6))

    y_pos = np.arange(len(comp_df))
    best_idx = comp_df["loo"].idxmax()

    colors = [PRIMARY if idx != best_idx else SECONDARY for idx in comp_df.index]

    ax.barh(
        y_pos,
        comp_df["loo"],
        xerr=comp_df["loo_se"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
        alpha=0.85,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp_df["display"], fontsize=FONT_SIZES["tick_label"])
    ax.set_xlabel("ELPD-LOO (higher is better)", fontsize=FONT_SIZES["axis_label"])
    ax.set_title(
        "Model Comparison (Binomial)", fontsize=FONT_SIZES["title"], fontweight="bold"
    )

    range_frame(ax, comp_df["loo"].to_numpy(), y_pos, pad=0.1)

    _save(fig, output_dir / "model_comparison.png")


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------


def plot_variance_decomposition(best_model, df, results_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idata = load_trace(best_model, results_dir)
    posterior = idata.posterior

    contributions = {}
    for coef_name, label in [("knowledge", "Knowledge"), ("reasoning", "Reasoning")]:
        total_key = f"{coef_name}_coef_total"
        base_key = f"{coef_name}_coef"
        if total_key in posterior:
            coef_vals = posterior[total_key].mean(dim=["chain", "draw"]).values  # noqa: PD011
            contributions[label] = float(
                (coef_vals * df[f"{coef_name}_z"].to_numpy()).var()
            )
        elif base_key in posterior:
            coef_val = float(posterior[base_key].mean().values)
            contributions[label] = float(
                (coef_val * df[f"{coef_name}_z"].to_numpy()).var()
            )

    effect_label_map = {
        "env_level_effect": "Env x Scope",
    }

    for effect_name in [
        "scaffold_effect",
        "level_effect",
        "verbosity_effect",
        "category_effect",
        "env_level_effect",
        "environment_effect",
        "task_effect",
    ]:
        if effect_name in posterior:
            effect_var = float(posterior[effect_name].var().mean().values)
            label = effect_label_map.get(
                effect_name, effect_name.replace("_effect", "").title()
            )
            contributions[label] = effect_var

    if not contributions:
        logger.warning("No variance contributions computed")
        return

    total_var = sum(contributions.values())
    pct = {k: 100 * v / total_var for k, v in contributions.items()}
    pct = dict(sorted(pct.items(), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    components = list(pct.keys())
    values = list(pct.values())

    bar_colors = ["#7150e0"] * len(components)

    ax.barh(components, values, color=bar_colors)
    ax.set_xlabel("Variance Explained (%)", fontsize=FONT_SIZES["tick_label"])

    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    for i, val in enumerate(values):
        ax.text(
            val + 0.8, i, f"{val:.1f}%", va="center", fontsize=FONT_SIZES["tick_label"]
        )

    range_frame(ax, np.array(values), np.arange(len(components)), pad=0.1)

    _save(fig, output_dir / f"{best_model}_variance_decomposition.png")


# ---------------------------------------------------------------------------
# Posterior distributions
# ---------------------------------------------------------------------------


def plot_posterior_distributions(best_model, results_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idata = load_trace(best_model, results_dir)
    posterior = idata.posterior

    key_params = [
        "intercept",
        "knowledge_coef",
        "reasoning_coef",
        "knowledge_coef_base",
        "reasoning_coef_base",
    ]
    available = [p for p in key_params if p in posterior]
    if not available:
        logger.warning("No key parameters found")
        return

    fig, axes = plt.subplots(
        len(available),
        1,
        figsize=(ONE_COL_WIDTH * 1.5, ONE_COL_HEIGHT * len(available)),
    )
    if len(available) == 1:
        axes = [axes]

    for ax, param in zip(axes, available, strict=False):
        samples = posterior[param].values.flatten()  # noqa: PD011
        hdi = az.hdi(samples, hdi_prob=0.9)

        ax.hist(
            samples, bins=50, alpha=0.7, color=PRIMARY, edgecolor="black", linewidth=0.3
        )
        ax.axvline(
            samples.mean(), color=SECONDARY, linestyle="--", linewidth=1.5, label="Mean"
        )
        ax.axvline(hdi[0], color=ACCENT, linestyle=":", linewidth=1.5, label="90% HDI")
        ax.axvline(hdi[1], color=ACCENT, linestyle=":", linewidth=1.5)

        ax.set_xlabel(param, fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Frequency", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            f"{param}: \u03bc={samples.mean():.3f}, 90% HDI=[{hdi[0]:.3f}, {hdi[1]:.3f}]",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(fontsize=FONT_SIZES["legend"])

    plt.tight_layout()
    _save(fig, output_dir / f"{best_model}_posterior_distributions.png")


# ---------------------------------------------------------------------------
# LOO predictions (PSIS importance reweighting)
# ---------------------------------------------------------------------------


def compute_loo_predicted_probs(idata):
    """PSIS importance-weighted LOO predictions of p."""
    loo = az.loo(idata, pointwise=True)

    log_lik = idata.log_likelihood["y_obs"].values  # noqa: PD011
    p_samples = idata.posterior["p"].values  # noqa: PD011

    n_chains, n_draws, n_obs = log_lik.shape
    S = n_chains * n_draws
    log_lik_flat = log_lik.reshape(S, n_obs)
    p_flat = p_samples.reshape(S, n_obs)

    p_loo = np.zeros(n_obs)
    for i in range(n_obs):
        log_w = -log_lik_flat[:, i].copy()
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        p_loo[i] = np.dot(w, p_flat[:, i])

    return p_loo, loo


# ---------------------------------------------------------------------------
# LOO predictions by observed outcome (jitter scatter)
# ---------------------------------------------------------------------------


def plot_task_level_predictions(best_model, df, p_loo, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df.copy()
    df_plot["p_loo"] = p_loo
    df_plot["obs_rate"] = df_plot["k_success"] / df_plot["n_trials"]
    df_plot["obs_rate_rounded"] = df_plot["obs_rate"].round(2)

    outcome_order = sorted(df_plot["obs_rate_rounded"].unique())
    outcome_labels = [f"{r:.0%}" for r in outcome_order]

    oc = (OUTCOME_COLOURS * 3)[: len(outcome_order)]

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    medians = []
    for i, o in enumerate(outcome_order):
        vals = df_plot[df_plot["obs_rate_rounded"] == o]["p_loo"].to_numpy()
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(
            i + jitter,
            vals,
            s=3,
            alpha=0.2,
            color=oc[i],
            edgecolors="none",
            zorder=2,
        )
        medians.append(np.median(vals))

    ax.scatter(
        range(len(outcome_order)),
        medians,
        color=SECONDARY,
        s=30,
        zorder=5,
        marker="_",
        linewidths=1.5,
        label="Median prediction",
    )

    ax.plot(
        range(len(outcome_order)),
        outcome_order,
        "k--",
        linewidth=1,
        alpha=0.6,
        label="Perfect calibration",
        zorder=4,
    )
    ax.scatter(
        range(len(outcome_order)),
        outcome_order,
        color="black",
        s=10,
        zorder=5,
        marker="o",
    )

    ax.set_xticks(range(len(outcome_order)))
    ax.set_xticklabels(outcome_labels, fontsize=FONT_SIZES["tick_label"])
    ax.set_xlabel("Observed Success Rate (k/n)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("LOO Predicted Probability", fontsize=FONT_SIZES["axis_label"])
    ax.set_title(
        "LOO Predictions by Observed Outcome",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )
    ax.legend(fontsize=FONT_SIZES["legend"], loc="upper left")

    range_frame(
        ax, np.arange(len(outcome_order), dtype=float), np.array([0.0, 1.0]), pad=0.05
    )

    _save(fig, output_dir / f"{best_model}_loo_predictions_by_outcome.png")


# ---------------------------------------------------------------------------
# Task-averaged LOO scatter
# ---------------------------------------------------------------------------


def plot_task_averaged_predictions(best_model, df, p_loo, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df.copy()
    df_plot["p_loo"] = p_loo
    df_plot["obs_rate"] = df_plot["k_success"] / df_plot["n_trials"]

    task_col = "task" if "task" in df.columns else "task_id"
    task_data = (
        df_plot.groupby(task_col)
        .agg(
            predicted=("p_loo", "mean"),
            observed=("obs_rate", "mean"),
            n=("n_trials", "sum"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH))

    ax.scatter(
        task_data["predicted"],
        task_data["observed"],
        s=30,
        alpha=0.6,
        color=PRIMARY,
        label="Tasks",
        edgecolors="none",
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="Perfect prediction")

    r, p_val = stats.pearsonr(task_data["predicted"], task_data["observed"])

    ax.set_xlabel("Predicted Success Rate", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Observed Success Rate", fontsize=FONT_SIZES["axis_label"])
    ax.set_title(
        "Task-Averaged LOO Predictions",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )
    ax.legend(fontsize=FONT_SIZES["legend"], loc="lower right", framealpha=0.9)

    range_frame(ax, np.array([0, 1]), np.array([0, 1]), pad=0.05)

    _save(fig, output_dir / f"{best_model}_task_averaged_predictions.png")


# ---------------------------------------------------------------------------
# Environment-level LOO (bar plot)
# ---------------------------------------------------------------------------


def plot_environment_predictions(best_model, df, p_loo, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df.copy()
    df_plot["p_loo"] = p_loo
    df_plot["obs_rate"] = df_plot["k_success"] / df_plot["n_trials"]

    env_data = (
        df_plot.groupby("environment")
        .apply(
            lambda g: pd.Series(
                {
                    "predicted": np.average(g["p_loo"], weights=g["n_trials"]),
                    "observed": np.average(g["obs_rate"], weights=g["n_trials"]),
                }
            )
        )
        .reset_index()
    )
    env_data["display"] = (
        env_data["environment"].map(ENVIRONMENT_NAMES).fillna(env_data["environment"])
    )
    env_data = env_data.sort_values("observed", ascending=False)

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.2))

    x = np.arange(len(env_data))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        env_data["observed"],
        width,
        label="Observed",
        color=PRIMARY,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        env_data["predicted"],
        width,
        label="LOO Predicted",
        color=SECONDARY,
        edgecolor="black",
        linewidth=0.5,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xlabel("Environment", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Success Rate", fontsize=FONT_SIZES["axis_label"])
    ax.set_title(
        "Environment-Level: LOO vs Observed",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        env_data["display"], rotation=45, ha="right", fontsize=FONT_SIZES["tick_label"]
    )
    ax.legend(fontsize=FONT_SIZES["legend"])

    range_frame(ax, x, np.array([0, 1]), pad=0.1)

    _save(fig, output_dir / f"{best_model}_environment_predictions.png")


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------


def plot_calibration(best_model, df, p_loo, output_dir, n_bins=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_rate = df["k_success"].to_numpy() / df["n_trials"].to_numpy()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.clip(np.digitize(p_loo, bins) - 1, 0, n_bins - 1)

    observed_freqs = []
    bin_counts = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            observed_freqs.append(obs_rate[mask].mean())
            bin_counts.append(mask.sum())
        else:
            observed_freqs.append(np.nan)
            bin_counts.append(0)

    valid = ~np.isnan(observed_freqs)

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT))

    ax.plot(
        bin_centers[valid],
        np.array(observed_freqs)[valid],
        marker="o",
        markersize=6,
        linewidth=2,
        color=PRIMARY,
        label="Model calibration",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Perfect calibration")

    # Prediction histogram inset
    ax_hist = ax.inset_axes([0.1, 0.02, 0.8, 0.15])
    ax_hist.hist(p_loo, bins=bins, color=SECONDARY, alpha=0.5, edgecolor="black")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.spines["left"].set_visible(False)
    ax_hist.set_xlabel("Predicted P(success)", fontsize=6)

    ax.set_xlabel("Predicted P(success)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Observed Frequency", fontsize=FONT_SIZES["axis_label"])
    ax.set_title("Calibration Curve", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.legend(fontsize=FONT_SIZES["legend"])

    range_frame(ax, np.array([0, 1]), np.array([0, 1]), pad=0.05)

    _save(fig, output_dir / f"{best_model}_calibration.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(best_model=None, results_dir=None, output_dir=None):
    """
    Generate final report plots from lfm-binomial results.

    Args:
        best_model: Override best model (auto-detected if None)
        results_dir: Path to lfm-binomial results (default: analysis/results/lfm-binomial)
        output_dir: Output directory for plots (default: ./output)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output"
    output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info("GENERATING FINAL REPORT (Binomial LFM)")
    logger.info("=" * 60)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Output dir:  {output_dir}")

    comparison_df = load_comparison_table(results_dir)

    if best_model is None:
        converged = comparison_df[comparison_df["converged"]]
        if len(converged) == 0:
            logger.error("No converged models found")
            return
        best_model = converged.sort_values("loo", ascending=False).iloc[0]["model"]
        logger.info(f"Auto-detected best model: {best_model}")

    data_df = load_data(results_dir)

    logger.info("\nGenerating visualizations...")

    plot_model_comparison(comparison_df, output_dir)
    plot_variance_decomposition(best_model, data_df, results_dir, output_dir)
    plot_posterior_distributions(best_model, results_dir, output_dir)

    # Compute LOO predictions once, reuse across plots
    logger.info("Computing LOO predictions (PSIS reweighting)...")
    idata = load_trace(best_model, results_dir)
    p_loo, _ = compute_loo_predicted_probs(idata)

    plot_task_level_predictions(best_model, data_df, p_loo, output_dir)
    plot_task_averaged_predictions(best_model, data_df, p_loo, output_dir)
    plot_environment_predictions(best_model, data_df, p_loo, output_dir)
    plot_calibration(best_model, data_df, p_loo, output_dir)

    logger.success("\nFinal report generation complete")
    logger.info(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
