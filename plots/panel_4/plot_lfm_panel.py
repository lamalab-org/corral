"""
LFM Panel: 3-subplot figure combining LOO predictions, task-averaged scatter,
and variance decomposition.

Usage:
    cd corral/plots/panel_4
    uv run python plot_lfm_panel.py
    uv run python plot_lfm_panel.py --best-model model7_abilities_env_level
"""

import importlib.util
from pathlib import Path

import arviz as az
import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger

# Load plot_config
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "plot_config", REPO_ROOT / "analysis" / "plot_config.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("Could not load plot config")
plot_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_config)

_FS = plot_config.FONT_SIZES
# Unified font size for everything
FONT = _FS["axis_label"]
FONT_SIZES = {k: FONT for k in _FS}
FONT_SIZES["legend"] = FONT  # legend same size as axis labels

# Unified purple
PURPLE = "#7150e0"
SECONDARY = "#e84ab5"

RESULTS_DIR = REPO_ROOT / "analysis" / "results" / "lfm-binomial"

lama_aesthetics.get_style("main")


def load_trace(model_name, results_dir):
    return az.from_netcdf(Path(results_dir) / f"{model_name}_trace.nc")


def load_data(results_dir):
    return pd.read_csv(Path(results_dir) / "prepared_data.csv")


def compute_loo_predicted_probs(idata):
    """PSIS importance-weighted LOO predictions of p."""
    log_lik = idata.log_likelihood["y_obs"].to_numpy()
    p_samples = idata.posterior["p"].to_numpy()

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

    return p_loo


# ---------------------------------------------------------------------------
# Subplot A: LOO predictions by observed outcome (jitter scatter)
# ---------------------------------------------------------------------------


def _plot_loo_by_outcome(ax, df, p_loo):
    df_plot = df.copy()
    df_plot["p_loo"] = p_loo
    df_plot["obs_rate"] = df_plot["k_success"] / df_plot["n_trials"]
    df_plot["obs_rate_rounded"] = df_plot["obs_rate"].round(2)

    outcome_order = sorted(df_plot["obs_rate_rounded"].unique())
    # No % sign; show every other label to avoid crowding
    outcome_labels = []
    for idx, r in enumerate(outcome_order):
        if idx % 2 == 0:
            outcome_labels.append(f"{r:.1f}")
        else:
            outcome_labels.append("")

    medians = []
    for i, o in enumerate(outcome_order):
        vals = df_plot[df_plot["obs_rate_rounded"] == o]["p_loo"].to_numpy()
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(
            i + jitter,
            vals,
            s=3,
            alpha=0.2,
            color=PURPLE,
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
    ax.set_xticklabels(outcome_labels, fontsize=FONT)
    ax.set_xlabel("Observed Success Rate (k/n)", fontsize=FONT)
    ax.set_ylabel("LOO Predicted Probability", fontsize=FONT)
    ax.legend(
        fontsize=FONT - 2,
        loc="lower right",
        framealpha=0.9,
        markerfirst=False,
        bbox_to_anchor=(1.02, 0.0),
    )

    range_frame(
        ax, np.arange(len(outcome_order), dtype=float), np.array([0.0, 1.0]), pad=0.05
    )


# ---------------------------------------------------------------------------
# Subplot B: Task-averaged LOO scatter
# ---------------------------------------------------------------------------


def _plot_task_averaged(ax, df, p_loo):
    df_plot = df.copy()
    df_plot["p_loo"] = p_loo
    df_plot["obs_rate"] = df_plot["k_success"] / df_plot["n_trials"]

    task_col = "task" if "task" in df.columns else "task_id"
    task_data = (
        df_plot.groupby(task_col)
        .agg(
            predicted=("p_loo", "mean"),
            observed=("obs_rate", "mean"),
        )
        .reset_index()
    )

    ax.scatter(
        task_data["predicted"],
        task_data["observed"],
        s=30,
        alpha=0.6,
        color=PURPLE,
        label="Tasks",
        edgecolors="none",
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Predicted Success Rate", fontsize=FONT)
    ax.set_ylabel("Observed Success Rate", fontsize=FONT)
    ax.legend(
        fontsize=FONT - 2,
        loc="lower right",
        framealpha=0.9,
        markerfirst=False,
        bbox_to_anchor=(1.02, 0.0),
    )

    range_frame(ax, np.array([0, 1]), np.array([0, 1]), pad=0.05)


# ---------------------------------------------------------------------------
# Subplot C: Variance decomposition
# ---------------------------------------------------------------------------


def _plot_variance_decomposition(ax, best_model, df, results_dir):
    idata = load_trace(best_model, results_dir)
    posterior = idata.posterior

    contributions = {}
    for coef_name, label in [("knowledge", "Knowledge"), ("reasoning", "Reasoning")]:
        total_key = f"{coef_name}_coef_total"
        base_key = f"{coef_name}_coef"
        if total_key in posterior:
            coef_vals = posterior[total_key].mean(dim=["chain", "draw"]).to_numpy()
            contributions[label] = float(
                (coef_vals * df[f"{coef_name}_z"].to_numpy()).var()
            )
        elif base_key in posterior:
            coef_val = float(posterior[base_key].mean().values)
            contributions[label] = float(
                (coef_val * df[f"{coef_name}_z"].to_numpy()).var()
            )

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
            contributions[effect_name.replace("_effect", "").title()] = effect_var

    if not contributions:
        logger.warning("No variance contributions computed")
        return

    total_var = sum(contributions.values())
    pct = {k: 100 * v / total_var for k, v in contributions.items()}
    pct = dict(sorted(pct.items(), key=lambda x: x[1]))

    components = list(pct.keys())
    values = list(pct.values())

    ax.barh(components, values, color=PURPLE)
    ax.set_xlabel("Variance Explained (%)", fontsize=FONT)

    for i, val in enumerate(values):
        ax.text(val + 0.8, i, f"{val:.1f}%", va="center", fontsize=FONT)

    range_frame(ax, np.array(values), np.arange(len(components)), pad=0.1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(best_model=None, results_dir=None, output_dir=None):
    """
    Generate 3-subplot LFM panel figure.

    Args:
        best_model: Override best model (auto-detected if None)
        results_dir: Path to lfm-binomial results
        output_dir: Output directory for the panel plot
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect best model
    if best_model is None:
        comparison_df = pd.read_csv(results_dir / "model_comparison.csv")
        converged = comparison_df[comparison_df["converged"]]
        if len(converged) == 0:
            logger.error("No converged models found")
            return
        best_model = converged.sort_values("loo", ascending=False).iloc[0]["model"]
        logger.info(f"Auto-detected best model: {best_model}")

    data_df = load_data(results_dir)

    logger.info("Computing LOO predictions (PSIS reweighting)...")
    idata = load_trace(best_model, results_dir)
    p_loo = compute_loo_predicted_probs(idata)

    # Create 3-subplot panel
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 1.1),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    _plot_variance_decomposition(axes[0], best_model, data_df, results_dir)
    _plot_loo_by_outcome(axes[1], data_df, p_loo)
    _plot_task_averaged(axes[2], data_df, p_loo)

    # Add subplot labels
    for ax, label in zip(axes, ["b", "c", "d"], strict=False):
        ax.text(
            -0.15,
            1.12,
            f"({label})",
            transform=ax.transAxes,
            fontsize=FONT,
            fontweight="bold",
            va="top",
        )

    plt.tight_layout()

    out_path = output_dir / "lfm_panel.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved panel to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
