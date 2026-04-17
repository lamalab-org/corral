"""
Forest plot of posterior variance shares for M1–M8 latent factor models.

For each MCMC draw, computes the linear-predictor variance attributable to
each component, converts to percentage shares, and reports the median + 90%
HDI as a forest plot.

Single-model mode:  python plot_m7_variance_forest.py
All-models panel:   python plot_m7_variance_forest.py --all_models

Usage:
    python plot_m7_variance_forest.py [--all_models]
"""

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
from plot_config import FONT_SIZES

RESULTS_DIR = Path(__file__).parent / "results" / "lfm-binomial"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_4"

MEDIAN_COLOUR = "#5f59d0"
HDI_COLOUR = "#b8b5e8"

lama_aesthetics.get_style("main")

# Canonical component labels — full (single-plot) and compact (panel)
COMPONENT_LABELS = {
    "model": r"$\sigma_{\mathrm{model}}$",
    "scaffold": r"$\sigma_{\mathrm{scaffold}}$",
    "verbosity": r"$\sigma_{\mathrm{verbosity}}$",
    "category": r"$\sigma_{\mathrm{category}}$",
    "scope": r"$\sigma_{\mathrm{scope}}$",
    "environment": r"$\sigma_{\mathrm{environment}}$",
    "env_x_scope": r"$\sigma_{\mathrm{env \times scope}}$",
    "scaffold_x_env": r"$\sigma_{\mathrm{scaffold \times env}}$",
    "scaffold_x_scope": r"$\sigma_{\mathrm{scaffold \times scope}}$",
    "task": r"$\sigma_{\mathrm{task}}$",
}

COMPONENT_LABELS_COMPACT = {
    "model": "model",
    "scaffold": "scaffold",
    "verbosity": "verbosity",
    "category": "category",
    "scope": "scope",
    "environment": "env",
    "env_x_scope": "env\u00d7scope",
    "scaffold_x_env": "scaff.\u00d7env",
    "scaffold_x_scope": "scaff.\u00d7scope",
    "task": "task",
}

MODEL_ORDER = [
    ("model1_baseline_tasks", "M1"),
    ("model2_tasks_environment", "M2"),
    ("model3_abilities_env", "M3"),
    ("model4_scaffold_env", "M4"),
    ("model5_scaffold_level", "M5"),
    ("model6_env_level", "M6"),
    ("model7_abilities_env_level", "M7"),
    ("model8_abilities_env_envlevel_intercept", "M8"),
]


def _flat(posterior, name):
    """Flatten (chain, draw, ...) → (total_draws, ...)."""
    arr = posterior[name].values
    shape = arr.shape
    return arr.reshape(shape[0] * shape[1], *shape[2:])


def compute_variance_shares(posterior, df):
    """Compute per-draw variance shares for any of the 8 models.

    Returns (total_draws, n_components) array and list of component keys.
    """
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    total_draws = n_chains * n_draws

    knowledge_z = df["knowledge_z"].to_numpy()
    reasoning_z = df["reasoning_z"].to_numpy()

    # Build component extractors based on available parameters
    components = []  # list of (key, per-draw variance function)

    # --- Model (knowledge + reasoning) ---
    if "knowledge_coef_total" in posterior:
        k_total = _flat(posterior, "knowledge_coef_total")
        r_total = _flat(posterior, "reasoning_coef_total")
        components.append(("model",
            lambda d, kt=k_total, rt=r_total:
                np.var(kt[d] * knowledge_z) + np.var(rt[d] * reasoning_z)))
    elif "knowledge_coef" in posterior:
        k_coef = _flat(posterior, "knowledge_coef")  # (D,) scalar
        r_coef = _flat(posterior, "reasoning_coef")
        components.append(("model",
            lambda d, kc=k_coef, rc=r_coef:
                np.var(kc[d] * knowledge_z) + np.var(rc[d] * reasoning_z)))

    # --- Scaffold ---
    if "scaffold_effect" in posterior:
        idx = df["scaffold_id"].to_numpy()
        eff = _flat(posterior, "scaffold_effect")
        components.append(("scaffold",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    # --- Scaffold × Environment ---
    if "scaffold_env_effect" in posterior:
        idx = df["env_scaffold_id"].to_numpy()
        eff = _flat(posterior, "scaffold_env_effect")
        components.append(("scaffold_x_env",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    # --- Scaffold × Scope ---
    if "scaffold_level_effect" in posterior:
        idx = df["scaffold_level_id"].to_numpy()
        eff = _flat(posterior, "scaffold_level_effect")
        components.append(("scaffold_x_scope",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    # --- Scope (level_effect) ---
    if "level_effect" in posterior:
        idx = df["level_id"].to_numpy()
        eff = _flat(posterior, "level_effect")
        components.append(("scope",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    # --- Verbosity ---
    if "verbosity_effect" in posterior:
        idx = df["verbosity_id"].to_numpy()
        eff = _flat(posterior, "verbosity_effect")
        components.append(("verbosity",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    # --- Category ---
    if "category_effect" in posterior:
        idx = df["category_id"].to_numpy()
        eff = _flat(posterior, "category_effect")
        components.append(("category",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    # --- Environment ---
    if "environment_effect" in posterior:
        idx = df["environment_id"].to_numpy()
        eff = _flat(posterior, "environment_effect")
        components.append(("environment",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    # --- Env × Scope ---
    if "env_level_effect" in posterior:
        idx = df["env_level_id"].to_numpy()
        eff = _flat(posterior, "env_level_effect")
        components.append(("env_x_scope",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    # --- Task ---
    if "task_effect" in posterior:
        idx = df["task_id"].to_numpy()
        eff = _flat(posterior, "task_effect")
        components.append(("task",
            lambda d, e=eff, i=idx: np.var(e[d][i])))

    keys = [k for k, _ in components]
    funcs = [f for _, f in components]
    n_comp = len(components)
    shares = np.empty((total_draws, n_comp))

    for d in range(total_draws):
        variances = np.array([f(d) for f in funcs])
        total = variances.sum()
        shares[d] = 100 * variances / total if total > 0 else 0

    return shares, keys


def plot_single_forest(shares, keys, output_path):
    """Forest plot for a single model."""
    n_comp = len(keys)
    medians = np.median(shares, axis=0)
    hdi_low = np.percentile(shares, 5, axis=0)
    hdi_high = np.percentile(shares, 95, axis=0)

    order = np.argsort(medians)
    medians, hdi_low, hdi_high = medians[order], hdi_low[order], hdi_high[order]
    labels = [COMPONENT_LABELS[keys[i]] for i in order]

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_HEIGHT * 0.85))
    y_pos = np.arange(n_comp)

    for y, lo, hi in zip(y_pos, hdi_low, hdi_high):
        ax.plot([lo, hi], [y, y], color=HDI_COLOUR, linewidth=2.5, solid_capstyle="round")
    ax.scatter(medians, y_pos, color=MEDIAN_COLOUR, s=40, zorder=5,
               edgecolors="white", linewidths=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES["axis_label"])
    ax.set_xlabel("Variance share (%)", fontsize=FONT_SIZES["axis_label"])
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick_label"])

    for y, med, hi in zip(y_pos, medians, hdi_high):
        ax.text(hi + 1.0, y, f"{med:.1f}%", va="center",
                fontsize=FONT_SIZES["tick_label"], color=MEDIAN_COLOUR)

    range_frame(ax, np.concatenate([hdi_low, hdi_high]), y_pos, pad=0.12)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved -> {output_path}")


def _top_k(shares, keys, k=5):
    """Keep only the top-k components by median variance share."""
    medians = np.median(shares, axis=0)
    top_idx = np.argsort(medians)[-k:]  # indices of top-k (ascending)
    return shares[:, top_idx], [keys[i] for i in top_idx]


def plot_all_models_panel(all_results, output_path, top_k=5):
    """2×4 panel of forest plots, one per model, showing top-k components."""
    fig, axes = plt.subplots(2, 4, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT * 2.2),
                             sharey=False)
    axes = axes.flatten()

    # Compute a shared x-limit across all panels
    global_max = 0
    for model_name, _ in MODEL_ORDER:
        if model_name not in all_results:
            continue
        shares, keys = all_results[model_name]
        shares_k, _ = _top_k(shares, keys, k=top_k)
        hi = np.percentile(shares_k, 95, axis=0).max()
        global_max = max(global_max, hi)
    x_max = global_max * 1.25  # pad for annotation text

    for idx, (model_name, display_name) in enumerate(MODEL_ORDER):
        ax = axes[idx]
        if model_name not in all_results:
            ax.set_visible(False)
            continue

        shares, keys = all_results[model_name]
        shares, keys = _top_k(shares, keys, k=top_k)

        n_comp = len(keys)
        medians = np.median(shares, axis=0)
        hdi_low = np.percentile(shares, 5, axis=0)
        hdi_high = np.percentile(shares, 95, axis=0)

        order = np.argsort(medians)
        medians, hdi_low, hdi_high = medians[order], hdi_low[order], hdi_high[order]
        labels = [COMPONENT_LABELS_COMPACT[keys[i]] for i in order]

        y_pos = np.arange(n_comp)

        for y, lo, hi in zip(y_pos, hdi_low, hdi_high):
            ax.plot([lo, hi], [y, y], color=HDI_COLOUR, linewidth=2.0,
                    solid_capstyle="round")
        ax.scatter(medians, y_pos, color=MEDIAN_COLOUR, s=20, zorder=5,
                   edgecolors="white", linewidths=0.3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_title(display_name, fontsize=FONT_SIZES["axis_label"], fontweight="bold",
                     pad=6)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.tick_params(axis="x", labelsize=6.5)

        range_frame(ax, np.concatenate([hdi_low, hdi_high]), y_pos, pad=0.15)

        for y, med, hi in zip(y_pos, medians, hdi_high):
            ax.text(hi + 3.5, y, f"{med:.0f}%", va="center",
                    fontsize=5.5, color=MEDIAN_COLOUR)

    fig.supxlabel("Variance share (%)", fontsize=FONT_SIZES["axis_label"])
    fig.subplots_adjust(hspace=0.5, wspace=0.7)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved -> {output_path}")


def main(all_models=False):
    df = pd.read_csv(RESULTS_DIR / "prepared_data.csv")

    if not all_models:
        # Single M7 forest plot (original behaviour)
        model_name = "model7_abilities_env_level"
        logger.info(f"Loading {model_name} trace...")
        idata = az.from_netcdf(RESULTS_DIR / f"{model_name}_trace.nc")
        logger.info("Computing per-draw variance shares (all 8000 draws)...")
        shares, keys = compute_variance_shares(idata.posterior, df)

        logger.info("Posterior variance shares (median [5%, 95%]):")
        for j, key in enumerate(keys):
            med = np.median(shares[:, j])
            lo = np.percentile(shares[:, j], 5)
            hi = np.percentile(shares[:, j], 95)
            logger.info(f"  {COMPONENT_LABELS[key]:30s}  {med:5.1f}%  [{lo:5.1f}%, {hi:5.1f}%]")

        plot_single_forest(shares, keys, OUT_DIR / "m7_variance_forest.png")
    else:
        # All 8 models in a 2×4 panel
        all_results = {}
        for model_name, display_name in MODEL_ORDER:
            trace_path = RESULTS_DIR / f"{model_name}_trace.nc"
            if not trace_path.exists():
                logger.warning(f"Trace not found: {trace_path}, skipping")
                continue
            logger.info(f"Loading {display_name}...")
            idata = az.from_netcdf(trace_path)
            shares, keys = compute_variance_shares(idata.posterior, df)
            all_results[model_name] = (shares, keys)

            logger.info(f"  {display_name} variance shares (median):")
            for j, key in enumerate(keys):
                med = np.median(shares[:, j])
                logger.info(f"    {COMPONENT_LABELS[key]:30s}  {med:5.1f}%")

        plot_all_models_panel(all_results, OUT_DIR / "variance_forest_all_models.png")


if __name__ == "__main__":
    fire.Fire(main)
