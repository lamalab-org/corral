"""Compare multiple samplers' budget sweeps head to head.

Reads several sweep CSVs from run_subsampling_sweep.py (one per sampler) and
plots, per metric, median + IQR band vs. budget with one line per sampler:
  - global MSE                  - global Spearman rho
  - mean per-environment MSE    - mean per-environment Spearman rho

This is the direct comparison for "does stratified sampling beat plain
random", separate from plot_subsampling_results.py (which compares
global-vs-per-environment granularity for a single sampler).

Usage:
    python plot_sampler_comparison.py
    python plot_sampler_comparison.py --results=random,stratified_proportional,stratified_equal
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

SAMPLER_COLOURS = {
    "random": "#7150e0",
    "stratified_proportional": "#16a34a",
    "stratified_equal": "#ea580c",
    "stratified_proportional_nested": "#2563eb",
}


def _band(df: pd.DataFrame, col: str, group: str = "budget") -> pd.DataFrame:
    g = df.groupby(group)[col]
    return pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75)})


def plot_comparison(results: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    panels = [
        (axes[0, 0], "global_mse", "Global MSE", "MSE"),
        (axes[0, 1], "mean_per_env_mse", "Mean per-environment MSE", "MSE"),
        (axes[1, 0], "global_spearman", "Global ranking preservation", "Spearman ρ"),
        (axes[1, 1], "mean_per_env_spearman", "Mean per-environment ranking preservation", "Spearman ρ"),
    ]
    for ax, col, title, ylabel in panels:
        for sampler, df in results.items():
            colour = SAMPLER_COLOURS.get(sampler, "#999999")
            b = _band(df, col)
            ax.plot(b.index, b["median"], color=colour, linewidth=2, label=sampler)
            ax.fill_between(b.index, b["p25"], b["p75"], color=colour, alpha=0.15, linewidth=0)
        if "spearman" in col:
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("budget (# items)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)

    n_repeats = next(iter(results.values())).groupby("budget").size().iloc[0]
    fig.suptitle(f"Sampler comparison — median ± IQR over {n_repeats} repeats per budget", fontsize=11)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(
    samplers: str = "random,stratified_proportional,stratified_equal,stratified_proportional_nested",
    tag: str = "excl-resistor",
    output: str | None = None,
) -> None:
    names = [s.strip() for s in samplers.split(",")]
    results = {}
    for name in names:
        path = DATA_DIR / f"subsampling_sweep_{name}_{tag}.csv"
        logger.info(f"Loading {path}")
        results[name] = pd.read_csv(path)

    out = Path(output) if output else OUT_DIR / f"sampler_comparison_{tag}.pdf"
    plot_comparison(results, out)


if __name__ == "__main__":
    fire.Fire(main)
