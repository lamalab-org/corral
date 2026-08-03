"""Plot MSE / ranking-preservation / environment-starvation vs. budget.

Reads a sweep CSV from run_subsampling_sweep.py (one row per budget x
repeat) and plots, each as median line + IQR band across repeats:
  1. MSE vs budget — global (item-pooled) vs mean-per-environment.
  2. Spearman rank correlation (true vs. mini-set subject ranking) vs
     budget — global vs mean-per-environment.
  3. P(environment gets zero sampled items) vs budget, one line per
     environment — the direct answer to "does unstratified random sampling
     starve small environments".

Usage:
    python plot_subsampling_results.py
    python plot_subsampling_results.py --results=data/subsampling_sweep_random.csv
"""

import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from plot_config import ENVIRONMENT_COLOURS  # noqa: E402

GLOBAL_COLOUR = "#7150e0"
PER_ENV_COLOUR = "#e87584"

# same fallback convention as plot_sorted_response_matrix.py, so "wetlab"
# (missing from ENVIRONMENT_COLOURS) gets a consistent colour across figures
ENV_FALLBACK = {"wetlab": "#e377c2"}


def env_colour(env: str) -> str:
    return ENVIRONMENT_COLOURS.get(env, ENV_FALLBACK.get(env, "#999999"))


def _band(df: pd.DataFrame, col: str, group: str = "budget") -> pd.DataFrame:
    g = df.groupby(group)[col]
    return pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75)})


def plot_results(results: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- panel 1: MSE vs budget ---
    ax = axes[0]
    for col, colour, label in [
        ("global_mse", GLOBAL_COLOUR, "global (item-pooled)"),
        ("mean_per_env_mse", PER_ENV_COLOUR, "mean per-environment"),
    ]:
        b = _band(results, col)
        ax.plot(b.index, b["median"], color=colour, linewidth=2, label=label)
        ax.fill_between(b.index, b["p25"], b["p75"], color=colour, alpha=0.2, linewidth=0)
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("MSE (mini-set vs full-set score)")
    ax.set_title("Score preservation")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # --- panel 2: ranking preservation vs budget ---
    ax = axes[1]
    for col, colour, label in [
        ("global_spearman", GLOBAL_COLOUR, "global (item-pooled)"),
        ("mean_per_env_spearman", PER_ENV_COLOUR, "mean per-environment"),
    ]:
        b = _band(results, col)
        ax.plot(b.index, b["median"], color=colour, linewidth=2, label=label)
        ax.fill_between(b.index, b["p25"], b["p75"], color=colour, alpha=0.2, linewidth=0)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("Spearman ρ (true vs. mini-set subject ranking)")
    ax.set_title("Ranking preservation")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # --- panel 3: environment starvation risk ---
    ax = axes[2]
    cov_cols = [c for c in results.columns if c.startswith("coverage__")]
    for col in cov_cols:
        env = col.removeprefix("coverage__")
        starved = results.groupby("budget")[col].apply(lambda s: (s == 0).mean())
        ax.plot(starved.index, starved.values, color=env_colour(env), linewidth=2, label=env)
    ax.set_xlabel("budget (# items)")
    ax.set_ylabel("P(environment gets 0 sampled items)")
    ax.set_title("Environment starvation risk")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    n_repeats = results.groupby("budget").size().iloc[0]
    fig.suptitle(
        f"{results['sampler'].iloc[0]} sampling — median ± IQR over {n_repeats} repeats per budget",
        fontsize=10,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(results: str | None = None, output: str | None = None) -> None:
    path = Path(results) if results else DATA_DIR / "subsampling_sweep_random_excl-resistor.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    tag = path.stem.removeprefix("subsampling_sweep_")
    out = Path(output) if output else OUT_DIR / f"subsampling_{tag}.pdf"
    plot_results(df, out)


if __name__ == "__main__":
    fire.Fire(main)
