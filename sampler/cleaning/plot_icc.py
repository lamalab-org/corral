"""Item Characteristic Curves for the 2PL IRT fit.

P(success | subject ability theta) = sigmoid(a_i * (theta - b_i))

Each item gets a curve (posterior-mean a/b), a shaded 89% ETI credible band
(from posterior draws of a/b, same convention as report_irt_top4_cleaned.html
-- see that report for what ETI means), and the actual observed data overlaid:
one dot per subject at (that subject's posterior-mean theta, that subject's
empirical success rate on this item).

Caveat worth looking at before reading too much into any one curve: the 8
fitted subjects only span theta in about [-1.7, 1.3], but item difficulty (b)
spans about [-3.6, 3.7] -- items with |b| beyond the subject range have their
curve's transition region extrapolated past where any real subject sits, and
that's exactly where the credible band widens the most. The band is the
honest signal for "trust this part of the curve less."

Default output: one paginated PDF per environment (items sorted easiest ->
hardest, small-multiples grid), plus a PNG of that PDF's first page as a
quick-look preview -- a multi-page PDF doesn't have one PNG equivalent, so
this is a deliberate loosening of the rest of the repo's "same figure, two
formats" convention, not an oversight.

Usage:
    python plot_icc.py                                  # all 4 environments, paginated
    python plot_icc.py --environment=retro               # just retro
    python plot_icc.py --items=retro|level_1|make_1_lvl1  # one big single-panel plot
"""

import sys
from pathlib import Path

import arviz as az
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent.parent))
from fit_irt_2pl import build_trial_counts

try:
    import lama_aesthetics

    lama_aesthetics.get_style("main")
except ImportError:
    pass

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "figures" / "icc"

DEFAULT_TRACE = DATA_DIR / "irt_2pl_bayesian_trace_top4_cleaned.nc"
DEFAULT_SOURCE = DATA_DIR / "corral_mini_source_cleaned.csv"
DEFAULT_EXCLUDE_MODELS = ["gpt-4o", "gpt-oss-120b"]  # matches the top4_cleaned fit


def load_fit(
    trace_path: Path,
    data_path: Path,
    exclude_environments: list[str] | None,
    exclude_models: list[str] | None,
):
    """Re-derive the exact (subjects, items) index order the trace was fit on.

    The trace itself only stores integer dims (theta_dim_0, a_dim_0, ...) --
    build_trial_counts with identical args reproduces the same sorted
    subjects/items lists deterministically, which is how the .nc's positional
    indices get their labels back.
    """
    trace = az.from_netcdf(trace_path)
    subjects, items, K, N, item_env = build_trial_counts(
        exclude_environments=exclude_environments,
        exclude_models=exclude_models,
        data_path=data_path,
    )
    n_theta = trace.posterior.sizes["theta_dim_0"]
    n_item = trace.posterior.sizes["a_dim_0"]
    if len(subjects) != n_theta or len(items) != n_item:
        raise ValueError(
            f"Re-derived pool ({len(subjects)} subjects, {len(items)} items) doesn't "
            f"match the trace ({n_theta} subjects, {n_item} items) -- wrong "
            f"exclude_environments/exclude_models/data_path for this trace file?"
        )
    return trace, subjects, items, K, N, item_env


def icc_band(
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    theta_grid: np.ndarray,
    ci: float = 0.89,
    n_sub: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean ICC + equal-tailed credible band over theta_grid.

    a_draws/b_draws: (n_posterior_samples,) flattened (chain, draw) draws for
    one item. n_sub=None (default) uses every available draw -- see
    total_information_band's docstring in plot_information_curve.py for why
    subsampling was dropped (seed-dependent jitter for no real speed gain at
    this problem size).
    """
    rng = np.random.default_rng(seed)
    n_sub = len(a_draws) if n_sub is None else min(n_sub, len(a_draws))
    idx = rng.choice(len(a_draws), size=n_sub, replace=False)
    a_s, b_s = a_draws[idx][:, None], b_draws[idx][:, None]
    p = expit(a_s * (theta_grid[None, :] - b_s))  # (n_sub, n_grid)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return p.mean(axis=0), np.quantile(p, lo_q, axis=0), np.quantile(p, hi_q, axis=0)


def plot_one(
    ax: plt.Axes,
    label: str,
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    theta_subjects: np.ndarray,
    obs_rates: np.ndarray,
    theta_grid: np.ndarray,
    fontsize: int = 7,
) -> None:
    mean_curve, lo, hi = icc_band(a_draws, b_draws, theta_grid)
    b_mean = float(np.mean(b_draws))
    a_mean = float(np.mean(a_draws))

    ax.fill_between(theta_grid, lo, hi, color="C0", alpha=0.18, lw=0)
    ax.plot(theta_grid, mean_curve, color="C0", lw=1.4)
    ax.axvline(b_mean, color="0.6", ls="--", lw=0.7)
    ax.scatter(theta_subjects, obs_rates, color="black", s=10, zorder=5, alpha=0.85)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(theta_grid.min(), theta_grid.max())
    ax.set_title(f"{label}\na={a_mean:.2f} b={b_mean:.2f}", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)


def plot_environment_grid(
    env: str,
    item_rows: pd.DataFrame,
    trace,
    items: list[str],
    K: np.ndarray,
    N: np.ndarray,
    theta_mean: np.ndarray,
    theta_grid: np.ndarray,
    out_pdf: Path,
    ncols: int = 5,
    nrows: int = 4,
) -> None:
    a_all = (
        trace.posterior["a"].stack(sample=("chain", "draw")).to_numpy()
    )  # (n_item, n_sample)
    b_all = trace.posterior["b"].stack(sample=("chain", "draw")).to_numpy()
    item_idx = {it: i for i, it in enumerate(items)}

    rows = item_rows.sort_values("b_mean").reset_index(drop=True)
    per_page = ncols * nrows
    n_pages = int(np.ceil(len(rows) / per_page))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    first_page_fig = None
    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            chunk = rows.iloc[page * per_page : (page + 1) * per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.3 * nrows))
            axes = np.atleast_1d(axes).flatten()
            for ax, (_, row) in zip(axes, chunk.iterrows(), strict=False):
                i = item_idx[row["item"]]
                obs_rates = K[:, i] / N[:, i]
                plot_one(
                    ax,
                    row["item"].split("|", 1)[1],  # drop redundant "env|" prefix
                    a_all[i],
                    b_all[i],
                    theta_mean,
                    obs_rates,
                    theta_grid,
                )
            for ax in axes[len(chunk) :]:
                ax.axis("off")
            fig.suptitle(
                f"{env} -- Item Characteristic Curves (page {page + 1}/{n_pages})\n"
                "black dots = observed (subject theta, empirical success rate); "
                "shaded = 89% ETI credible band",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            if page == 0:
                first_page_fig = fig
            else:
                plt.close(fig)

    if first_page_fig is not None:
        first_page_fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
        plt.close(first_page_fig)
    logger.success(
        f"Saved -> {out_pdf} ({n_pages} pages) + {out_pdf.with_suffix('.png')} (page 1)"
    )


def plot_single_items(
    item_names: list[str],
    item_rows: pd.DataFrame,
    trace,
    items: list[str],
    K: np.ndarray,
    N: np.ndarray,
    theta_mean: np.ndarray,
    theta_grid: np.ndarray,
    out_path: Path,
) -> None:
    a_all = trace.posterior["a"].stack(sample=("chain", "draw")).to_numpy()
    b_all = trace.posterior["b"].stack(sample=("chain", "draw")).to_numpy()
    item_idx = {it: i for i, it in enumerate(items)}

    missing = [it for it in item_names if it not in item_idx]
    if missing:
        raise ValueError(f"Not in this fit's item pool: {missing}")

    n = len(item_names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), squeeze=False
    )
    axes = axes.flatten()
    for ax, item in zip(axes, item_names, strict=False):
        i = item_idx[item]
        obs_rates = K[:, i] / N[:, i]
        plot_one(
            ax, item, a_all[i], b_all[i], theta_mean, obs_rates, theta_grid, fontsize=10
        )
        ax.set_xlabel("theta (ability)", fontsize=9)
        ax.set_ylabel("P(success)", fontsize=9)
    for ax in axes[len(item_names) :]:
        ax.axis("off")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.success(f"Saved -> {out_path} + {out_path.with_suffix('.png')}")


def main(
    environment: str | None = None,
    items: str | None = None,
    trace_path: str = str(DEFAULT_TRACE),
    data_path: str = str(DEFAULT_SOURCE),
    exclude_environments: str = "",
    exclude_models: str = ",".join(DEFAULT_EXCLUDE_MODELS),
    theta_min: float = -4.0,
    theta_max: float = 4.0,
    ncols: int = 5,
    nrows: int = 4,
    output: str | None = None,
) -> None:
    """
    Args:
        environment: plot only this environment's items (paginated grid). If
            neither this nor `items` is set, plots all environments present
            in the pool, one PDF each.
        items: comma-separated item ids ("env|level|task", e.g.
            "retro|level_1|make_1_lvl1") for single, larger detail plots
            instead of the small-multiples grid. Takes precedence over
            `environment`.
        trace_path/data_path/exclude_environments/exclude_models: must match
            whatever fit_irt_2pl_bayesian.py run produced trace_path -- see
            load_fit()'s docstring for why (positional index <-> item-name
            mapping is re-derived, not stored in the .nc file).
        theta_min/theta_max: x-axis range for the curves. Defaults span well
            past the observed subject range ([-1.7, 1.3] for the top4_cleaned
            fit) since several items' difficulty falls outside it.
    """
    trace, subjects, item_list, K, N, item_env = load_fit(
        Path(trace_path),
        Path(data_path),
        [e.strip() for e in exclude_environments.split(",") if e.strip()],
        [m.strip() for m in exclude_models.split(",") if m.strip()],
    )
    theta_mean = trace.posterior["theta"].mean(dim=["chain", "draw"]).to_numpy()
    a_mean = trace.posterior["a"].mean(dim=["chain", "draw"]).to_numpy()
    b_mean = trace.posterior["b"].mean(dim=["chain", "draw"]).to_numpy()
    logger.info(
        f"{len(subjects)} subjects x {len(item_list)} items, theta range "
        f"[{theta_mean.min():.2f}, {theta_mean.max():.2f}]"
    )

    item_rows = pd.DataFrame(
        {
            "item": item_list,
            "environment": item_env.reindex(item_list).to_numpy(),
            "a_mean": a_mean,
            "b_mean": b_mean,
        }
    )
    theta_grid = np.linspace(theta_min, theta_max, 200)

    if items:
        item_names = [i.strip() for i in items.split(",") if i.strip()]
        out = Path(output) if output else OUT_DIR / "icc_items.pdf"
        plot_single_items(
            item_names, item_rows, trace, item_list, K, N, theta_mean, theta_grid, out
        )
        return

    envs = [environment] if environment else sorted(item_rows["environment"].unique())
    for env in envs:
        rows = item_rows[item_rows["environment"] == env]
        if rows.empty:
            logger.warning(f"No items for environment={env}, skipping")
            continue
        out = (
            Path(output) if (output and len(envs) == 1) else OUT_DIR / f"icc_{env}.pdf"
        )
        plot_environment_grid(
            env,
            rows,
            trace,
            item_list,
            K,
            N,
            theta_mean,
            theta_grid,
            out,
            ncols=ncols,
            nrows=nrows,
        )


if __name__ == "__main__":
    fire.Fire(main)
