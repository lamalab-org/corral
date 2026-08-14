"""Item + test information curves for the 2PL IRT fit.

Item information: I_i(theta) = a_i^2 * P_i(theta) * (1 - P_i(theta))
  -- how much a single item tells you about a subject's ability at a given
  theta. Peaks at theta = b_i (the item's difficulty), peak height set by
  a_i^2 (discrimination) -- unlike the ICC, whose useful range spans the
  whole curve, an item's information is concentrated near its own difficulty
  and drops off fast either side, which is exactly why overlaying many of
  them in one plot works: each is a compact, mostly-non-overlapping bump.

Test information: T(theta) = sum_i I_i(theta)
  -- what the whole item set (an environment, here) tells you about ability
  at each theta, and where it doesn't. This is the one plotted per
  environment: all item curves as thin transparent lines (texture, not
  meant to be read individually -- see plot_icc.py for that), the sum as one
  bold line with a posterior credible band, plus a rug of where the 8
  fitted subjects' abilities actually sit.

Credible band for the total is computed from *joint* posterior draws (same
draw index s used for every item's a_i,s/b_i,s when summing), not from
independently-resampled per-item quantiles -- summing independently-quantiled
bands would be both statistically wrong and artificially narrow.

Styling: lama-aesthetics (https://github.com/lamalab-org/lama-aesthetics) -- the
same package plot_sorted_response_matrix.py already opts into. get_style("main")
sets fonts + an 8-color categorical cycle (used here for the 4 environments,
so colors match anything else in the repo drawn with the same style) and
range_frame gives Tufte-style axes (spines trimmed to the data range, offset
from the plot, nice tick placement) instead of matplotlib's default boxed axes.

Usage:
    python plot_information_curve.py                    # one PDF/PNG per environment
    python plot_information_curve.py --environment=retro
    python plot_information_curve.py --combined=True     # all 4 overlaid in one plot
"""

import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent))
from plot_icc import (
    DEFAULT_EXCLUDE_MODELS,
    DEFAULT_SOURCE,
    DEFAULT_TRACE,
    load_fit,
)

try:
    import lama_aesthetics

    lama_aesthetics.get_style("main")
    HAVE_LAMA_AESTHETICS = True
except ImportError:
    HAVE_LAMA_AESTHETICS = False

OUT_DIR = Path(__file__).parent.parent / "figures" / "information"


def item_information(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (a**2) * p * (1 - p)


def total_information_band(
    a_all: np.ndarray,
    b_all: np.ndarray,
    theta_grid: np.ndarray,
    ci: float = 0.89,
    n_sub: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """a_all/b_all: (n_item, n_posterior_sample). Sums info across items
    within each posterior draw (joint), then summarizes across draws.

    n_sub=None (default) uses every available posterior draw -- no
    subsampling. An earlier version subsampled to 500 draws "for speed";
    checked empirically and that introduced ~15% seed-dependent jitter in
    the reported band width (whichever 500 of 8000 happened to get picked),
    while using all 8000 costs <2s even for the largest environment (39
    items) and is deterministic. Not worth the noise -- only subsample if
    you have a real speed reason to (n_sub still works if you pass it)."""
    n_item, n_sample = a_all.shape
    n_sub = n_sample if n_sub is None else min(n_sub, n_sample)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_sample, size=n_sub, replace=False)
    a_s, b_s = a_all[:, idx], b_all[:, idx]  # (n_item, n_sub)

    p = expit(
        a_s[:, :, None] * (theta_grid[None, None, :] - b_s[:, :, None])
    )  # (n_item, n_sub, n_grid)
    info = item_information(a_s[:, :, None], p)
    total = info.sum(axis=0)  # (n_sub, n_grid)

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return (
        total.mean(axis=0),
        np.quantile(total, lo_q, axis=0),
        np.quantile(total, hi_q, axis=0),
    )


def plot_environment_information(
    env: str,
    rows: pd.DataFrame,
    theta_mean_subjects: np.ndarray,
    theta_grid: np.ndarray,
    a_all_env: np.ndarray,
    b_all_env: np.ndarray,
    a_point: np.ndarray,
    b_point: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    # individual items: thin, transparent, point-estimate a/b -- texture only
    for a_i, b_i in zip(a_point, b_point, strict=False):
        p = expit(a_i * (theta_grid - b_i))
        ax.plot(theta_grid, item_information(a_i, p), color="C0", lw=0.6, alpha=0.25)

    # total (test) information: joint-posterior mean + credible band
    mean_total, lo, hi = total_information_band(a_all_env, b_all_env, theta_grid)
    ax.fill_between(
        theta_grid, lo, hi, color="C1", alpha=0.2, lw=0, label="89% ETI (total)"
    )
    ax.plot(
        theta_grid, mean_total, color="C1", lw=2.4, label="total (test) information"
    )

    peak_theta = theta_grid[np.argmax(mean_total)]
    ax.axvline(peak_theta, color="C1", ls=":", lw=1, alpha=0.7)
    ax.annotate(
        f"peak at θ~{peak_theta:.2f}",
        xy=(peak_theta, mean_total.max()),
        xytext=(peak_theta + 0.3, mean_total.max() * 1.03),
        fontsize=8,
        color="C1",
    )

    # rug of observed subject abilities
    ax.plot(
        theta_mean_subjects,
        np.full_like(theta_mean_subjects, -0.02 * mean_total.max()),
        "|",
        color="black",
        markersize=10,
        clip_on=False,
        label="subject θ (observed)",
    )

    ax.set_xlim(theta_grid.min(), theta_grid.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("theta (ability)")
    ax.set_ylabel("information")
    ax.set_title(
        f"{env} — item + test information ({len(rows)} items)\n"
        f"thin lines = individual items (point estimate); bold = sum across items (joint posterior)"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.success(f"Saved -> {out_path} + {out_path.with_suffix('.png')}")


def plot_all_environments_information(
    item_rows: pd.DataFrame,
    theta_mean_subjects: np.ndarray,
    theta_grid: np.ndarray,
    a_all: np.ndarray,
    b_all: np.ndarray,
    out_path: Path,
) -> None:
    """All 4 environments' TOTAL information curves overlaid in one axes --
    no individual-item lines this time (that clutter is exactly what the
    per-environment plots already show); this one is for comparing shape
    and location across environments at a glance."""
    envs = sorted(item_rows["environment"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for env, color in zip(envs, colors, strict=False):
        mask = (item_rows["environment"] == env).to_numpy()
        mean_total, lo, hi = total_information_band(
            a_all[mask], b_all[mask], theta_grid
        )
        ax.fill_between(theta_grid, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(
            theta_grid,
            mean_total,
            color=color,
            lw=2,
            label=f"{env} ({mask.sum()} items)",
        )

    ax.plot(
        theta_mean_subjects,
        np.full_like(theta_mean_subjects, -0.3),
        "|",
        color="0.3",
        markersize=10,
        clip_on=False,
        label="subject θ (observed)",
    )

    ax.set_xlabel("theta (ability)")
    ax.set_ylabel("test information")
    ax.set_title("Test information by environment (89% ETI bands)")
    ax.legend(fontsize=8, loc="upper left")

    if HAVE_LAMA_AESTHETICS:
        all_y = np.concatenate(
            [ax.lines[i].get_ydata() for i in range(len(envs))] + [np.array([0.0])]
        )
        lama_aesthetics.range_frame(ax, theta_grid, all_y, pad=0.04)
    else:
        ax.set_xlim(theta_grid.min(), theta_grid.max())
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.success(f"Saved -> {out_path} + {out_path.with_suffix('.png')}")


def main(
    environment: str | None = None,
    combined: bool = False,
    trace_path: str = str(DEFAULT_TRACE),
    data_path: str = str(DEFAULT_SOURCE),
    exclude_environments: str = "",
    exclude_models: str = ",".join(DEFAULT_EXCLUDE_MODELS),
    theta_min: float = -4.0,
    theta_max: float = 4.0,
    output: str | None = None,
) -> None:
    """
    Args:
        combined: overlay all 4 environments' total-information curves in one
            plot instead of one PDF/PNG per environment. Takes precedence
            over `environment`.
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
    a_all = (
        trace.posterior["a"].stack(sample=("chain", "draw")).to_numpy()
    )  # (n_item, n_sample)
    b_all = trace.posterior["b"].stack(sample=("chain", "draw")).to_numpy()

    item_rows = pd.DataFrame(
        {
            "item": item_list,
            "environment": item_env.reindex(item_list).to_numpy(),
            "a_mean": a_mean,
            "b_mean": b_mean,
        }
    )
    theta_grid = np.linspace(theta_min, theta_max, 300)

    if combined:
        out = Path(output) if output else OUT_DIR / "information_all_environments.pdf"
        plot_all_environments_information(
            item_rows, theta_mean, theta_grid, a_all, b_all, out
        )
        return

    envs = [environment] if environment else sorted(item_rows["environment"].unique())
    for env in envs:
        mask = (item_rows["environment"] == env).to_numpy()
        if not mask.any():
            logger.warning(f"No items for environment={env}, skipping")
            continue
        out = (
            Path(output)
            if (output and len(envs) == 1)
            else OUT_DIR / f"information_{env}.pdf"
        )
        plot_environment_information(
            env,
            item_rows[mask],
            theta_mean,
            theta_grid,
            a_all[mask],
            b_all[mask],
            a_mean[mask],
            b_mean[mask],
            out,
        )


if __name__ == "__main__":
    fire.Fire(main)
