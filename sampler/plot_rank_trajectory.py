"""Plot per-new-model rank trajectories from run_rank_trajectory.py.

Small multiples, one panel per method (random, stratified, stratified+IRT),
x=budget, y=rank (1=best) among the 6 new (model, scaffold) subjects, one
line per subject. A faint dotted horizontal line in the same colour marks
that subject's TRUE rank — the vertical gap between a solid line and its own
dotted reference is exactly "how wrong is this specific model's rank right
now", which a single aggregate Spearman rho can't show you.

Usage:
    python plot_rank_trajectory.py
"""

import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from plot_mini_item_response_matrix import MODEL_COLOURS, MODEL_NAMES  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"

METHOD_ORDER = ["random", "stratified_proportional", "stratified_irt"]
METHOD_TITLES = {"random": "random", "stratified_proportional": "stratified", "stratified_irt": "stratified + IRT"}
LINESTYLES = {"react": "-", "tool_calling": "--"}


def plot_rank_trajectories(df: pd.DataFrame, output_path: Path) -> None:
    subjects = sorted(df["subject"].unique())
    n_subjects = len(subjects)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, method in zip(axes, METHOD_ORDER):
        sub = df[df.method == method]
        for subject in subjects:
            model, scaffold = subject.split("__")
            colour = MODEL_COLOURS.get(model, "#999999")
            ls = LINESTYLES.get(scaffold, "-")
            g = sub[sub.subject == subject].sort_values("budget")
            ax.axhline(g["true_rank"].iloc[0], color=colour, linewidth=1, linestyle=":", alpha=0.5)
            ax.plot(g.budget, g["rank"], color=colour, linestyle=ls, linewidth=2, marker="o", markersize=3)
        ax.set_title(METHOD_TITLES.get(method, method), fontsize=11)
        ax.set_xlabel("budget (# items)")
        ax.set_yticks(range(1, n_subjects + 1))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("rank among the 6 new (model, scaffold) subjects\n(1 = best; dotted = true rank)")

    handles = [
        plt.Line2D([0], [0], color=MODEL_COLOURS.get(m, "#999"), linewidth=2, label=MODEL_NAMES.get(m, m))
        for m in sorted({s.split("__")[0] for s in subjects})
    ]
    handles += [
        plt.Line2D([0], [0], color="black", linestyle=ls, linewidth=2, label=scaffold)
        for scaffold, ls in LINESTYLES.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.08), fontsize=8, frameon=False)

    fig.suptitle(
        "Rank trajectory of new models on an item set fit ONLY on the 3 legacy models",
        fontsize=12,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    logger.success(f"Saved: {output_path}")
    plt.close(fig)


def main(results: str | None = None, output: str | None = None) -> None:
    path = Path(results) if results else DATA_DIR / "rank_trajectory_results.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    out = Path(output) if output else OUT_DIR / "rank_trajectory.pdf"
    plot_rank_trajectories(df, out)


if __name__ == "__main__":
    fire.Fire(main)
