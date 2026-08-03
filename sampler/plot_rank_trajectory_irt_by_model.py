"""Standalone stratified_irt rank trajectory, one panel per new model.

Same underlying data as plot_rank_trajectory.py (rank_trajectory_results.csv)
but filtered to stratified_irt only and faceted by MODEL instead of by
sampling method — each panel shows one new model's two scaffold variants
(react solid, tool_calling dashed) so you can see whether the two agent
scaffolds for the same model behave differently under this design.

Usage:
    python plot_rank_trajectory_irt_by_model.py
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
LINESTYLES = {"react": "-", "tool_calling": "--"}


def plot_by_model(df: pd.DataFrame, output_path: Path) -> None:
    df = df[df.method == "stratified_irt"].copy()
    df["model"] = df["subject"].str.split("__").str[0]
    models = sorted(df["model"].unique(), key=lambda m: df[df.model == m]["true_rank"].min())
    n_subjects_total = df["subject"].nunique()

    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df.model == model]
        colour = MODEL_COLOURS.get(model, "#999999")
        for scaffold, ls in LINESTYLES.items():
            g = sub[sub.subject == f"{model}__{scaffold}"].sort_values("budget")
            if g.empty:
                continue
            ax.axhline(g["true_rank"].iloc[0], color=colour, linewidth=1, linestyle=":", alpha=0.5)
            ax.plot(g.budget, g["rank"], color=colour, linestyle=ls, linewidth=2, marker="o", markersize=3, label=scaffold)
        ax.set_title(MODEL_NAMES.get(model, model), fontsize=11, color=colour)
        ax.set_xlabel("budget (# items)")
        ax.set_yticks(range(1, n_subjects_total + 1))
        ax.invert_yaxis()
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("rank among the 6 new (model, scaffold) subjects\n(1 = best; dotted = true rank)")

    fig.suptitle(
        "stratified_irt: rank trajectory per new model — item set fit ONLY on the 3 legacy models",
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

    out = Path(output) if output else OUT_DIR / "rank_trajectory_irt_by_model.pdf"
    plot_by_model(df, out)


if __name__ == "__main__":
    fire.Fire(main)
