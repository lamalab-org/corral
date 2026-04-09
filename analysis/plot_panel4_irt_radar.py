"""
Spider chart of IRT reasoning capability (theta_R) per model across environments.

Each model is a polygon; each spoke is an environment.
Uses a custom RadarAxes projection with polygon frame.
"""

from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_WIDTH
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from plot_config import ENVIRONMENT_NAMES, FONT_SIZES, MODEL_COLOURS, MODEL_NAMES

MODEL_COLOR_MAP = MODEL_COLOURS

lama_aesthetics.get_style("main")

RESULTS_DIR = Path(__file__).parent / "results" / "lfm-binomial" / "irt_baseline"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_4"


def radar_factory(num_vars, frame="polygon"):  # noqa: ARG001
    """Create a RadarAxes projection with polygon grid lines."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            def _split_label(lb):
                words = lb.split()
                if len(words) <= 2:
                    return "\n".join(words) if len(words) == 2 else lb
                mid = len(words) // 2
                return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

            labels_wrapped = [_split_label(lb) for lb in labels]
            _lines, texts = self.set_thetagrids(np.degrees(theta), labels_wrapped)
            for t in texts:
                t.set_fontsize(FONT_SIZES["legend"])
            half = (len(texts) - 1) // 2
            for t in texts[1:half]:
                t.set_horizontalalignment("left")
            for t in texts[-half + 1 :]:
                t.set_horizontalalignment("right")

        def _gen_axes_patch(self):
            return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")

        def _gen_axes_spines(self):
            spine = Spine(
                axes=self,
                spine_type="circle",
                path=MplPath.unit_regular_polygon(num_vars),
            )
            spine.set_transform(
                Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
            )
            return {"polar": spine}

    register_projection(RadarAxes)
    return theta


def plot_capability_radar(output_path: Path):
    """Generate a spider chart for reasoning capability (theta_R)."""
    reasoning_df = pd.read_csv(RESULTS_DIR / "reasoning_theta.csv")

    plot_df = reasoning_df.copy()
    plot_df["model"] = plot_df["model"].map(MODEL_NAMES)
    plot_df["environment"] = plot_df["environment"].map(ENVIRONMENT_NAMES)

    pivot = plot_df.pivot_table(
        index="environment", columns="model", values="theta_mean"
    )

    env_order = [
        "spectra",
        "wetlab",
        "resistor",
        "retro",
        "afm",
        "catalyst",
        "md",
        "ml",
    ]
    env_display_order = [
        ENVIRONMENT_NAMES[e] for e in env_order if ENVIRONMENT_NAMES[e] in pivot.index
    ]
    pivot = pivot.reindex(env_display_order)

    environments = pivot.index.tolist()
    theta = radar_factory(len(environments), frame="polygon")

    fig, ax = plt.subplots(
        figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH),
        subplot_kw={"projection": "radar"},
    )

    for model_id, display_name in MODEL_NAMES.items():
        if display_name not in pivot.columns:
            continue
        values = pivot[display_name].to_numpy()
        color = MODEL_COLOR_MAP.get(model_id, "#999999")
        ax.fill(theta, values, alpha=0.08, color=color, label=display_name)
        ax.plot(theta, values, color=color, linewidth=1.2)

    ax.set_varlabels(environments)
    ax.tick_params(pad=25)

    ax.set_yticklabels([])
    ax.set_yticks([])

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR_MAP.get(mid, "#999999"),
            linewidth=2,
            label=dname,
        )
        for mid, dname in MODEL_NAMES.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(MODEL_NAMES),
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        columnspacing=1.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved -> {output_path}")


def main():
    plot_capability_radar(OUT_DIR / "capability_radar.png")


if __name__ == "__main__":
    main()
