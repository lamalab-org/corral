"""
Combined panel: Radar chart (left) + Variance decomposition (right).

TWO_COL_WIDTH x ONE_COL_HEIGHT, matching font sizes.
"""

from pathlib import Path

import arviz as az
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
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

RESULTS_DIR = Path(__file__).parent / "results" / "lfm-binomial"
IRT_DIR = RESULTS_DIR / "irt_baseline"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "panel_4"

ENV_ORDER = ["spectra", "wetlab", "resistor", "retro", "afm", "catalyst", "md", "ml"]


def radar_factory(num_vars, frame="polygon"):  # noqa: ARG001
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
            _LABEL_WRAP = {
                "Spectroscopic Structure Elucidation": "Spectroscopic\nStructure Elucidation",
                "Inorganic Qualitative Analysis": "Inorganic Qualitative\nAnalysis",
                "ML-based Property Prediction": "ML-based\nProperty Prediction",
                "Adsorption Surface Construction": "Adsorption\nSurface Construction",
                "AFM Experiment Execution": "AFM Experiment\nExecution",
                "Molecular Simulation": "Molecular\nSimulation",
                "Retrosynthetic Planning": "Retrosynthetic\nPlanning",
                "Circuit Inference": "Circuit\nInference",
            }
            wrapped = [_LABEL_WRAP.get(lb, lb) for lb in labels]
            _lines, texts = self.set_thetagrids(np.degrees(theta), wrapped)
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


def main(best_model="model3_abilities_env"):
    reasoning_df = pd.read_csv(IRT_DIR / "reasoning_theta.csv")
    df_r = reasoning_df.copy()
    df_r["model"] = df_r["model"].map(MODEL_NAMES)
    df_r["environment"] = df_r["environment"].map(ENVIRONMENT_NAMES)
    pivot = df_r.pivot_table(index="environment", columns="model", values="theta_mean")
    env_display_order = [
        ENVIRONMENT_NAMES[e] for e in ENV_ORDER if ENVIRONMENT_NAMES[e] in pivot.index
    ]
    pivot = pivot.reindex(env_display_order)
    environments = pivot.index.tolist()
    theta = radar_factory(len(environments), frame="polygon")

    idata = az.from_netcdf(RESULTS_DIR / f"{best_model}_trace.nc")
    data_df = pd.read_csv(RESULTS_DIR / "prepared_data.csv")
    posterior = idata.posterior

    contributions = {}
    for coef_name, label in [("knowledge", "Knowledge"), ("reasoning", "Reasoning")]:
        total_key = f"{coef_name}_coef_total"
        base_key = f"{coef_name}_coef"
        if total_key in posterior:
            coef_vals = posterior[total_key].mean(dim=["chain", "draw"]).to_numpy()
            contributions[label] = float(
                (coef_vals * data_df[f"{coef_name}_z"].to_numpy()).var()
            )
        elif base_key in posterior:
            coef_val = float(posterior[base_key].mean().values)
            contributions[label] = float(
                (coef_val * data_df[f"{coef_name}_z"].to_numpy()).var()
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
            label = {"env_level_effect": "Env x Scope"}.get(
                effect_name, effect_name.replace("_effect", "").title()
            )
            contributions[label] = effect_var

    total_var = sum(contributions.values())
    pct = {k: 100 * v / total_var for k, v in contributions.items()}
    pct = dict(sorted(pct.items(), key=lambda x: x[1]))

    fig = plt.figure(figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.65)

    ax_radar = fig.add_subplot(gs[0, 0], projection="radar")

    for model_id, display_name in MODEL_NAMES.items():
        if display_name not in pivot.columns:
            continue
        values = pivot[display_name].to_numpy()
        color = MODEL_COLOR_MAP.get(model_id, "#999999")
        ax_radar.fill(theta, values, alpha=0.08, color=color, label=display_name)
        ax_radar.plot(theta, values, color=color, linewidth=1.2)

    ax_radar.set_varlabels(environments)
    ax_radar.tick_params(pad=28)
    ax_radar.set_yticklabels([])
    ax_radar.set_yticks([])

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

    ax_var = fig.add_subplot(gs[0, 1])

    components = list(pct.keys())
    values = list(pct.values())
    ax_var.barh(components, values, color="#7150e0")
    ax_var.set_xlabel(
        "Variance Explained (%)", fontsize=FONT_SIZES["axis_label"], fontweight="bold"
    )
    ax_var.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    for i, val in enumerate(values):
        ax_var.text(
            val + 0.8, i, f"{val:.1f}%", va="center", fontsize=FONT_SIZES["tick_label"]
        )

    range_frame(ax_var, np.array(values), np.arange(len(components)), pad=0.1)

    ax_var.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=len(MODEL_NAMES),
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
    )

    output_path = OUT_DIR / "radar_and_variance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.success(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
