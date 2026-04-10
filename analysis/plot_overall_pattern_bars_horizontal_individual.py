"""
Plot every individual anti-pattern and productive-pattern prevalence as its
own bar, grouped by family, with a coupled heatmap showing prevalence across
reasoning-type groups.

Top panel:   heatmap with one row per environment group and one column per pattern.
Bottom panel: vertical bar chart (one bar per pattern), averaged over all
             models, domains, and levels.

X-axis labels show the group names (Hypothesis Handling, Evidence Handling,
Inquiry Control), centered under their respective clusters.

Variant of plot_overall_pattern_bars_horizontal.py but without group-level
averaging: every motif / breakdown gets its own bar and heatmap column.

Usage:
  python analysis/plot_overall_pattern_bars_horizontal_individual.py
  python analysis/plot_overall_pattern_bars_horizontal_individual.py --summary-path /path/to/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import lama_aesthetics
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from lama_aesthetics import TWO_COL_HEIGHT, TWO_COL_WIDTH
from loguru import logger
from matplotlib.patches import FancyBboxPatch, Patch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

lama_aesthetics.get_style("main")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT / "reasoning_reports" / "analysis" / "annotation_summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "results" / "figures" / "fig_epistemology"

# Pattern keys whose data lives under "subgraph_presence_global" in the JSON.
_SUBGRAPH_SET: set[str] = {
    "refutation_driven_belief_revision",
    "hypothesis_reranking",
    "evidence_led_hypothesis_generation",
    "explore_then_test_transition",
    "convergent_multi_test_evidence",
    "precommitted_test_plan",
    "evidence_guided_test_redesign",
    "fixed_hypothesis_test_tuning",
}

# Three conceptual groups, each containing productive motifs and breakdowns.
GROUPS: dict[str, dict[str, list[str]]] = {
    "hypothesis_handling": {
        "productive": [
            "evidence_led_hypothesis_generation",
            "hypothesis_reranking",
            "refutation_driven_belief_revision",
            "explore_then_test_transition",
        ],
        "breakdowns": [
            "untested_claim",
            "one_sided_confirmation",
            "contradiction_without_repair",
            "premature_commitment",
        ],
    },
    "evidence_handling": {
        "productive": [
            "convergent_multi_test_evidence",
        ],
        "breakdowns": [
            "evidence_non_uptake",
            "disconnected_evidence",
            "unsupported_judgment",
            "uninformative_test",
        ],
    },
    "inquiry_control": {
        "productive": [
            "precommitted_test_plan",
            "evidence_guided_test_redesign",
        ],
        "breakdowns": [
            "fixed_belief_trace",
            "fixed_hypothesis_test_tuning",
            "stalled_revision",
        ],
    },
}

GROUP_ORDER: list[str] = list(GROUPS.keys())

GOOD_COLOR = "#3C77B1"
BAD_COLOR = "#C62828"

GROUP_DISPLAY: dict[str, str] = {
    "hypothesis_handling": "Hypothesis handling",
    "evidence_handling": "Evidence handling",
    "inquiry_control": "Inquiry control",
}

PATTERN_SHORT: dict[str, str] = {
    # Antipatterns
    "untested_claim": "Untested claim",
    "contradiction_without_repair": "Contradiction without repair",
    "one_sided_confirmation": "One-sided confirmation",
    "evidence_non_uptake": "Evidence non-uptake",
    "disconnected_evidence": "Disconnected evidence",
    "unsupported_judgment": "Unsupported judgment",
    "uninformative_test": "Uninformative test",
    "stalled_revision": "Stalled revision",
    "fixed_belief_trace": "Fixed belief trace",
    "premature_commitment": "Premature commitment",
    # Productive subgraphs
    "refutation_driven_belief_revision": "Refutation-driven belief revision",
    "hypothesis_reranking": "Hypothesis reranking",
    "evidence_led_hypothesis_generation": "Evidence-led\nhypothesis generation",
    "convergent_multi_test_evidence": "Convergent multi-test\nevidence",
    "explore_then_test_transition": "Explore-then-test transition",
    "fixed_hypothesis_test_tuning": "Fixed hypothesis test tuning",
    "precommitted_test_plan": "Precommitted test plan",
    "evidence_guided_test_redesign": "Evidence-guided test redesign",
}

ENV_GROUPS: dict[str, list[str]] = {
    "Workflow": ["ml", "afm", "catalyst", "md"],
    "Strategic": ["retrosynthesis"],
    "Hypothesis-driven": ["spectra", "wetlab", "resistor"],
}

ENV_FULL_NAME: dict[str, str] = {
    "afm": "AFM experimental execution",
    "catalyst": "Adsorption surface construction",
    "md": "Molecular simulation",
    "ml": "ML-based property",
    "resistor": "Circuit inference",
    "retrosynthesis": "Retrosynthetic planning",
    "spectra": "Spectroscopic structure elucidation",
    "wetlab": "Inorganic qualitative analysis",
}

ENV_SHORT_NAME: dict[str, str] = {
    "afm": "AFM\nexperimental\nexecution",
    "catalyst": "Adsorption\nsurface\nconstruction",
    "md": "Molecular\nsimulation",
    "ml": "ML-based\nproperty",
    "resistor": "Circuit\ninference",
    "retrosynthesis": "Retrosynthetic\nplanning",
    "spectra": "Spectroscopic\nstructure\nelucidation",
    "wetlab": "Inorganic\nqualitative\nanalysis",
}

ENV_GROUP_DISPLAY: dict[str, str] = {
    "Workflow": "Workflow execution",
    "Strategic": "Strategic reasoning",
    "Hypothesis-driven": "Hypothesis-driven enquiry",
}

ENV_GROUP_COLORS: dict[str, str] = {
    "Workflow": "#4C72B0",
    "Strategic": "#DD8452",
    "Hypothesis-driven": "#55A868",
}

MODEL_DISPLAY: dict[str, str] = {
    "claude_sonnet_45": "Claude-4.5-Sonnet",
    "gpt_4o": "GPT-4o",
}


def _pretty(key: str) -> str:
    """Turn a snake_case pattern key into a readable label."""
    return PATTERN_SHORT.get(key, key.replace("_", " ").title())


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _frac(data: dict, section: str, field: str) -> float:
    v = data.get(section, {}).get(field, {}).get("fraction")
    if v is None:
        return 0.0
    return float(v)


def _data_section(pat: str) -> str:
    """Return the summary JSON section key for a pattern."""
    return (
        "subgraph_presence_global"
        if pat in _SUBGRAPH_SET
        else "antipattern_presence_global"
    )


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, format=path.suffix.lstrip("."), bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)
    logger.info(f"  saved: {path}")


def _group_frac(by_env: dict, envs: list[str], section: str, field: str) -> float:
    """Return the n_traces-weighted average fraction for field across the given environments."""
    total_traces = 0
    weighted_sum = 0.0
    for env in envs:
        env_data = by_env.get(env)
        if env_data is None:
            continue
        n = env_data.get("n_traces", 0)
        frac = _frac(env_data, section, field)
        weighted_sum += frac * n
        total_traces += n
    if total_traces == 0:
        return 0.0
    return weighted_sum / total_traces


def _build_individual_bars(overall: dict) -> tuple[list[dict], list[float]]:
    """Return one bar dict per individual pattern, ordered by group then kind.

    Within each group the productive patterns come first (blue), then the
    breakdowns (red).  Returns the list of bar dicts and a parallel list of
    group-centre x positions for labelling.
    """
    bars: list[dict] = []
    group_centres: list[float] = []

    bar_width = 0.25
    intra_gap = 0.08  # gap between bars within the same kind
    kind_gap = 0.0  # no extra gap between productive and breakdown clusters
    group_gap = 0.35  # gap between different groups

    x = 0.0
    for _gi, group_key in enumerate(GROUP_ORDER):
        group = GROUPS[group_key]
        group_start = x

        for ki, (kind, color) in enumerate(
            [("productive", GOOD_COLOR), ("breakdowns", BAD_COLOR)]
        ):
            pats = group[kind]
            for _pi, pat in enumerate(pats):
                val = _frac(overall, _data_section(pat), pat)
                bars.append(
                    {
                        "group": group_key,
                        "kind": kind,
                        "pattern": pat,
                        "value": val,
                        "color": color,
                        "x": x,
                    }
                )
                x += bar_width + intra_gap
            # after finishing a kind block, add the kind gap (unless last kind)
            if ki == 0:
                x += kind_gap

        group_end = x - bar_width - intra_gap  # last bar centre
        group_centres.append((group_start + group_end) / 2)
        # add group gap before next group
        x += group_gap

    return bars, group_centres


def plot(summary: dict, out: Path) -> None:
    """Render and save the individual-pattern bar chart with a coupled heatmap.

    Generates a two-panel figure: an upper heatmap showing per-environment-group
    prevalence and a lower bar chart showing overall prevalence, one bar per
    pattern. The figure is saved as a PDF to out.

    Args:
        summary: Parsed annotation_summary.json as a dict.
        out: Directory where the output PDF will be written.
    """
    overall = summary["groupings"]["overall"]
    by_env = summary["groupings"]["by_env"]
    bars, _group_centres = _build_individual_bars(overall)

    x_arr = np.array([b["x"] for b in bars])
    n_bars = len(bars)
    bar_width = 0.25

    env_group_names = list(ENV_GROUPS.keys())
    n_env_groups = len(env_group_names)

    fig, (ax_heat, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 1),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    rounding = 0.15
    values = [b["value"] * 100 for b in bars]
    colors = [b["color"] for b in bars]
    max_val = max(values) if values else 1.0

    x_range = (x_arr[-1] + 0.8) - (x_arr[0] - 0.8)
    bar_ax_height = TWO_COL_WIDTH * (2.5 / 3.5)
    mutation_aspect = (
        (bar_ax_height * x_range) / (TWO_COL_HEIGHT * max_val * 1.15)
        if max_val > 0
        else 1.0
    )

    for xp, val, color in zip(x_arr, values, colors, strict=False):
        if val > 0:
            box = FancyBboxPatch(
                (xp - bar_width / 2, 0),
                bar_width,
                val,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
                mutation_aspect=mutation_aspect,
            )
            ax_bar.add_patch(box)
        ax_bar.text(
            xp,
            val + 0.5,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
        )

    ax_bar.set_xlim(x_arr[0] - 0.3, x_arr[-1] + 0.3)
    ax_bar.set_ylim(0, 100)
    ax_bar.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=100, decimals=0),
    )

    ax_bar.set_xticks(x_arr)
    ax_bar.set_xticklabels(
        [_pretty(b["pattern"]) for b in bars],
        fontsize=10,
    )
    # Color-code x-tick labels to match bar colors
    for tick_label, b in zip(ax_bar.get_xticklabels(), bars, strict=False):
        tick_label.set_color(b["color"])
    ax_bar.set_yticks([0, 50, 100])
    ax_bar.tick_params(axis="y", labelsize=10)
    ax_bar.tick_params(axis="x", labelrotation=60)
    plt.setp(
        ax_bar.get_xticklabels(),
        rotation=60,
        ha="right",
        rotation_mode="anchor",
    )
    # Nudge multiline labels leftward so they don't crowd neighbours
    _shift_labels = {
        "evidence_led_hypothesis_generation",
        "convergent_multi_test_evidence",
    }
    from matplotlib.transforms import ScaledTranslation

    dx_pt = -4  # points
    for tick_label, b in zip(ax_bar.get_xticklabels(), bars, strict=False):
        if b["pattern"] in _shift_labels:
            offset = ScaledTranslation(dx_pt / 72, 0, fig.dpi_scale_trans)
            tick_label.set_transform(tick_label.get_transform() + offset)

    ax_bar.spines["left"].set_position(("outward", 0))
    ax_bar.spines["left"].set_bounds(0, 100)
    ax_bar.spines["bottom"].set_bounds(x_arr[0] - 0.15, x_arr[-1] + 0.15)

    legend_elements = [
        Patch(facecolor=GOOD_COLOR, alpha=0.85, label="Productive motifs"),
        Patch(facecolor=BAD_COLOR, alpha=0.85, label="Reasoning breakdowns"),
    ]
    ax_bar.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(1.0, 1.1),
    )

    heat_matrix = np.full((n_env_groups, n_bars), np.nan)
    for bi, brow in enumerate(bars):
        pat = brow["pattern"]
        for gi, gname in enumerate(env_group_names):
            heat_matrix[gi, bi] = _group_frac(
                by_env,
                ENV_GROUPS[gname],
                _data_section(pat),
                pat,
            )

    cmap = plt.get_cmap("Purples")
    vmin = 0.0
    vmax = float(np.nanmax(heat_matrix)) if np.nanmax(heat_matrix) > 0 else 1.0

    cell_w = bar_width + 0.08  # matches intra_gap so cells are edge-to-edge
    cell_h = 1.0

    for gi in range(n_env_groups):
        for bi in range(n_bars):
            val = heat_matrix[gi, bi]
            if np.isnan(val):
                continue
            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            fc = cmap(norm_val)
            rect = plt.Rectangle(
                (x_arr[bi] - cell_w / 2, gi - cell_h / 2),
                cell_w,
                cell_h,
                facecolor=fc,
                edgecolor="white",
                linewidth=0.5,
            )
            ax_heat.add_patch(rect)
            text_color = "white" if norm_val > 0.55 else "#333333"
            ax_heat.text(
                x_arr[bi],
                gi,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    ax_heat.set_ylim(-0.5, n_env_groups - 0.5)
    ax_heat.set_xlim(x_arr[0] - 0.3, x_arr[-1] + 0.3)
    ax_heat.set_yticks(range(n_env_groups))
    ax_heat.set_yticklabels(env_group_names, fontsize=10, fontweight="bold")
    ax_heat.invert_yaxis()
    ax_heat.set_xticks([])
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(left=False, bottom=False)

    fig.subplots_adjust(bottom=0.38, left=0.10, right=0.97, top=0.98)

    _save(fig, out / "overall_pattern_bars_horizontal_individual.pdf")
    _save(fig, out / "overall_pattern_bars_horizontal_individual.png")
    _save_table(bars, heat_matrix, env_group_names, out)
    _save_table_per_model(summary, out)
    plot_env_level(summary, out)


def _save_table(
    bars: list[dict],
    heat_matrix: np.ndarray,
    env_group_names: list[str],
    out: Path,
) -> None:
    """Write a tabularx table (inner only) with pattern scores."""
    env_cols = " ".join("c" for _ in env_group_names)
    header_cells = " & ".join(env_group_names)
    n_data_cols = len(env_group_names) + 1  # env groups + overall

    lines: list[str] = []
    lines.append(r"\begin{tabularx}{\linewidth}{X " + env_cols + " c}")
    lines.append(r"\toprule")
    lines.append(f"Pattern & {header_cells} & Overall \\\\")

    prev_group = None
    for bi, b in enumerate(bars):
        group_key = b["group"]
        if group_key != prev_group:
            lines.append(r"\midrule")
            heading = GROUP_DISPLAY.get(group_key, group_key)
            lines.append(
                f"\\multicolumn{{{n_data_cols + 1}}}{{l}}{{\\textit{{{heading}}}}} \\\\"
            )
            lines.append(r"\midrule")
            prev_group = group_key

        name = _pretty(b["pattern"])
        env_vals = " & ".join(
            f"{heat_matrix[gi, bi]:.2f}" if not np.isnan(heat_matrix[gi, bi]) else "--"
            for gi in range(len(env_group_names))
        )
        overall = f"{b['value']:.2f}"
        lines.append(f"{name} & {env_vals} & {overall} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")

    table_dir = out.parents[1] / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    tex_path = table_dir / "overall_pattern_bars_horizontal_individual.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  saved: {tex_path}")


def _save_table_per_model(
    summary: dict,
    out: Path,
) -> None:
    """Write a tabularx table with one overall column per model."""
    by_model = summary["groupings"]["by_model"]
    model_keys = sorted(by_model.keys())
    model_labels = [MODEL_DISPLAY.get(m, m) for m in model_keys]
    model_cols = " ".join("c" for _ in model_keys)
    header_cells = " & ".join(model_labels)
    n_data_cols = len(model_keys)

    lines: list[str] = []
    lines.append(r"\begin{tabularx}{\linewidth}{X " + model_cols + "}")
    lines.append(r"\toprule")
    lines.append(f"Pattern & {header_cells} \\\\")

    for group_key in GROUP_ORDER:
        lines.append(r"\midrule")
        heading = GROUP_DISPLAY.get(group_key, group_key)
        lines.append(
            f"\\multicolumn{{{n_data_cols + 1}}}{{l}}{{\\textit{{{heading}}}}} \\\\"
        )
        lines.append(r"\midrule")

        group = GROUPS[group_key]
        for kind in ("productive", "breakdowns"):
            for pat in group[kind]:
                section = _data_section(pat)
                vals = []
                for mk in model_keys:
                    frac = _frac(by_model[mk], section, pat)
                    vals.append(f"{frac:.2f}")
                name = _pretty(pat)
                lines.append(f"{name} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")

    table_dir = out.parents[1] / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    tex_path = table_dir / "overall_pattern_bars_horizontal_individual_per_model.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  saved: {tex_path}")


def _env_level_label(env: str, level: str) -> str:
    """Build a display label like 'Spectroscopic structure elucidation S1'."""
    full = ENV_FULL_NAME.get(env, env.replace("_", " ").title())
    num = level.replace("level_", "")
    return f"{full} S{num}"


def _env_level_frac(
    by_mel: dict, env: str, level: str, section: str, field: str
) -> float:
    """Return n_traces-weighted fraction across models for one env/level."""
    total_traces = 0
    weighted_sum = 0.0
    for key, data in by_mel.items():
        parts = key.split("/")
        if parts[1] == env and parts[2] == level:
            n = data.get("n_traces", 0)
            frac = _frac(data, section, field)
            weighted_sum += frac * n
            total_traces += n
    if total_traces == 0:
        return 0.0
    return weighted_sum / total_traces


def plot_env_level(summary: dict, out: Path) -> None:
    """Horizontal lollipop chart of pattern prevalence per environment and level.

    Each env/level scope (e.g. "Spectroscopic structure elucidation S1") gets a
    row.  Three colour-coded groups are shown: Workflow execution, Strategic
    reasoning, and Hypothesis-driven enquiry.
    """
    by_mel = summary["groupings"]["by_model_env_level"]

    # Collect all env/level combos, ordered by group then env then level
    scopes: list[dict] = [
        {
            "env": env,
            "level": lvl,
            "label": _env_level_label(env, lvl),
            "group": gname,
        }
        for gname, envs in ENV_GROUPS.items()
        for env in envs
        for lvl in sorted({k.split("/")[2] for k in by_mel if k.split("/")[1] == env})
    ]  # {env, level, label, group_name}

    n_scopes = len(scopes)
    if n_scopes == 0:
        return

    # Compute all pattern fractions per scope
    all_pats: list[str] = []
    for gk in GROUP_ORDER:
        for kind in ("productive", "breakdowns"):
            all_pats.extend(GROUPS[gk][kind])

    # Build matrix: rows = scopes, cols = patterns
    mat = np.zeros((n_scopes, len(all_pats)))
    for si, sc in enumerate(scopes):
        for pi, pat in enumerate(all_pats):
            mat[si, pi] = _env_level_frac(
                by_mel, sc["env"], sc["level"], _data_section(pat), pat
            )

    # Average across all patterns for each scope (overall prevalence)
    productive_idx = []
    breakdown_idx = []
    for pi, pat in enumerate(all_pats):
        if pat in _SUBGRAPH_SET or any(
            pat in GROUPS[gk]["productive"] for gk in GROUP_ORDER
        ):
            productive_idx.append(pi)
        else:
            breakdown_idx.append(pi)

    prod_mean = (
        mat[:, productive_idx].mean(axis=1) if productive_idx else np.zeros(n_scopes)
    )
    break_mean = (
        mat[:, breakdown_idx].mean(axis=1) if breakdown_idx else np.zeros(n_scopes)
    )

    # Vertical lollipop chart
    bar_width = 0.28
    gap_width = 0.12
    x_centers = np.arange(n_scopes) * (2 * bar_width + gap_width)

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH, TWO_COL_HEIGHT * 1.0))

    for si, _sc in enumerate(scopes):
        x_prod = x_centers[si] - bar_width / 2
        x_break = x_centers[si] + bar_width / 2

        # Productive (vertical)
        ax.vlines(x_prod, 0, prod_mean[si], color=GOOD_COLOR, alpha=0.75, linewidth=5)
        ax.plot(x_prod, prod_mean[si], "o", markersize=5, color=GOOD_COLOR)

        # Breakdowns (vertical)
        ax.vlines(x_break, 0, break_mean[si], color=BAD_COLOR, alpha=0.75, linewidth=5)
        ax.plot(x_break, break_mean[si], "o", markersize=5, color=BAD_COLOR)

    # x-axis labels: just Sn
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [f"S{sc['level'].replace('level_', '')}" for sc in scopes],
        fontsize=10,
    )
    ax.set_ylabel("Mean prevalence")

    # Spine adjustments
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(x_centers[0], x_centers[-1])
    x_pad = bar_width * 2.0
    ax.set_xlim(x_centers[0] - x_pad, x_centers[-1] + x_pad)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.2, 0.4, 0.6])

    # Legend
    legend_elements = [
        Patch(facecolor=GOOD_COLOR, alpha=0.7, label="Productive motifs"),
        Patch(facecolor=BAD_COLOR, alpha=0.7, label="Reasoning breakdowns"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        frameon=False,
    )

    # Draw per-environment brackets below the x-axis
    env_idx = 0
    prev_env = None
    env_start = 0
    for si, sc in enumerate(scopes):
        if sc["env"] != prev_env:
            if prev_env is not None:
                _draw_env_bracket(ax, prev_env, x_centers[env_start], x_centers[si - 1])
                env_idx += 1
            env_start = si
            prev_env = sc["env"]
    if prev_env is not None:
        _draw_env_bracket(ax, prev_env, x_centers[env_start], x_centers[n_scopes - 1])

    # Draw environment group brackets below the env brackets
    prev_group = None
    group_start = 0
    for si, sc in enumerate(scopes):
        if sc["group"] != prev_group:
            if prev_group is not None:
                _draw_group_bracket(
                    ax, prev_group, x_centers[group_start], x_centers[si - 1]
                )
            group_start = si
            prev_group = sc["group"]
    if prev_group is not None:
        _draw_group_bracket(
            ax, prev_group, x_centers[group_start], x_centers[n_scopes - 1]
        )

    fig.subplots_adjust(bottom=0.45, left=0.08, right=0.97, top=0.97)
    _save(fig, out / "env_level_pattern_prevalence.pdf")


def _draw_group_bracket(
    ax: plt.Axes, group_name: str, x_left: float, x_right: float
) -> None:
    """Draw a coloured bracket and label below the env brackets."""
    color = ENV_GROUP_COLORS.get(group_name, "#333333")
    display = ENV_GROUP_DISPLAY.get(group_name, group_name)
    y_bot = ax.get_ylim()[0]
    y_range = ax.get_ylim()[1] - y_bot
    y_bracket = y_bot - y_range * 0.48
    ax.plot(
        [x_left, x_right],
        [y_bracket, y_bracket],
        color=color,
        linewidth=2.5,
        clip_on=False,
        solid_capstyle="round",
    )
    ax.text(
        (x_left + x_right) / 2,
        y_bracket - y_range * 0.03,
        display,
        va="top",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=color,
        clip_on=False,
    )


def _draw_env_bracket(ax: plt.Axes, env: str, x_left: float, x_right: float) -> None:
    """Draw a bracket and environment label below the x-axis tick labels."""
    display = ENV_SHORT_NAME.get(env, env.replace("_", " ").title())
    # Find which group this env belongs to, use that colour
    color = "#333333"
    for gname, envs in ENV_GROUPS.items():
        if env in envs:
            color = ENV_GROUP_COLORS.get(gname, color)
            break
    y_bot = ax.get_ylim()[0]
    y_range = ax.get_ylim()[1] - y_bot
    y_bracket = y_bot - y_range * 0.08
    ax.plot(
        [x_left, x_right],
        [y_bracket, y_bracket],
        color=color,
        linewidth=2.0,
        clip_on=False,
        solid_capstyle="round",
    )
    # Shift catalyst label slightly left to avoid overlap
    x_text = (x_left + x_right) / 2
    if env == "catalyst":
        x_text -= 0.15
    ax.text(
        x_text,
        y_bracket - y_range * 0.02,
        display,
        va="top",
        ha="right",
        rotation=45,
        rotation_mode="anchor",
        fontsize=10,
        color=color,
        clip_on=False,
    )


def main(
    summary_path: str = str(DEFAULT_SUMMARY_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    """Generate the individual-pattern bar chart (horizontal) with heatmap.

    Args:
        summary_path: Path to the annotation_summary.json file.
        output_dir: Directory where the figure will be written.
    """
    summary = load_summary(Path(summary_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating individual pattern bar chart + heatmap in {out} ...")
    plot(summary, out)
    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(main)
