import json
import re
from datetime import UTC, datetime
from pathlib import Path

# Aesthetic imports
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame
from matplotlib.lines import Line2D
from scipy.constants import golden
from scipy.interpolate import interp1d

# Apply theme
lama_aesthetics.get_style("main")

# Figure dimensions
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

# Define steps
steps = ["step_1", "step_2", "step_3", "step_n_1", "step_n_2"]

# Hardcoded color palette
STEP_COLORS = {
    "step_1": "#768eab",  # Muted Blue
    "step_2": "#a285a6",  # Muted Purple
    "step_3": "#83a598",  # Muted Green
    "step_n_1": "#d48265",  # Muted Orange
    "step_n_2": "#c69c6d",  # Muted Brown/Tan
}

# Mapping line styles to outcomes
CONDITION_STYLES = {"successful": "-", "failed": "--"}

cond_dir_names = {
    "successful": "gpt_oss_intervention_success",
    "failed": "gpt_oss_intervention_failed",
}

report_files = {
    "successful": "gpt_oss_120-react-resistor_successful-workflow_verbosity-single_try.json",
    "failed": "gpt_oss_120-react-resistor_failed-workflow_verbosity-single_try.json",
}


def parse_timestamp(filename):
    match = re.search(r"(\d{8}T\d{6}Z)", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return None


def group_files_into_traces(logprobs_dir):
    logprobs_path = Path(logprobs_dir)
    if not logprobs_path.exists():
        return []
    files = [f.name for f in logprobs_path.glob("*.json")]
    sorted_files = sorted(files, key=lambda f: parse_timestamp(f))

    traces, current_trace, prev_iter = [], [], 0
    for file in sorted_files:
        with (logprobs_path / file).open() as f:
            data = json.load(f)
        iter_num = data.get("iteration", 0)
        if iter_num <= prev_iter and current_trace:
            traces.append(current_trace)
            current_trace = []
        current_trace.append(data)
        prev_iter = iter_num
    if current_trace:
        traces.append(current_trace)
    return traces


def process_trace(trace_data):
    all_logprobs, all_top_entropies = [], []
    for iter_data in trace_data:
        content = iter_data.get("logprobs", {}).get("content", [])
        for tok in content:
            all_logprobs.append(tok.get("logprob", 0))
            top_logprobs = tok.get("top_logprobs", [])
            if top_logprobs:
                top_lps = [tl.get("logprob", 0) for tl in top_logprobs]
                top_probs = np.exp(top_lps) / np.sum(np.exp(top_lps))
                all_top_entropies.append(-np.sum(top_probs * np.log(top_probs + 1e-10)))

    results = {}
    for key, data_list, _label in [
        ("cum_avg_func", all_logprobs, "lp"),
        ("cum_ent_func", all_top_entropies, "ent"),
    ]:
        if data_list:
            cum_vals = np.cumsum(data_list)
            pos = np.arange(1, len(data_list) + 1)
            results[key] = interp1d(
                pos / len(data_list),
                cum_vals / pos,
                kind="linear",
                fill_value="extrapolate",
            )
    return results


def consolidate_results():
    all_data = {cond: {} for cond in cond_dir_names}
    for cond, dir_base in cond_dir_names.items():
        for step in steps:
            cond_dir = Path(step) / dir_base
            if not cond_dir.exists():
                continue
            traces = group_files_into_traces(str(cond_dir / "logprobs"))
            all_data[cond][step] = {"trace_metrics": [process_trace(t) for t in traces]}
    return all_data


def plot_combined_results(data):
    num_points = 100
    positions = np.linspace(0.01, 1, num_points)

    plots = [
        ("cum_avg_func", "Cumulative Average Log Prob", "combined_logprob.pdf"),
        ("cum_ent_func", "Cumulative Average Top Entropy", "combined_entropy.pdf"),
    ]

    for key, ylabel, filename in plots:
        fig, ax = plt.subplots(
            figsize=(ONE_COL_GOLDEN_RATIO_HEIGHT_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
        )
        all_y_values = []

        for step in steps:
            for cond in ["successful", "failed"]:
                if step not in data[cond]:
                    continue

                step_curves = [
                    tm[key](positions)
                    for tm in data[cond][step]["trace_metrics"]
                    if key in tm
                ]

                if step_curves:
                    mean_cum = np.mean(step_curves, axis=0)
                    std_cum = np.std(step_curves, axis=0)
                    color = STEP_COLORS[step]
                    linestyle = CONDITION_STYLES[cond]

                    ax.plot(positions, mean_cum, color=color, linestyle=linestyle, lw=2)
                    ax.fill_between(
                        positions,
                        mean_cum - std_cum,
                        mean_cum + std_cum,
                        color=color,
                        alpha=0.1,
                    )
                    all_y_values.extend(mean_cum)

        ax.set_xlabel("Normalized Generation Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Range frame for Tufte aesthetic
        if all_y_values:
            range_frame(ax, positions, np.array(all_y_values))

        # --- Custom Legend ---
        # 1. Step Colors
        color_handles = [
            Line2D([0], [0], color=c, lw=3, label=s.replace("_", "-").title())
            for s, c in STEP_COLORS.items()
        ]
        # 2. Condition Styles
        style_handles = [
            Line2D([0], [0], color="black", linestyle="-", lw=2, label="Successful"),
            Line2D([0], [0], color="black", linestyle="--", lw=2, label="Failed"),
        ]

        first_legend = ax.legend(
            handles=color_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            title="Steps",
            frameon=False,
            fontsize=10,
        )
        ax.add_artist(first_legend)
        ax.legend(
            handles=style_handles,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            title="Outcome",
            frameon=False,
            fontsize=10,
        )

        plt.tight_layout()
        fig.subplots_adjust(right=0.78)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    consolidated_data = consolidate_results()
    plot_combined_results(consolidated_data)
