import json
import re
from datetime import UTC, datetime
from pathlib import Path

# Aesthetic imports
import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame
from scipy.constants import golden
from scipy.interpolate import interp1d

# Apply theme
lama_aesthetics.get_style("main")

# Figure dimensions
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

# Define steps including the new ones
steps = ["step_1", "step_2", "step_3", "step_n_1", "step_n_2"]

# Hardcoded color palette for steps
STEP_COLORS = {
    "step_1": "#768eab",  # Muted Blue
    "step_2": "#a285a6",  # Muted Purple
    "step_3": "#83a598",  # Muted Green
    "step_n_1": "#5C6F2B",  # Muted Orange
    "step_n_2": "#547792",  # Muted Brown/Tan
}

conditions = ["successful", "failed"]

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

    traces = []
    current_trace = []
    prev_iter = 0
    for file in sorted_files:
        with (logprobs_path / file).open() as f:
            data = json.load(f)
        iter_num = data.get("iteration", 0)
        if iter_num <= prev_iter:
            if current_trace:
                traces.append(current_trace)
            current_trace = []
        current_trace.append(data)
        prev_iter = iter_num
    if current_trace:
        traces.append(current_trace)
    return traces


def process_trace(trace_data):
    all_logprobs = []
    all_top_entropies = []
    for iter_data in trace_data:
        logprobs_obj = iter_data.get("logprobs", {})
        content = logprobs_obj.get("content", [])
        if not content:
            continue

        lps = [tok.get("logprob", 0) for tok in content]
        all_logprobs.extend(lps)

        for tok in content:
            top_logprobs = tok.get("top_logprobs", [])
            if top_logprobs:
                top_lps = [tl.get("logprob", 0) for tl in top_logprobs]
                top_probs = np.exp(top_lps) / np.sum(np.exp(top_lps))
                top_ent = -np.sum(top_probs * np.log(top_probs + 1e-10))
                all_top_entropies.append(top_ent)

    results = {}
    if all_logprobs:
        cumsums = np.cumsum(all_logprobs)
        positions = np.arange(1, len(all_logprobs) + 1)
        results["cum_avg_func"] = interp1d(
            positions / len(all_logprobs),
            cumsums / positions,
            kind="linear",
            fill_value="extrapolate",
        )
        results["y_range_lp"] = cumsums / positions

    if all_top_entropies:
        cum_ents = np.cumsum(all_top_entropies)
        ent_positions = np.arange(1, len(all_top_entropies) + 1)
        results["cum_ent_func"] = interp1d(
            ent_positions / len(all_top_entropies),
            cum_ents / ent_positions,
            kind="linear",
            fill_value="extrapolate",
        )
        results["y_range_ent"] = cum_ents / ent_positions

    return results


def consolidate_results(steps, cond_dir_names, report_files):
    all_data = {cond: {} for cond in cond_dir_names}
    for cond, dir_base in cond_dir_names.items():
        for step in steps:
            cond_dir = Path(step) / dir_base
            report_path = cond_dir / report_files[cond]
            if not report_path.exists():
                continue

            with report_path.open() as f:
                report = json.load(f)

            task_results = report["task_results"]["task_4"]
            successes = [trial["success"] for trial in task_results["trials"]]

            logprobs_dir = cond_dir / "logprobs"
            traces = group_files_into_traces(str(logprobs_dir))

            trace_metrics = []
            for i, trace in enumerate(traces):
                metrics = process_trace(trace)
                if i < len(successes):
                    metrics["success"] = successes[i]
                trace_metrics.append(metrics)

            all_data[cond][step] = {"trace_metrics": trace_metrics}
    return all_data


def plot_cum_for_condition(data, condition):
    num_points = 100
    positions = np.linspace(0.01, 1, num_points)

    metrics_to_plot = [
        (
            "cum_avg_func",
            "Cumulative Average Log Prob",
            f"line_cum_logprob_{condition}.pdf",
            "y_range_lp",
        ),
        (
            "cum_ent_func",
            "Cumulative Average Top Entropy",
            f"line_cum_entropy_{condition}.pdf",
            "y_range_ent",
        ),
    ]

    for key, ylabel, filename, _range_key in metrics_to_plot:
        fig, ax = plt.subplots(
            figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
        )

        all_y_values = []

        for step in steps:
            if step not in data[condition]:
                continue

            step_curves = []
            for tm in data[condition][step]["trace_metrics"]:
                if key in tm:
                    y_vals = tm[key](positions)
                    step_curves.append(y_vals)
                    all_y_values.extend(y_vals)

            if step_curves:
                mean_cum = np.mean(step_curves, axis=0)
                std_cum = np.std(step_curves, axis=0)
                color = STEP_COLORS.get(step, "gray")

                ax.plot(
                    positions,
                    mean_cum,
                    label=step.replace("_", " ").title(),
                    color=color,
                    lw=2,
                )
                ax.fill_between(
                    positions,
                    mean_cum - std_cum,
                    mean_cum + std_cum,
                    color=color,
                    alpha=0.15,
                )

        ax.set_xlabel("Normalized Generation Step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Apply Tufte-style range frame
        if all_y_values:
            range_frame(ax, positions, np.array(all_y_values))

        # Legend placement (outside to the right like the example)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10
        )

        plt.tight_layout()
        fig.subplots_adjust(right=0.8)
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    consolidated_data = consolidate_results(steps, cond_dir_names, report_files)
    for cond in conditions:
        plot_cum_for_condition(consolidated_data, cond)
