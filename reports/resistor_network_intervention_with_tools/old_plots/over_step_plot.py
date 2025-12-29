import json
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# from sentence_transformers import SentenceTransformer
from statsmodels.stats.proportion import proportion_confint

# Define steps and conditions
steps = ["step_1", "step_2", "step_3"]
conditions = ["successful", "failed"]

# Define base directory names for conditions
cond_dir_names = {
    "successful": "gpt_oss_intervention_success",
    "failed": "gpt_oss_intervention_failed",
}

# Assume report file names based on user's example
report_files = {
    "successful": "gpt_oss_120-react-resistor_successful-workflow_verbosity-single_try.json",
    "failed": "gpt_oss_120-react-resistor_failed-workflow_verbosity-single_try.json",
}

# Load sentence transformer for embeddings (commented out as in original)
# embedder = SentenceTransformer("all-MiniLM-L6-v2")


def parse_timestamp(filename):
    # Extract timestamp like 20251225T023109Z
    match = re.search(r"(\d{8}T\d{6}Z)", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return None


def group_files_into_traces(logprobs_dir):
    logprobs_path = Path(logprobs_dir)
    files = [f.name for f in logprobs_path.glob("*.json")]
    # Sort by timestamp
    sorted_files = sorted(files, key=lambda f: parse_timestamp(f))

    traces = []
    current_trace = []
    prev_iter = 0
    for file in sorted_files:
        with (logprobs_path / file).open() as f:
            data = json.load(f)
        iter_num = data.get("iteration", 0)
        if iter_num <= prev_iter:  # Assuming iteration resets to 1 or small
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
    full_text = ""
    all_top_ents_seq = []  # For sequence
    for iter_data in trace_data:
        logprobs_obj = iter_data.get("logprobs", {})
        content = logprobs_obj.get("content", [])
        if not content:
            continue
        tokens = [tok.get("token", "") for tok in content]
        full_text += "".join(tokens)
        logprobs = [tok.get("logprob", 0) for tok in content]
        all_logprobs.extend(logprobs)
        top_ents = []
        for tok in content:
            top_logprobs = tok.get("top_logprobs", [])
            if top_logprobs:
                top_lps = [tl.get("logprob", 0) for tl in top_logprobs]
                top_probs = np.exp(top_lps) / np.sum(np.exp(top_lps))
                top_ent = -np.sum(top_probs * np.log(top_probs + 1e-10))
                top_ents.append(top_ent)
        all_top_entropies.extend(top_ents)
        all_top_ents_seq.extend(top_ents)

    results = {}
    if all_logprobs:
        results["avg_logprob"] = np.mean(all_logprobs)
        results["var_logprob"] = np.var(all_logprobs)
        cumsums = np.cumsum(all_logprobs)
        positions = np.arange(1, len(all_logprobs) + 1)
        results["cum_avg_func"] = interp1d(
            positions / len(all_logprobs),
            cumsums / positions,
            kind="linear",
            fill_value="extrapolate",
        )

        mid = len(all_logprobs) // 2
        results["early_avg_logprob"] = np.mean(all_logprobs[:mid]) if mid > 0 else 0
        results["late_avg_logprob"] = (
            np.mean(all_logprobs[mid:]) if mid < len(all_logprobs) else 0
        )

    if all_top_entropies:
        results["avg_top_entropy"] = np.mean(all_top_entropies)
        cum_ents = np.cumsum(all_top_ents_seq)
        ent_positions = np.arange(1, len(all_top_ents_seq) + 1)
        results["cum_ent_func"] = interp1d(
            ent_positions / len(all_top_ents_seq),
            cum_ents / ent_positions,
            kind="linear",
            fill_value="extrapolate",
        )

        ent_mid = len(all_top_ents_seq) // 2
        results["early_avg_top_entropy"] = (
            np.mean(all_top_ents_seq[:ent_mid]) if ent_mid > 0 else 0
        )
        results["late_avg_top_entropy"] = (
            np.mean(all_top_ents_seq[ent_mid:])
            if ent_mid < len(all_top_ents_seq)
            else 0
        )

    # if full_text:
    #     results["embedding"] = embedder.encode(full_text)

    return results


def consolidate_results(steps, cond_dir_names, report_files):
    all_data = {cond: {} for cond in cond_dir_names}
    for cond, dir_base in cond_dir_names.items():
        for step in steps:
            cond_dir = Path(step) / dir_base
            report_path = cond_dir / report_files[cond]
            with report_path.open() as f:
                report = json.load(f)

            task_results = report["task_results"]["task_4"]
            successes = [trial["success"] for trial in task_results["trials"]]
            success_rate = np.mean(successes)
            lower, upper = proportion_confint(
                sum(successes), len(successes), method="wilson"
            )

            logprobs_dir = cond_dir / "logprobs"
            traces = group_files_into_traces(str(logprobs_dir))

            trace_metrics = []
            for i, trace in enumerate(traces):
                metrics = process_trace(trace)
                metrics["success"] = successes[i]
                trace_metrics.append(metrics)

            all_data[cond][step] = {
                "success_rate": success_rate,
                "ci_lower": lower,
                "ci_upper": upper,
                "trace_metrics": trace_metrics,
            }
    return all_data


def plot_cum_for_condition(data, condition):
    num_points = 100
    positions = np.linspace(0, 1, num_points)

    # Cumulative avg logprob
    cum_avgs = {step: [] for step in steps}
    plt.figure(figsize=(10, 6))
    for step in data[condition]:
        for tm in data[condition][step]["trace_metrics"]:
            if "cum_avg_func" in tm:
                cum_avgs[step].append(tm["cum_avg_func"](positions))
        if cum_avgs[step]:
            mean_cum = np.mean(cum_avgs[step], axis=0)
            std_cum = np.std(cum_avgs[step], axis=0)
            plt.plot(positions, mean_cum, label=step)
            plt.fill_between(
                positions, mean_cum - std_cum, mean_cum + std_cum, alpha=0.2
            )
    plt.xlabel("Normalized Generation Step")
    plt.ylabel("Cumulative Average Log Prob")
    plt.legend()
    plt.title(
        f"Cumulative Average Log Prob Over Generation Steps - {condition.capitalize()}"
    )
    plt.savefig(f"line_cum_logprob_{condition}.pdf", format="pdf")
    plt.close()

    # Cumulative avg top entropy
    cum_ents = {step: [] for step in steps}
    plt.figure(figsize=(10, 6))
    for step in data[condition]:
        for tm in data[condition][step]["trace_metrics"]:
            if "cum_ent_func" in tm:
                cum_ents[step].append(tm["cum_ent_func"](positions))
        if cum_ents[step]:
            mean_ent = np.mean(cum_ents[step], axis=0)
            std_ent = np.std(cum_ents[step], axis=0)
            plt.plot(positions, mean_ent, label=step)
            plt.fill_between(
                positions, mean_ent - std_ent, mean_ent + std_ent, alpha=0.2
            )
    plt.xlabel("Normalized Generation Step")
    plt.ylabel("Cumulative Average Top Entropy")
    plt.legend()
    plt.title(
        f"Cumulative Average Top Entropy Over Generation Steps - {condition.capitalize()}"
    )
    plt.savefig(f"line_cum_entropy_{condition}.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    consolidated_data = consolidate_results(steps, cond_dir_names, report_files)
    for cond in conditions:
        plot_cum_for_condition(consolidated_data, cond)
