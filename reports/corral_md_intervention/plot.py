import json
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE

# from sentence_transformers import SentenceTransformer
from statsmodels.stats.proportion import proportion_confint

# Define directories for the three conditions
condition_dirs = {
    "successful_context_18": "corral_md_intervention_success/workflow_18",
    "successful_context_12": "corral_md_intervention_success/workflow_12",
    # "failed": "gpt_oss_intervention_failed",
    "none": "corral_md_intervention/workflow",
}

# Assume report file names based on user's example
report_files = {
    "successful_context_18": "gpt_oss_120-react-corral_md_successful-workflow_verbosity-single_18_try.json",
    "successful_context_12": "gpt_oss_120-react-corral_md_successful-workflow_verbosity-single_12_try.json",
    # "failed": "gpt_oss_120-react-resistor_failed-workflow_verbosity-single_try.json",
    "none": "gpt_oss_120-react-corral_md_none-workflow_verbosity-single_try.json",  # Adjust if different
}

# Load sentence transformer for embeddings
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


def consolidate_results():
    all_data = {}
    for cond, dir_name in condition_dirs.items():
        cond_dir = Path(dir_name)
        report_path = cond_dir / report_files[cond]
        with report_path.open() as f:
            report = json.load(f)

        task_results = report["task_results"]["aluminum_surface_energy_1"]
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

        all_data[cond] = {
            "success_rate": success_rate,
            "ci_lower": lower,
            "ci_upper": upper,
            "trace_metrics": trace_metrics,
        }
    return all_data


def plot_bar_charts(data):
    conditions = list(data.keys())
    rates = [data[c]["success_rate"] for c in conditions]
    lowers = [data[c]["ci_lower"] for c in conditions]
    uppers = [data[c]["ci_upper"] for c in conditions]
    yerr = (
        [rates[i] - lowers[i] for i in range(len(rates))],
        [uppers[i] - rates[i] for i in range(len(rates))],
    )

    plt.figure(figsize=(8, 6))
    plt.bar(conditions, rates, yerr=yerr, capsize=5)
    plt.ylabel("Success Rate")
    plt.title("Success Rates Across Conditions")
    plt.savefig("bar_success_rates.pdf", format="pdf")
    plt.close()


def plot_box_violin(data):
    # For avg_logprob
    logprob_data = [
        {"condition": cond, "avg_logprob": tm.get("avg_logprob", 0)}
        for cond in data
        for tm in data[cond]["trace_metrics"]
    ]

    logprob_df = pd.DataFrame(logprob_data)

    plt.figure(figsize=(8, 6))
    sns.violinplot(x="condition", y="avg_logprob", data=logprob_df)
    plt.title("Distribution of Average Log Probs per Trace")
    plt.savefig("violin_avg_logprob.pdf", format="pdf")
    plt.close()

    # Early vs Late
    df_early_late = []
    for cond in data:
        for tm in data[cond]["trace_metrics"]:
            df_early_late.append(
                {
                    "condition": cond,
                    "stage": "early",
                    "value": tm.get("early_avg_logprob", 0),
                }
            )
            df_early_late.append(
                {
                    "condition": cond,
                    "stage": "late",
                    "value": tm.get("late_avg_logprob", 0),
                }
            )
    df_el = pd.DataFrame(df_early_late)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="condition", y="value", hue="stage", data=df_el)
    plt.title("Early vs Late Average Log Probs per Condition")
    plt.savefig("box_early_late_logprob.pdf", format="pdf")
    plt.close()


def plot_line_plots(data):
    num_points = 100
    positions = np.linspace(0, 1, num_points)

    # Cumulative avg logprob
    cum_avgs = {cond: [] for cond in data}
    for cond in data:
        for tm in data[cond]["trace_metrics"]:
            if "cum_avg_func" in tm:
                cum_avgs[cond].append(tm["cum_avg_func"](positions))
    plt.figure(figsize=(10, 6))
    for cond in cum_avgs:
        if cum_avgs[cond]:
            mean_cum = np.mean(cum_avgs[cond], axis=0)
            std_cum = np.std(cum_avgs[cond], axis=0)
            plt.plot(positions, mean_cum, label=cond)
            plt.fill_between(
                positions, mean_cum - std_cum, mean_cum + std_cum, alpha=0.2
            )
    plt.xlabel("Normalized Generation Step")
    plt.ylabel("Cumulative Average Log Prob")
    plt.legend()
    plt.title("Cumulative Average Log Prob Over Generation Steps")
    plt.savefig("line_cum_logprob.pdf", format="pdf")
    plt.close()

    # Entropy over steps
    cum_ents = {cond: [] for cond in data}
    for cond in data:
        for tm in data[cond]["trace_metrics"]:
            if "cum_ent_func" in tm:
                cum_ents[cond].append(tm["cum_ent_func"](positions))
    plt.figure(figsize=(10, 6))
    for cond in cum_ents:
        if cum_ents[cond]:
            mean_ent = np.mean(cum_ents[cond], axis=0)
            std_ent = np.std(cum_ents[cond], axis=0)
            plt.plot(positions, mean_ent, label=cond)
            plt.fill_between(
                positions, mean_ent - std_ent, mean_ent + std_ent, alpha=0.2
            )
    plt.xlabel("Normalized Generation Step")
    plt.ylabel("Cumulative Average Top Entropy")
    plt.legend()
    plt.title("Cumulative Average Top Entropy Over Generation Steps")
    plt.savefig("line_cum_entropy.pdf", format="pdf")
    plt.close()


def plot_scatter(data):
    scatter_data = []
    rng = np.random.default_rng()
    for cond in data:
        for tm in data[cond]["trace_metrics"]:
            success = 1 if tm["success"] else 0
            jitter = success + 0.05 * rng.standard_normal()
            scatter_data.append(
                {
                    "condition": cond,
                    "avg_logprob": tm.get("avg_logprob", 0),
                    "success_jitter": jitter,
                }
            )

    scatter_df = pd.DataFrame(scatter_data)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="avg_logprob",
        y="success_jitter",
        hue="condition",
        data=scatter_df,
        alpha=0.7,
    )
    plt.ylabel("Success (with jitter)")
    plt.title("Success vs Average Log Prob by Condition")
    plt.savefig("scatter_success_logprob.pdf", format="pdf")
    plt.close()


def plot_tsne(data):
    all_embeds = []
    labels = []
    successes = []
    for cond in data:
        for tm in data[cond]["trace_metrics"]:
            if "embedding" in tm:
                all_embeds.append(tm["embedding"])
                labels.append(cond)
                successes.append(1 if tm["success"] else 0)

    if not all_embeds:
        return

    embeds = np.array(all_embeds)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeds = tsne.fit_transform(embeds)

    plt.figure(figsize=(10, 8))
    for cond in set(labels):
        idx = [i for i, label in enumerate(labels) if label == cond]
        plt.scatter(tsne_embeds[idx, 0], tsne_embeds[idx, 1], label=cond, alpha=0.6)
    plt.legend()
    plt.title("t-SNE Projection of Trace Embeddings by Condition")
    plt.savefig("tsne_traces.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    consolidated_data = consolidate_results()
    plot_bar_charts(consolidated_data)
    plot_box_violin(consolidated_data)
    plot_line_plots(consolidated_data)
    plot_scatter(consolidated_data)
    # plot_tsne(consolidated_data)
