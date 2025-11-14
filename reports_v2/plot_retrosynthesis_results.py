"""
Script to analyze and plot retrosynthesis benchmark results from reports_v2.
"""

import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from scipy.constants import golden

# Figure dimensions
ONE_COL_WIDTH_INCH = 3
TWO_COL_WIDTH_INCH = 7.25
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

lama_aesthetics.get_style("main")

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Define consistent color palette for models
MODEL_COLORS = {
    "claude_sonnet_45": "#1f77b4",  # blue
    "gpt-4o": "#ff7f0e",  # orange
}

# Define consistent markers for agents
AGENT_MARKERS = {
    "ToolCalling": "o",  # circle
    "ReAct": "s",  # square
    "Unknown": "D",  # diamond
}


def get_model_color(model_name):
    """Get consistent color for a model."""
    return MODEL_COLORS.get(model_name, "#2ca02c")  # default green


def get_agent_marker(agent_name):
    """Get consistent marker for an agent."""
    return AGENT_MARKERS.get(agent_name, "o")  # default circle


def load_results(base_path):
    """Load all retrosynthesis results from the reports_v2 directory."""
    results = []
    base_path = Path(base_path)

    # Iterate through models
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        model_name = model_dir.name
        retro_path = model_dir / "retrosynthesis"

        if not retro_path.exists():
            continue

        # Iterate through levels
        for level_dir in retro_path.iterdir():
            if not level_dir.is_dir() or level_dir.name.startswith("."):
                continue

            level = level_dir.name  # e.g., "level_1", "level_2", "level_3"

            # Check both tasks and subtasks
            for task_type in ["tasks", "subtasks"]:
                task_dir = level_dir / task_type
                if not task_dir.exists():
                    continue

                # Find JSON summary files (not individual task logs)
                for json_file in task_dir.glob("*.json"):
                    # Skip individual task log files
                    if json_file.parent.name.startswith("agent_logs-"):
                        continue

                    try:
                        with Path(json_file).open("r") as f:
                            data = json.load(f)

                        # Extract metadata from filename
                        # Format: {model}-{agent}-retro_{level}_{env_type}-{verbosity}_verbosity.json
                        filename = json_file.stem

                        # Parse filename
                        if "tool_calling" in filename:
                            agent_type = "ToolCalling"
                        elif "react" in filename:
                            agent_type = "ReAct"
                        else:
                            agent_type = "Unknown"

                        # Extract verbosity
                        if "brief_verbosity" in filename:
                            verbosity = "brief"
                        elif "workflow_verbosity" in filename:
                            verbosity = "workflow"
                        elif "comprehensive_verbosity" in filename:
                            verbosity = "comprehensive"
                        else:
                            verbosity = "unknown"

                        # Extract env type from directory name (not filename)
                        # task_type is already "tasks" or "subtasks" from the directory
                        env_type = task_type

                        result = {
                            "model": model_name,
                            "agent": agent_type,
                            "level": level,
                            "env_type": env_type,
                            "verbosity": verbosity,
                            "file": str(json_file),
                            "metrics": data.get("metrics", {}),
                            "filename": filename,
                        }
                        results.append(result)
                        logger.info(
                            f"Loaded: {model_name}/{level}/{task_type}/{json_file.name}"
                        )
                    except Exception as e:
                        logger.error(f"Error loading {json_file}: {e}")

    return results


def create_dataframe(results):
    """Convert results to a pandas DataFrame."""
    rows = []
    for r in results:
        metrics = r["metrics"]
        row = {
            "model": r["model"],
            "agent": r["agent"],
            "level": r["level"],
            "env_type": r["env_type"],
            "verbosity": r["verbosity"],
            "average_score": metrics.get("average_score", 0),
            "success_rate": metrics.get("overall_success_rate", 0),
            "pass@1": metrics.get("pass@1", 0),
            "pass@2": metrics.get("pass@2", 0),
            "pass@3": metrics.get("pass@3", 0),
            "pass@4": metrics.get("pass@4", 0),
            "pass@5": metrics.get("pass@5", 0),
            "total_tasks": metrics.get("total_tasks", 0),
            "total_tool_calls": metrics.get("total_tool_calls", 0),
            "failed_tool_calls": metrics.get("failed_tool_calls", 0),
            "total_tokens": metrics.get("total_token_usage", {}).get("total_tokens", 0),
            "prompt_tokens": metrics.get("total_token_usage", {}).get(
                "prompt_tokens", 0
            ),
            "completion_tokens": metrics.get("total_token_usage", {}).get(
                "completion_tokens", 0
            ),
            "duration": metrics.get("total_benchmark_duration", 0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_success_by_level(df, output_dir):
    """Plot success rates across different levels."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot success rate by level for tasks only (no subtasks)
    df_filtered = df[df["env_type"] == "tasks"]

    if df_filtered.empty:
        logger.warning("No tasks data found for success by level plot")
        plt.close()
        return

    # Group by level, model, and agent
    grouped = (
        df_filtered.groupby(["level", "model", "agent"])["success_rate"]
        .mean()
        .reset_index()
    )

    # Plot with color for model and marker for agent
    for model in grouped["model"].unique():
        for agent in grouped["agent"].unique():
            subset = grouped[(grouped["model"] == model) & (grouped["agent"] == agent)]
            if not subset.empty:
                color = get_model_color(model)
                marker = get_agent_marker(agent)
                label = f"{model}-{agent}"
                ax.plot(
                    subset["level"],
                    subset["success_rate"],
                    marker=marker,
                    color=color,
                    label=label,
                    linewidth=2,
                    markersize=8,
                )

    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model-Agent", bbox_to_anchor=(1.05, 1), loc="upper left")
    range_frame(ax, np.array([0, 2]), np.array([0, 1]))

    plt.tight_layout()
    plt.savefig(output_dir / "success_by_level.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: success_by_level.png")
    plt.close()


def plot_verbosity_comparison(df, output_dir):
    """Compare performance across different verbosity levels."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = [
        ("success_rate", "Success Rate"),
        ("completion_tokens", "Output Tokens"),
        ("total_tool_calls", "Total Tool Calls"),
        ("duration", "Duration (seconds)"),
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        # Filter to only tasks (not subtasks)
        df_filtered = df[df["env_type"] == "tasks"]

        # Group by verbosity, model, and agent
        grouped = (
            df_filtered.groupby(["verbosity", "model", "agent"])[metric]
            .mean()
            .reset_index()
        )

        # Create grouped bar plot
        verbosity_order = ["brief", "workflow", "comprehensive"]
        grouped["verbosity"] = pd.Categorical(
            grouped["verbosity"], categories=verbosity_order, ordered=True
        )
        grouped = grouped.sort_values("verbosity")

        # Get unique model-agent combinations
        model_agent_combos = grouped[["model", "agent"]].drop_duplicates().to_numpy()

        # Calculate positions for each verbosity level with offsets for different model-agent combos
        verbosity_positions = {v: i for i, v in enumerate(verbosity_order)}
        num_combos = len(model_agent_combos)
        offset_step = 0.4 / max(
            num_combos, 1
        )  # Total width of 0.2 divided among combos for tighter spacing

        # Plot vertical lines with markers for each model-agent combination
        handles = []
        labels = []
        for combo_idx, (model, agent) in enumerate(model_agent_combos):
            subset = grouped[(grouped["model"] == model) & (grouped["agent"] == agent)]
            if subset.empty:
                continue

            color = get_model_color(model)
            marker = get_agent_marker(agent)
            label = f"{model}-{agent}"

            # Calculate x positions with offset
            x_positions = [
                verbosity_positions[v]
                + (combo_idx - num_combos / 2 + 0.5) * offset_step
                for v in subset["verbosity"]
            ]
            y_values = subset[metric].to_numpy()

            # Draw vertical lines
            for x, y in zip(x_positions, y_values, strict=True):
                ax.vlines(x, 0, y, color=color, alpha=0.6, linewidth=5)

            # Draw markers
            line = ax.plot(
                x_positions,
                y_values,
                marker=marker,
                markersize=10,
                color=color,
                alpha=0.8,
                linestyle="none",
                label=label,
            )

            handles.append(line[0])
            labels.append(label)

        ax.set_title(f"{title} by Verbosity Level", fontsize=14, fontweight="bold")
        ax.set_ylabel(title, fontsize=13)
        ax.set_xticks(range(len(verbosity_order)))
        ax.set_xticklabels(verbosity_order, fontsize=13)
        ax.tick_params(axis="y", labelsize=12)

        # Apply range_frame
        if metric == "success_rate":
            ax.set_ylim(0, 1.05)
            range_frame(ax, np.array([0, len(verbosity_order) - 1]), np.array([0, 1]))
        else:
            # For other metrics, use data range
            y_min = 0
            y_max = grouped[metric].max() * 1.1  # Add 10% padding
            range_frame(
                ax, np.array([0, len(verbosity_order) - 1]), np.array([y_min, y_max])
            )

        # # Only show legend in the first subplot
        # if idx == 0:
        #     ax.legend(handles, labels, title="Model-Agent",
        #              bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=13, title_fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / "verbosity_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: verbosity_comparison.png")
    plt.close()


def plot_agent_comparison(df, output_dir):
    """Compare ReAct vs ToolCalling agents."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ("success_rate", "Success Rate"),
        ("completion_tokens", "Output Tokens"),
        ("total_tool_calls", "Tool Calls"),
        ("duration", "Duration (seconds)"),
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Filter to only tasks (not subtasks)
        df_filtered = df[df["env_type"] == "tasks"]

        # Group by level, model, and agent
        grouped = (
            df_filtered.groupby(["level", "model", "agent"])[metric]
            .mean()
            .reset_index()
        )

        # Get unique levels and model-agent combinations
        level_order = sorted(grouped["level"].unique())
        model_agent_combos = grouped[["model", "agent"]].drop_duplicates().to_numpy()

        # Calculate positions for each level with offsets for different model-agent combos
        level_positions = {v: i for i, v in enumerate(level_order)}
        num_combos = len(model_agent_combos)
        offset_step = 0.4 / max(
            num_combos, 1
        )  # Total width of 0.2 divided among combos for tighter spacing

        # Plot vertical lines with markers for each model-agent combination
        handles = []
        labels = []
        for combo_idx, (model, agent) in enumerate(model_agent_combos):
            subset = grouped[(grouped["model"] == model) & (grouped["agent"] == agent)]
            if subset.empty:
                continue

            color = get_model_color(model)
            marker = get_agent_marker(agent)
            label = f"{model}-{agent}"

            # Calculate x positions with offset
            x_positions = [
                level_positions[v] + (combo_idx - num_combos / 2 + 0.5) * offset_step
                for v in subset["level"]
            ]
            y_values = subset[metric].to_numpy()

            # Draw vertical lines
            for x, y in zip(x_positions, y_values, strict=True):
                ax.vlines(x, 0, y, color=color, alpha=0.6, linewidth=5)

            # Draw markers
            line = ax.plot(
                x_positions,
                y_values,
                marker=marker,
                markersize=10,
                color=color,
                alpha=0.8,
                linestyle="none",
                label=label,
            )

            handles.append(line[0])
            labels.append(label)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(title, fontsize=13)
        ax.set_xticks(range(len(level_order)))
        ax.set_xticklabels(level_order, fontsize=13)
        ax.tick_params(axis="y", labelsize=12)

        # Apply range_frame
        if metric == "success_rate":
            ax.set_ylim(0, 1.05)
            range_frame(ax, np.array([0, len(level_order) - 1]), np.array([0, 1]))
        else:
            # For other metrics, use data range
            y_min = 0
            y_max = grouped[metric].max() * 1.1  # Add 10% padding
            range_frame(
                ax, np.array([0, len(level_order) - 1]), np.array([y_min, y_max])
            )

    plt.tight_layout()
    plt.savefig(output_dir / "agent_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: agent_comparison.png")
    plt.close()


def plot_model_comparison(df, output_dir):
    """Compare different models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ("success_rate", "Success Rate"),
        ("pass@1", "Pass@1"),
        ("completion_tokens", "Output Tokens"),
        ("duration", "Duration (seconds)"),
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Filter to only tasks (not subtasks)
        df_filtered = df[df["env_type"] == "tasks"]

        # Group by level, model, and agent
        grouped = (
            df_filtered.groupby(["level", "model", "agent"])[metric]
            .mean()
            .reset_index()
        )

        # Get unique levels and model-agent combinations
        level_order = sorted(grouped["level"].unique())
        model_agent_combos = grouped[["model", "agent"]].drop_duplicates().to_numpy()

        # Calculate positions for each level with offsets for different model-agent combos
        level_positions = {v: i for i, v in enumerate(level_order)}
        num_combos = len(model_agent_combos)
        offset_step = 0.4 / max(
            num_combos, 1
        )  # Total width of 0.4 divided among combos for tighter spacing

        # Plot vertical lines with markers for each model-agent combination
        handles = []
        labels = []
        for combo_idx, (model, agent) in enumerate(model_agent_combos):
            subset = grouped[(grouped["model"] == model) & (grouped["agent"] == agent)]
            if subset.empty:
                continue

            color = get_model_color(model)
            marker = get_agent_marker(agent)
            label = f"{model}-{agent}"

            # Calculate x positions with offset
            x_positions = [
                level_positions[v] + (combo_idx - num_combos / 2 + 0.5) * offset_step
                for v in subset["level"]
            ]
            y_values = subset[metric].to_numpy()

            # Draw vertical lines
            for x, y in zip(x_positions, y_values, strict=True):
                ax.vlines(x, 0, y, color=color, alpha=0.6, linewidth=5)

            # Draw markers
            line = ax.plot(
                x_positions,
                y_values,
                marker=marker,
                markersize=10,
                color=color,
                alpha=0.8,
                linestyle="none",
                label=label,
            )

            handles.append(line[0])
            labels.append(label)

        ax.set_title(f"{title} by Model", fontsize=14, fontweight="bold")
        ax.set_ylabel(title, fontsize=13)
        ax.set_xticks(range(len(level_order)))
        ax.set_xticklabels(level_order, fontsize=14)
        ax.tick_params(axis="y", labelsize=12)

        # Apply range_frame
        if metric in ["success_rate", "pass@1"]:
            ax.set_ylim(0, 1.05)
            range_frame(ax, np.array([0, len(level_order) - 1]), np.array([0, 1]))
        else:
            # For other metrics, use data range
            y_min = 0
            y_max = grouped[metric].max() * 1.1  # Add 10% padding
            range_frame(
                ax, np.array([0, len(level_order) - 1]), np.array([y_min, y_max])
            )

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: model_comparison.png")
    plt.close()


def plot_efficiency_metrics(df, output_dir):
    """Plot efficiency metrics: output tokens per task, time per task."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Filter to only tasks (not subtasks)
    df_filtered = df[df["env_type"] == "tasks"].copy()

    # Calculate efficiency metrics
    df_filtered["output_tokens_per_task"] = df_filtered[
        "completion_tokens"
    ] / df_filtered["total_tasks"].replace(0, 1)
    df_filtered["time_per_task"] = df_filtered["duration"] / df_filtered[
        "total_tasks"
    ].replace(0, 1)

    metrics_to_plot = [
        ("output_tokens_per_task", "Output Tokens per Task"),
        ("time_per_task", "Time per Task (seconds)"),
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Group by level, model, and agent
        grouped = (
            df_filtered.groupby(["level", "model", "agent"])[metric]
            .mean()
            .reset_index()
        )

        # Get unique levels and model-agent combinations
        level_order = sorted(grouped["level"].unique())
        model_agent_combos = grouped[["model", "agent"]].drop_duplicates().to_numpy()

        # Calculate positions for each level with offsets for different model-agent combos
        level_positions = {v: i for i, v in enumerate(level_order)}
        num_combos = len(model_agent_combos)
        offset_step = 0.4 / max(
            num_combos, 1
        )  # Total width of 0.4 divided among combos for tighter spacing

        # Plot vertical lines with markers for each model-agent combination
        handles = []
        labels = []
        for combo_idx, (model, agent) in enumerate(model_agent_combos):
            subset = grouped[(grouped["model"] == model) & (grouped["agent"] == agent)]
            if subset.empty:
                continue

            color = get_model_color(model)
            marker = get_agent_marker(agent)
            label = f"{model}-{agent}"

            # Calculate x positions with offset
            x_positions = [
                level_positions[v] + (combo_idx - num_combos / 2 + 0.5) * offset_step
                for v in subset["level"]
            ]
            y_values = subset[metric].to_numpy()

            # Draw vertical lines
            for x, y in zip(x_positions, y_values, strict=True):
                ax.vlines(x, 0, y, color=color, alpha=0.6, linewidth=5)

            # Draw markers
            line = ax.plot(
                x_positions,
                y_values,
                marker=marker,
                markersize=10,
                color=color,
                alpha=0.8,
                linestyle="none",
                label=label,
            )

            handles.append(line[0])
            labels.append(label)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(title, fontsize=13)
        ax.set_xticks(range(len(level_order)))
        ax.set_xticklabels(level_order, fontsize=13)
        ax.tick_params(axis="y", labelsize=12)

        # Apply range_frame
        y_min = 0
        y_max = grouped[metric].max() * 1.1  # Add 10% padding
        range_frame(ax, np.array([0, len(level_order) - 1]), np.array([y_min, y_max]))

        # Only show legend in the first subplot
        if idx == 0:
            ax.legend(
                handles,
                labels,
                title="Model-Agent",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=13,
                title_fontsize=14,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_metrics.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: efficiency_metrics.png")
    plt.close()


def plot_pass_at_k(df, output_dir):
    """Plot pass@k and pass^k metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Filter to only tasks (not subtasks)
    df_filtered = df[df["env_type"] == "tasks"]

    # Prepare data - average pass@k across all configurations for each model-agent combination
    pass_metrics = ["pass@1", "pass@2", "pass@3", "pass@4", "pass@5"]
    k_values = [1, 2, 3, 4, 5]

    # Left subplot: pass@k
    ax = axes[0]
    for model in df_filtered["model"].unique():
        for agent in df_filtered["agent"].unique():
            df_subset = df_filtered[
                (df_filtered["model"] == model) & (df_filtered["agent"] == agent)
            ]
            if df_subset.empty:
                continue

            pass_values = [df_subset[m].mean() for m in pass_metrics]
            color = get_model_color(model)
            marker = get_agent_marker(agent)
            label = f"{model}-{agent}"
            ax.plot(
                k_values,
                pass_values,
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
            )

    ax.set_title("Pass@k Performance")
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticks(k_values)
    range_frame(ax, np.array([1, 5]), np.array([0, 1]))

    # Right subplot: pass^k (cumulative probability)
    ax = axes[1]
    for model in df_filtered["model"].unique():
        for agent in df_filtered["agent"].unique():
            df_subset = df_filtered[
                (df_filtered["model"] == model) & (df_filtered["agent"] == agent)
            ]
            if df_subset.empty:
                continue

            pass_values = [df_subset[m].mean() for m in pass_metrics]
            # Calculate pass^k: probability of success in exactly k attempts
            # pass^k = pass@k - pass@(k-1)
            pass_power_k = [pass_values[0]] + [
                pass_values[i] - pass_values[i - 1] for i in range(1, len(pass_values))
            ]

            color = get_model_color(model)
            marker = get_agent_marker(agent)
            label = f"{model}-{agent}"
            ax.plot(
                k_values,
                pass_power_k,
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
                markersize=8,
            )

    ax.set_title("Pass^k Performance")
    ax.set_xlabel("k")
    ax.set_ylabel("Pass^k Rate")
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))  # Dynamic y-limit
    ax.legend().set_visible(False)  # Hide legend on second plot
    ax.set_xticks(k_values)
    range_frame(ax, np.array([1, 5]), np.array([0, 1]))

    plt.tight_layout()
    plt.savefig(output_dir / "pass_at_k.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: pass_at_k.png")
    plt.close()


def plot_task_vs_subtask_comparison(df, output_dir):
    """Compare tasks vs subtasks performance across different levels."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ("success_rate", "Success Rate"),
        ("completion_tokens", "Output Tokens"),
        ("total_tool_calls", "Tool Calls"),
        ("duration", "Duration (seconds)"),
    ]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Group by env_type, level, model, and agent
        grouped = (
            df.groupby(["env_type", "level", "model", "agent"])[metric]
            .mean()
            .reset_index()
        )

        # Create combined labels for env_type + level
        grouped["env_level"] = grouped["env_type"] + "_" + grouped["level"]

        # Get unique env_level combinations and model-agent combinations
        env_level_order = sorted(grouped["env_level"].unique())
        model_agent_combos = grouped[["model", "agent"]].drop_duplicates().to_numpy()

        # Calculate positions for each env_level with offsets for different model-agent combos
        env_level_positions = {v: i for i, v in enumerate(env_level_order)}
        num_combos = len(model_agent_combos)
        offset_step = 0.4 / max(
            num_combos, 1
        )  # Total width of 0.4 divided among combos for tighter spacing

        # Plot vertical lines with markers for each model-agent combination
        handles = []
        labels = []
        for combo_idx, (model, agent) in enumerate(model_agent_combos):
            subset = grouped[(grouped["model"] == model) & (grouped["agent"] == agent)]
            if subset.empty:
                continue

            color = get_model_color(model)
            marker = get_agent_marker(agent)
            label = f"{model}-{agent}"

            # Calculate x positions with offset
            x_positions = [
                env_level_positions[v]
                + (combo_idx - num_combos / 2 + 0.5) * offset_step
                for v in subset["env_level"]
            ]
            y_values = subset[metric].to_numpy()

            # Draw vertical lines
            for x, y in zip(x_positions, y_values, strict=True):
                ax.vlines(x, 0, y, color=color, alpha=0.6, linewidth=5)

            # Draw markers
            line = ax.plot(
                x_positions,
                y_values,
                marker=marker,
                markersize=10,
                color=color,
                alpha=0.8,
                linestyle="none",
                label=label,
            )

            handles.append(line[0])
            labels.append(label)

        ax.set_title(
            f"{title}: Tasks vs Subtasks by Level", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel(title, fontsize=13)
        ax.set_xticks(range(len(env_level_order)))
        ax.set_xticklabels(env_level_order, fontsize=10, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=12)

        # Apply range_frame
        if metric == "success_rate":
            ax.set_ylim(0, 1.05)
            range_frame(ax, np.array([0, len(env_level_order) - 1]), np.array([0, 1]))
        else:
            # For other metrics, use data range
            y_min = 0
            y_max = grouped[metric].max() * 1.1  # Add 10% padding
            range_frame(
                ax, np.array([0, len(env_level_order) - 1]), np.array([y_min, y_max])
            )

    plt.tight_layout()
    plt.savefig(output_dir / "tasks_vs_subtasks.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: tasks_vs_subtasks.png")
    plt.close()


def generate_summary_stats(df, output_dir):
    """Generate and save summary statistics."""
    summary = []

    # Overall statistics
    summary.append("=" * 80)
    summary.append("OVERALL SUMMARY STATISTICS")
    summary.append("=" * 80)
    summary.append(f"\nTotal configurations: {len(df)}")
    summary.append(f"Models: {', '.join(df['model'].unique())}")
    summary.append(f"Agents: {', '.join(df['agent'].unique())}")
    summary.append(f"Levels: {', '.join(df['level'].unique())}")
    summary.append(f"Verbosity levels: {', '.join(df['verbosity'].unique())}")

    # Best performers
    summary.append("\n" + "=" * 80)
    summary.append("BEST PERFORMERS")
    summary.append("=" * 80)

    best_success = df.loc[df["success_rate"].idxmax()]
    summary.append(f"\nHighest Success Rate: {best_success['success_rate']:.3f}")
    summary.append(f"  Model: {best_success['model']}, Agent: {best_success['agent']}")
    summary.append(
        f"  Level: {best_success['level']}, Verbosity: {best_success['verbosity']}"
    )

    # By model
    summary.append("\n" + "=" * 80)
    summary.append("PERFORMANCE BY MODEL")
    summary.append("=" * 80)
    for model in df["model"].unique():
        df_model = df[df["model"] == model]
        summary.append(f"\n{model}:")
        summary.append(f"  Avg Success Rate: {df_model['success_rate'].mean():.3f}")
        summary.append(
            f"  Avg Output Tokens: {df_model['completion_tokens'].mean():.0f}"
        )
        summary.append(f"  Avg Duration: {df_model['duration'].mean():.2f}s")

    # By agent
    summary.append("\n" + "=" * 80)
    summary.append("PERFORMANCE BY AGENT")
    summary.append("=" * 80)
    for agent in df["agent"].unique():
        df_agent = df[df["agent"] == agent]
        summary.append(f"\n{agent}:")
        summary.append(f"  Avg Success Rate: {df_agent['success_rate'].mean():.3f}")
        summary.append(
            f"  Avg Output Tokens: {df_agent['completion_tokens'].mean():.0f}"
        )
        summary.append(f"  Avg Tool Calls: {df_agent['total_tool_calls'].mean():.1f}")

    # By level
    summary.append("\n" + "=" * 80)
    summary.append("PERFORMANCE BY LEVEL")
    summary.append("=" * 80)
    for level in sorted(df["level"].unique()):
        df_level = df[df["level"] == level]
        summary.append(f"\n{level}:")
        summary.append(f"  Avg Success Rate: {df_level['success_rate'].mean():.3f}")
        summary.append(
            f"  Avg Output Tokens: {df_level['completion_tokens'].mean():.0f}"
        )
        summary.append(f"  Avg Duration: {df_level['duration'].mean():.2f}s")
        summary.append(f"  Configurations: {len(df_level)}")

    # By verbosity
    summary.append("\n" + "=" * 80)
    summary.append("PERFORMANCE BY VERBOSITY")
    summary.append("=" * 80)
    for verbosity in ["brief", "workflow", "comprehensive"]:
        if verbosity in df["verbosity"].unique():
            df_verb = df[df["verbosity"] == verbosity]
            summary.append(f"\n{verbosity}:")
            summary.append(f"  Avg Success Rate: {df_verb['success_rate'].mean():.3f}")
            summary.append(
                f"  Avg Output Tokens: {df_verb['completion_tokens'].mean():.0f}"
            )
            summary.append(f"  Avg Duration: {df_verb['duration'].mean():.2f}s")

    # Tasks vs Subtasks
    summary.append("\n" + "=" * 80)
    summary.append("TASKS vs SUBTASKS")
    summary.append("=" * 80)
    for env_type in df["env_type"].unique():
        df_env = df[df["env_type"] == env_type]
        summary.append(f"\n{env_type}:")
        summary.append(f"  Avg Success Rate: {df_env['success_rate'].mean():.3f}")
        summary.append(f"  Avg Output Tokens: {df_env['completion_tokens'].mean():.0f}")

    summary_text = "\n".join(summary)

    # Save to file
    (output_dir / "summary_statistics.txt").open("w").write(summary_text)

    logger.info("\n" + summary_text)
    logger.info("Saved: summary_statistics.txt")

    # Also save detailed CSV
    df.to_csv(output_dir / "detailed_results.csv", index=False)
    logger.info("Saved: detailed_results.csv")


def main():
    """Main execution function."""
    # Set up paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "retrosynthesis_plots"
    output_dir.mkdir(exist_ok=True)

    logger.info("Loading retrosynthesis results...")
    results = load_results(script_dir)

    if not results:
        logger.warning("No results found!")
        return

    logger.info(f"Found {len(results)} result files")

    # Create DataFrame
    df_ = create_dataframe(results)
    logger.info(f"Created DataFrame with {len(df_)} rows")
    logger.debug(f"DataFrame columns: {df_.columns.tolist()}")
    logger.debug("Sample data:")
    logger.debug(str(df_.head()))

    # Generate plots
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 80)

    plot_success_by_level(df_, output_dir)
    plot_verbosity_comparison(df_, output_dir)
    plot_agent_comparison(df_, output_dir)
    plot_model_comparison(df_, output_dir)
    plot_efficiency_metrics(df_, output_dir)
    plot_pass_at_k(df_, output_dir)
    plot_task_vs_subtask_comparison(df_, output_dir)

    # Generate summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SUMMARY STATISTICS")
    logger.info("=" * 80)
    generate_summary_stats(df_, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info(f"All plots and statistics saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
