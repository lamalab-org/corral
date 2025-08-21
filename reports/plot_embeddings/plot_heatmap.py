import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize


def plot_embedding_heatmaps_fixed(
    task,
    verbosity,
    distance_metric="cosine",
    normalize_embeddings=True,
    annotate_threshold=15,
):
    """
    Plot annotated heatmaps showing distances between embeddings.
    FIXED VERSION: Addresses potential caching and display issues
    Updated to include global tasks from full embeddings
    """
    save_dir = Path.cwd() / "heatmaps" / task

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Load embeddings and names with explicit error checking
    try:
        tool_embeddings = np.load(f"embeddings/{task}_embeddings_{verbosity}.npy")
        tool_names = np.load(f"embeddings/{task}_tool_names_{verbosity}.npy")
        task_embeddings = np.load(f"embeddings/{task}_tasks_embeddings.npy")
        task_names = np.load(f"embeddings/{task}_task_names.npy")

        # Load global tasks
        full_task_embeddings = np.load(f"embeddings/{task}_tasks_embeddings_full.npy")
        full_task_names = np.load(f"embeddings/{task}_task_names_full.npy")

        # Print shapes for debugging
        logger.info(f"{verbosity} - Tool embeddings shape: {tool_embeddings.shape}")
        logger.info(f"{verbosity} - Task embeddings shape: {task_embeddings.shape}")
        logger.info(
            f"{verbosity} - Full task embeddings shape: {full_task_embeddings.shape}"
        )

    except FileNotFoundError as e:
        logger.error(f"Could not load embeddings for {task} - {verbosity}: {e}")
        return

    # Decode bytes to strings if necessary
    if tool_names.dtype.type is np.bytes_:
        tool_names = [name.decode("utf-8") for name in tool_names]
    if task_names.dtype.type is np.bytes_:
        task_names = [name.decode("utf-8") for name in task_names]
    if full_task_names.dtype.type is np.bytes_:
        full_task_names = [name.decode("utf-8") for name in full_task_names]

    # Append full tasks to existing tasks
    combined_task_embeddings = np.vstack([task_embeddings, full_task_embeddings])
    combined_task_names = list(task_names) + list(full_task_names)

    logger.info(
        f"{verbosity} - Combined task embeddings shape: {combined_task_embeddings.shape}"
    )
    logger.info(f"{verbosity} - Combined task names count: {len(combined_task_names)}")

    # Normalize embeddings if requested
    if normalize_embeddings:
        tool_embeddings = normalize(tool_embeddings, norm="l2")
        combined_task_embeddings = normalize(combined_task_embeddings, norm="l2")
        norm_suffix = "_normalized"
    else:
        norm_suffix = ""

    # Choose distance function
    if distance_metric == "cosine":
        distance_func = cosine_distances
        metric_name = "Cosine Distance"
    elif distance_metric == "euclidean":
        distance_func = euclidean_distances
        metric_name = "Euclidean Distance"
    else:
        raise ValueError("distance_metric must be 'cosine' or 'euclidean'")

    # Determine if we should annotate based on size
    show_tool_annot = len(tool_names) <= annotate_threshold
    show_task_annot = len(combined_task_names) <= annotate_threshold
    show_cross_annot = (
        len(tool_names) <= annotate_threshold
        and len(combined_task_names) <= annotate_threshold
    )

    plt.clf()
    plt.close("all")
    sns.reset_defaults()
    sns.reset_orig()

    # 1. Tools vs Tools heatmap
    fig, ax = plt.subplots(
        figsize=(max(8, len(tool_names) * 0.8), max(8, len(tool_names) * 0.8))
    )
    tool_distances = distance_func(tool_embeddings)

    # Print distance statistics AND a few actual values for debugging
    logger.info(
        f"{verbosity} - Tools vs Tools: min={tool_distances.min():.4f}, max={tool_distances.max():.4f}, mean={tool_distances.mean():.4f}"
    )
    logger.info("First few values of distance matrix:")
    logger.info(tool_distances[:3, :3])  # Print first 3x3 submatrix

    # Force matplotlib to not cache anything
    plt.ioff()  # Turn off interactive mode

    sns.heatmap(
        tool_distances,
        xticklabels=tool_names,
        yticklabels=tool_names,
        cmap="viridis",
        annot=show_tool_annot,
        fmt=".3f" if show_tool_annot else "",
        square=True,
        cbar_kws={"label": metric_name},
        ax=ax,
        vmin=0,
        vmax=1,
    )

    title = f"Tools vs Tools - {metric_name} ({verbosity})"
    if normalize_embeddings:
        title += " (L2 Normalized)"
    ax.set_title(title)
    ax.set_xlabel("Tools")
    ax.set_ylabel("Tools")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    # Save with more explicit path
    save_path = (
        save_dir
        / f"{task}_{verbosity}_tools_vs_tools_{distance_metric}{norm_suffix}_heatmap.pdf"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Also save as PNG
    save_path_png = (
        save_dir
        / f"{task}_{verbosity}_tools_vs_tools_{distance_metric}{norm_suffix}_heatmap.png"
    )
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")

    plt.clf()
    plt.close("all")
    sns.reset_defaults()
    sns.reset_orig()

    # 2. Tasks vs Tasks heatmap (now using combined tasks)
    fig, ax = plt.subplots(
        figsize=(
            max(8, len(combined_task_names) * 0.8),
            max(8, len(combined_task_names) * 0.8),
        )
    )
    task_distances = distance_func(combined_task_embeddings)

    logger.info(
        f"\n{verbosity} - Combined Tasks vs Tasks: min={task_distances.min():.4f}, max={task_distances.max():.4f}, mean={task_distances.mean():.4f}"
    )
    logger.info("First few values of distance matrix:")
    logger.info(task_distances[:3, :3])

    sns.heatmap(
        task_distances,
        xticklabels=combined_task_names,
        yticklabels=combined_task_names,
        cmap="viridis",
        annot=show_task_annot,
        fmt=".3f" if show_task_annot else "",
        square=True,
        cbar_kws={"label": metric_name},
        ax=ax,
        vmin=0,
        vmax=1,
    )

    title = f"Tasks vs Tasks (Including Global Tasks) - {metric_name}"
    if normalize_embeddings:
        title += " (L2 Normalized)"
    ax.set_title(title)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Tasks")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    save_path = (
        save_dir
        / f"{task}_{verbosity}_tasks_vs_tasks_{distance_metric}{norm_suffix}_heatmap.pdf"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Also save as PNG
    save_path_png = (
        save_dir
        / f"{task}_{verbosity}_tasks_vs_tasks_{distance_metric}{norm_suffix}_heatmap.png"
    )
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")

    plt.clf()
    plt.close("all")
    sns.reset_defaults()
    sns.reset_orig()

    # 3. Tools vs Tasks heatmap (now using combined tasks)
    fig, ax = plt.subplots(
        figsize=(max(10, len(combined_task_names) * 0.8), max(8, len(tool_names) * 0.6))
    )
    tools_vs_tasks_distances = distance_func(tool_embeddings, combined_task_embeddings)

    logger.info(
        f"\n{verbosity} - Tools vs Combined Tasks: min={tools_vs_tasks_distances.min():.4f}, max={tools_vs_tasks_distances.max():.4f}, mean={tools_vs_tasks_distances.mean():.4f}"
    )
    logger.info("First few values of distance matrix:")
    logger.info(tools_vs_tasks_distances[:3, :3])

    sns.heatmap(
        tools_vs_tasks_distances,
        xticklabels=combined_task_names,
        yticklabels=tool_names,
        cmap="viridis",
        annot=show_cross_annot,
        fmt=".3f" if show_cross_annot else "",
        cbar_kws={"label": metric_name},
        ax=ax,
        vmin=0,
        vmax=1,
    )

    title = f"Tools vs Tasks (Including Global Tasks) - {metric_name} ({verbosity})"
    if normalize_embeddings:
        title += " (L2 Normalized)"
    ax.set_title(title)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Tools")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    save_path = (
        save_dir
        / f"{task}_{verbosity}_tools_vs_tasks_{distance_metric}{norm_suffix}_heatmap.pdf"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Also save as PNG
    save_path_png = (
        save_dir
        / f"{task}_{verbosity}_tools_vs_tasks_{distance_metric}{norm_suffix}_heatmap.png"
    )
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")

    plt.clf()
    plt.close("all")
    sns.reset_defaults()
    sns.reset_orig()

    # 4. Statistical analysis for tools vs original tasks (excluding full_task_embeddings)
    tools_vs_original_tasks_distances = distance_func(tool_embeddings, task_embeddings)

    # Calculate statistical measures
    mean_distance = tools_vs_original_tasks_distances.mean()
    min_distance = tools_vs_original_tasks_distances.min()
    max_distance = tools_vs_original_tasks_distances.max()

    # Calculate mean of minimum distance for each task with tools
    min_distances_per_task = tools_vs_original_tasks_distances.min(
        axis=0
    )  # Min distance for each task across all tools
    mean_of_min_distances = min_distances_per_task.mean()

    # Calculate mean distance between consecutive tasks
    consecutive_distances = []
    if len(task_embeddings) > 1:
        for i in range(len(task_embeddings) - 1):
            dist = distance_func(
                task_embeddings[i : i + 1], task_embeddings[i + 1 : i + 2]
            )[0, 0]
            consecutive_distances.append(dist)
        mean_consecutive_distance = np.mean(consecutive_distances)
    else:
        mean_consecutive_distance = 0.0

    # Create statistics dictionary
    stats = {
        "task": task,
        "verbosity": verbosity,
        "distance_metric": distance_metric,
        "normalized": normalize_embeddings,
        "mean_distance_tools_vs_tasks": float(mean_distance),
        "min_distance_tools_vs_tasks": float(min_distance),
        "max_distance_tools_vs_tasks": float(max_distance),
        "mean_of_min_distances_per_task": float(mean_of_min_distances),
        "mean_consecutive_task_distance": float(mean_consecutive_distance),
        "tool_count": len(tool_names),
        "task_count": len(task_names),
    }

    # Save statistics to JSON file
    stats_filename = (
        f"{task}_{verbosity}_tools_vs_tasks_stats_{distance_metric}{norm_suffix}.json"
    )
    stats_path = save_dir / stats_filename

    with Path(stats_path).open("w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved statistics: {stats_path}")
    logger.info("Tools vs Original Tasks Statistics:")
    logger.info(f"  Mean distance: {mean_distance:.4f}")
    logger.info(f"  Min distance: {min_distance:.4f}")
    logger.info(f"  Max distance: {max_distance:.4f}")
    logger.info(f"  Mean of min distances per task: {mean_of_min_distances:.4f}")
    logger.info(f"  Mean consecutive task distance: {mean_consecutive_distance:.4f}")

    norm_status = (
        "with L2 normalization" if normalize_embeddings else "without normalization"
    )
    logger.info(
        f"Generated heatmaps for {task} - {verbosity} using {distance_metric} distance {norm_status}"
    )


if __name__ == "__main__":
    tasks = ["spectra_elucidation", "ml", "catalyst", "corral_md"]
    verbosity_levels = ["brief", "workflow", "full"]

    for task in tasks:
        for verbosity in verbosity_levels:
            plot_embedding_heatmaps_fixed(task, verbosity)
