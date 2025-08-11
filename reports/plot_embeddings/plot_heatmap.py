from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize


def plot_embedding_heatmaps(
    task,
    verbosity,
    distance_metric="cosine",
    normalize_embeddings=True,
    annotate_threshold=15,
):
    """
    Plot annotated heatmaps showing distances between embeddings.

    Parameters:
    - task: string, task name
    - verbosity: string, verbosity level
    - distance_metric: 'cosine' or 'euclidean'
    - normalize_embeddings: bool, whether to L2 normalize embeddings
    - annotate_threshold: if number of items is <= this threshold, show values in cells
    """
    save_dir = Path.cwd() / "heatmaps" / task

    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    # Load embeddings and names
    tool_embeddings = np.load(f"embeddings/{task}_embeddings_{verbosity}.npy")
    tool_names = np.load(f"embeddings/{task}_tool_names_{verbosity}.npy")
    task_embeddings = np.load(f"embeddings/{task}_tasks_embeddings.npy")
    task_names = np.load(f"embeddings/{task}_task_names.npy")

    # Decode bytes to strings if necessary
    if tool_names.dtype.type is np.bytes_:
        tool_names = [name.decode("utf-8") for name in tool_names]
    if task_names.dtype.type is np.bytes_:
        task_names = [name.decode("utf-8") for name in task_names]

    # Normalize embeddings if requested
    if normalize_embeddings:
        tool_embeddings = normalize(tool_embeddings, norm="l2")
        task_embeddings = normalize(task_embeddings, norm="l2")
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
    show_task_annot = len(task_names) <= annotate_threshold
    show_cross_annot = (
        len(tool_names) <= annotate_threshold and len(task_names) <= annotate_threshold
    )

    # 1. Tools vs Tools heatmap
    plt.figure(figsize=(max(8, len(tool_names) * 0.8), max(8, len(tool_names) * 0.8)))
    tool_distances = distance_func(tool_embeddings)

    sns.heatmap(
        tool_distances,
        xticklabels=tool_names,
        yticklabels=tool_names,
        cmap="viridis",
        annot=show_tool_annot,
        fmt=".3f" if show_tool_annot else "",
        square=True,
        cbar_kws={"label": metric_name},
        vmin=0,
        vmax=1,
    )

    title = f"Tools vs Tools - {metric_name}"
    if normalize_embeddings:
        title += " (L2 Normalized)"
    plt.title(title)
    plt.xlabel("Tools")
    plt.ylabel("Tools")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        Path(save_dir)
        / f"{task}_{verbosity}_tools_vs_tools_{distance_metric}{norm_suffix}_heatmap.pdf"
    )
    plt.close()

    # 2. Tasks vs Tasks heatmap
    plt.figure(figsize=(max(8, len(task_names) * 0.8), max(8, len(task_names) * 0.8)))
    task_distances = distance_func(task_embeddings)

    sns.heatmap(
        task_distances,
        xticklabels=task_names,
        yticklabels=task_names,
        cmap="viridis",
        annot=show_task_annot,
        fmt=".3f" if show_task_annot else "",
        square=True,
        cbar_kws={"label": metric_name},
        vmin=0,
        vmax=1,
    )

    title = f"Tasks vs Tasks - {metric_name}"
    if normalize_embeddings:
        title += " (L2 Normalized)"
    plt.title(title)
    plt.xlabel("Tasks")
    plt.ylabel("Tasks")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        Path(save_dir)
        / f"{task}_{verbosity}_tasks_vs_tasks_{distance_metric}{norm_suffix}_heatmap.pdf"
    )
    plt.close()

    # 3. Tools vs Tasks heatmap
    plt.figure(figsize=(max(10, len(task_names) * 0.8), max(8, len(tool_names) * 0.6)))
    tools_vs_tasks_distances = distance_func(tool_embeddings, task_embeddings)

    sns.heatmap(
        tools_vs_tasks_distances,
        xticklabels=task_names,
        yticklabels=tool_names,
        cmap="viridis",
        annot=show_cross_annot,
        fmt=".3f" if show_cross_annot else "",
        cbar_kws={"label": metric_name},
        vmin=0,
        vmax=1,
    )

    title = f"Tools vs Tasks - {metric_name}"
    if normalize_embeddings:
        title += " (L2 Normalized)"
    plt.title(title)
    plt.xlabel("Tasks")
    plt.ylabel("Tools")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        Path(save_dir)
        / f"{task}_{verbosity}_tools_vs_tasks_{distance_metric}{norm_suffix}_heatmap.pdf"
    )
    plt.close()

    norm_status = (
        "with L2 normalization" if normalize_embeddings else "without normalization"
    )
    logger.info(
        f"Generated heatmaps for {task} - {verbosity} using {distance_metric} distance {norm_status}"
    )


def compare_normalization_effects(task, verbosity, distance_metric="cosine"):
    """
    Generate both normalized and unnormalized heatmaps for comparison.
    """
    logger.info(f"\n=== Comparing normalization effects for {task} - {verbosity} ===")

    # Generate unnormalized heatmaps
    plot_embedding_heatmaps(
        task,
        verbosity,
        distance_metric,
        normalize_embeddings=False,
        annotate_threshold=15,
    )

    # Generate normalized heatmaps
    plot_embedding_heatmaps(
        task,
        verbosity,
        distance_metric,
        normalize_embeddings=True,
        annotate_threshold=15,
    )


if __name__ == "__main__":
    tasks = ["spectra_elucidation", "ml", "catalyst", "corral_md"]
    verbosity_levels = ["brief", "workflow", "full"]

    for task in tasks:
        for verbosity in verbosity_levels:
            # Generate normalized heatmaps (recommended default)
            plot_embedding_heatmaps(
                task,
                verbosity,
                distance_metric="cosine",
                normalize_embeddings=True,
                annotate_threshold=15,
            )

            # Optionally compare normalization effects
            # compare_normalization_effects(task, verbosity, distance_metric='cosine')
