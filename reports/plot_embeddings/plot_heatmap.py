import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize

ML_TOOLS = [
    "batch_retrieve_polymorphs",
    "filter_json_with_strategy",
    "select_polymorphs_with_strategy_to_file",
    "consolidate_polymorph_datasets",
    "select_polymorphs_with_strategy",
    "get_bulk_polymorphs_data",
    "sort_and_get_first_from_json",
    "get_bulk_polymorphs_data_to_file",
    "prepare_tabular_dataset",
    "train_xgboost_model",
    "evaluate_xgboost_model",
    "perform_cross_validation",
]
CATALYST_TOOLS = [
    "get_structure_from_mp_text",
    "enumerate_slabs_text",
    "choose_slab_text",
    "get_adsorption_sites_text",
    "choose_adsorption_site_text",
    "add_adsorbate_to_slab_text",
]


def plot_embedding_heatmaps_fixed(
    task,
    verbosity,
    distance_metric="cosine",
    normalize_embeddings=True,
    annotate_threshold=15,
    all_results=None,  # Add parameter to collect all results
):
    """
    Plot annotated heatmaps showing distances between embeddings.
    FIXED VERSION: Addresses potential caching and display issues
    Updated to include global tasks from full embeddings
    """
    save_dir = Path.cwd() / "heatmaps" / task

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    md_tasks = ["melting", "quenching", "surface_energy"]

    # Load embeddings and names with explicit error checking
    try:
        if task in md_tasks:
            tool_embeddings = np.load(
                f"embeddings/corral_md_embeddings_{verbosity}.npy"
            )
            tool_names = np.load(f"embeddings/corral_md_tool_names_{verbosity}.npy")
        else:
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
        return None

    # Decode bytes to strings if necessary
    if tool_names.dtype.type is np.bytes_:
        tool_names = [name.decode("utf-8") for name in tool_names]
    if task_names.dtype.type is np.bytes_:
        task_names = [name.decode("utf-8") for name in task_names]
    if full_task_names.dtype.type is np.bytes_:
        full_task_names = [name.decode("utf-8") for name in full_task_names]

    # Filter tools based on the predefined lists
    if task == "ml":
        allowed_tools = set(ML_TOOLS)
        # Find indices of tools that are in our allowed list
        tool_indices = [i for i, name in enumerate(tool_names) if name in allowed_tools]
        filtered_tool_names = [tool_names[i] for i in tool_indices]
        filtered_tool_embeddings = tool_embeddings[tool_indices]

        logger.info(
            f"{verbosity} - Original tools: {len(tool_names)}, Filtered tools: {len(filtered_tool_names)}"
        )
        logger.info(f"{verbosity} - Filtered tools: {filtered_tool_names}")

        # Use filtered tools for the rest of the function
        tool_names = filtered_tool_names
        tool_embeddings = filtered_tool_embeddings
    elif task == "catalyst":
        allowed_tools = set(CATALYST_TOOLS)
        # Find indices of tools that are in our allowed list
        tool_indices = [i for i, name in enumerate(tool_names) if name in allowed_tools]
        filtered_tool_names = [tool_names[i] for i in tool_indices]
        filtered_tool_embeddings = tool_embeddings[tool_indices]

        logger.info(
            f"{verbosity} - Original tools: {len(tool_names)}, Filtered tools: {len(filtered_tool_names)}"
        )
        logger.info(f"{verbosity} - Filtered tools: {filtered_tool_names}")

        # Use filtered tools for the rest of the function
        tool_names = filtered_tool_names
        tool_embeddings = filtered_tool_embeddings
    else:
        # For other tasks, keep all tools in the tool_names list
        logger.info(
            f"{verbosity} - Using all {len(tool_names)} tools for task '{task}'"
        )

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

    # Initialize dictionary to store all distance matrices for JSON export
    distance_data = {
        "task": task,
        "verbosity": verbosity,
        "distance_metric": distance_metric,
        "normalized": normalize_embeddings,
        "tool_names": list(tool_names),  # Ensure it's a list
        "task_names": list(combined_task_names),  # Ensure it's a list
    }

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

    # Store distance matrix for JSON export (convert to list)
    distance_data["tools_vs_tools"] = tool_distances.tolist()

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

    # Store distance matrix for JSON export (convert to list)
    distance_data["tasks_vs_tasks"] = task_distances.tolist()

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

    # Store distance matrix for JSON export (convert to list)
    distance_data["tools_vs_tasks"] = tools_vs_tasks_distances.tolist()

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

    # Add distance data to the global results collection
    if all_results is not None:
        # Create a unique key for this combination
        result_key = f"{task}_{verbosity}_{distance_metric}{norm_suffix}"
        all_results[result_key] = distance_data

    norm_status = (
        "with L2 normalization" if normalize_embeddings else "without normalization"
    )
    logger.info(
        f"Generated heatmaps for {task} - {verbosity} using {distance_metric} distance {norm_status}"
    )

    return distance_data  # Return the data for potential individual use


if __name__ == "__main__":
    tasks = [
        "spectra_elucidation",
        "ml",
        "catalyst",
        "melting",
        "quenching",
        "surface_energy",
    ]
    verbosity_levels = ["brief", "workflow", "full"]

    # Initialize global results dictionary
    all_distance_results = {
        "metadata": {
            "description": "Comprehensive distance matrices for all tasks and verbosity levels",
            "distance_metric": "cosine",  # Default metric
            "normalized": True,  # Default normalization
            "generated_tasks": tasks,
            "verbosity_levels": verbosity_levels,
        },
        "results": {},
    }

    # Collect all results
    for task in tasks:
        for verbosity in verbosity_levels:
            logger.info(f"Processing {task} - {verbosity}")
            distance_data = plot_embedding_heatmaps_fixed(
                task, verbosity, all_results=all_distance_results["results"]
            )

    # Save all results to a single comprehensive JSON file
    output_dir = Path.cwd() / "heatmaps"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    comprehensive_json_path = output_dir / "all_distance_matrices_comprehensive.json"
    with comprehensive_json_path.open("w") as f:
        json.dump(all_distance_results, f, indent=2)

    logger.info(f"Saved comprehensive distance matrices to: {comprehensive_json_path}")
    logger.info(f"Total combinations processed: {len(all_distance_results['results'])}")

    # Also create a summary of what's included
    summary = {
        "total_combinations": len(all_distance_results["results"]),
        "tasks_processed": list(
            {data["task"] for data in all_distance_results["results"].values()}
        ),
        "verbosity_levels_processed": list(
            {data["verbosity"] for data in all_distance_results["results"].values()}
        ),
        "combinations": list(all_distance_results["results"].keys()),
    }

    summary_path = output_dir / "distance_matrices_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary to: {summary_path}")
    logger.info(
        "All distance matrices have been consolidated into a single comprehensive JSON file!"
    )
