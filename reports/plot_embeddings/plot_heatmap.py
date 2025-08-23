"""
# Calculation of Distance Metrics for Embeddings

The embeddings are obtained using a pre-trained model, specifically the `text-embedding-3-large` model by the scripts `embedding_tool_analysis.py` and `embed_tasks.py`. The distance between embeddings is calculated using the cosine distance metric, and the results are saved as JSON files for further analysis. Below is a detailed breakdown of the entire process.

## 1. Overview

In this process, embeddings for tasks and tools are loaded, and the distances between them are calculated using a distance metric, typically the cosine distance. The embeddings represent the tasks and tools in a high-dimensional space, allowing us to compare their similarities or differences. The embeddings are normalized (if requested) before calculating the distances.

The distance metric used in this analysis is **cosine distance**, which is calculated as:

$$
D_{\\text{cosine}}(A, B) = 1 - \\frac{A \\cdot B}{\\|A\\|_2 \\|B\\|_2}
$$

where:

* $A$ and $B$ are the vectors (embeddings) representing the two entities (tasks or tools),
* $\\|A\\|_2$ and $\\|B\\|_2$ are the L2 norms (magnitudes) of the vectors,
* $A \\cdot B$ is the dot product between the vectors.

The cosine distance provides a measure of how similar or dissimilar two vectors are. A smaller value indicates higher similarity, while a larger value indicates lower similarity.

The embeddings are loaded and processed into two categories: tools and tasks.

## 2. Normalization of Embeddings

Before calculating distances, the embeddings can be normalized to ensure that all vectors are on the same scale. This is done by applying L2 normalization, which scales each vector so that its Euclidean norm (magnitude) is 1. Normalized embeddings ensure that the distance calculations focus on the directionality of the vectors rather than their magnitudes.

The L2 normalization of a vector $A$ is performed as:

$$
\\hat{A} = \\frac{A}{\\|A\\|_2}
$$

where $\\hat{A}$ is the normalized vector, and $\\|A\\|_2$ is the Euclidean norm of the vector.

If normalization is applied, the cosine distance is calculated using these normalized vectors.

## 3. Distance Calculation

The key step in this process is the calculation of distances between embeddings. The three main categories of distances are:

* **Tools vs Tools**: The cosine distance between the embeddings of the tools.
* **Tasks vs Tasks**: The cosine distance between the embeddings of the tasks.
* **Tools vs Tasks**: The cosine distance between the embeddings of tools and tasks.

The distance function used is chosen based on the parameter `distance_metric`. Currently, only **cosine distance** is supported, but the framework allows for flexibility in case other distance metrics (e.g., Euclidean distance) need to be incorporated.

For each of the three distance categories, the distance matrices are calculated as follows:

$$
\\text{Distance Matrix}(X, Y) = \\text{Cosine Distance}(X, Y)
$$

where:

* $X$ is the matrix of embeddings for one set (tools or tasks),
* $Y$ is the matrix of embeddings for the other set (tools or tasks).

These distance matrices are then stored for further analysis and visualization.

### 3.1 Tools vs Tools

This matrix compares the embeddings of all tools to one another, generating a symmetric matrix of distances. Each entry $D_{ij}$ in the matrix represents the cosine distance between tool $i$ and tool $j$.

### 3.2 Tasks vs Tasks

This matrix compares the embeddings of all tasks to one another. Similarly to the tools matrix, it is a symmetric matrix of distances where each entry represents the distance between tasks.

### 3.3 Tools vs Tasks

This matrix compares the embeddings of all tools against all tasks. It is a rectangular matrix where each entry $D_{ij}$ represents the distance between tool $i$ and task $j$.

### 3.4 Statistical Analysis

Once the distance matrices are computed, several statistical measures are calculated for analysis:

* **Mean distance**: The average cosine distance between all pairs of tasks and tools.
* **Min distance**: The smallest distance between any pair of tasks and tools.
* **Max distance**: The largest distance between any pair of tasks and tools.
* **Mean of minimum distances per task**: The mean of the smallest distances for each task against all tools.
* **Mean consecutive task distance**: The average cosine distance between consecutive tasks, providing insights into the relationships between tasks.

## 4. Visualization

Heatmaps are generated for the distance matrices. Each heatmap represents the pairwise distances between embeddings, where the color intensity corresponds to the magnitude of the distance.

The heatmaps are generated for the following categories:

* **Tools vs Tools**
* **Tasks vs Tasks**
* **Tools vs Tasks**

These visualizations help in understanding the relationships between different tasks and tools based on their embedding vectors.

## 5. Results and Storage

The distance matrices, along with the associated statistical measures, are stored in JSON files. These files are organized as follows:

* `task_{verbosity}_tools_vs_tasks_stats_{distance_metric}{norm_suffix}.json`: Contains the statistics for tools vs tasks distance calculations.
* Additionally, the distance matrices are saved as heatmap images in PDF and PNG formats.

An example of the saved file naming convention is:

```
task_ml_tools_vs_tasks_stats_cosine_normalized.json
```

## 6. Conclusion

This method provides a systematic approach to calculating and visualizing the relationships between tasks and tools based on their embeddings. The use of cosine distance allows for a meaningful comparison of how similar or different the descriptions of tasks and tools are in the high-dimensional embedding space. The statistical analysis and heatmap visualizations aid in understanding these relationships in a comprehensive manner.

"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize

ML_TOOLS = [
    "train_xgboost_model",
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

    # Calculate tool count based on predefined lists for ML and Catalyst
    if task == "ml":
        expected_tool_count = len(ML_TOOLS)
    elif task == "catalyst":
        expected_tool_count = len(CATALYST_TOOLS)
    else:
        expected_tool_count = len(tool_names)

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
        "tool_count": expected_tool_count,
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
