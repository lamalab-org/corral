from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics.plotutils import range_frame
from sklearn.decomposition import PCA

lama_aesthetics.get_style("main")


def plot_embeddings(task, verbosity):
    # Load embeddings and tool names
    tool_embeddings = np.load(f"embeddings/{task}_embeddings_{verbosity}.npy")
    tool_names = np.load(f"embeddings/{task}_tool_names_{verbosity}.npy")
    task_embeddings = np.load(f"embeddings/{task}_tasks_embeddings.npy")
    task_names = np.load(f"embeddings/{task}_task_names.npy")

    # If tool_names is bytes, decode to str
    if tool_names.dtype.type is np.bytes_:
        tool_names = [name.decode("utf-8") for name in tool_names]

    if task_names.dtype.type is np.bytes_:
        task_names = [name.decode("utf-8") for name in task_names]

    # Combine all embeddings for PCA fitting
    all_embeddings = np.vstack([tool_embeddings, task_embeddings])

    # Reduce to 2D using PCA on combined embeddings
    pca = PCA(n_components=2)
    all_embeddings_2d = pca.fit_transform(all_embeddings)

    # Split back into tool and task embeddings
    tool_embeddings_2d = all_embeddings_2d[: len(tool_embeddings)]
    task_embeddings_2d = all_embeddings_2d[len(tool_embeddings) :]

    # Create directory for saving if it doesn't exist
    save_dir = Path.cwd() / task
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    # Plot combined (tools + tasks)
    plt.figure(figsize=(12, 8))
    plt.scatter(
        tool_embeddings_2d[:, 0],
        tool_embeddings_2d[:, 1],
        c="green",
        s=50,
        label="Tools",
        alpha=0.7,
    )
    plt.scatter(
        task_embeddings_2d[:, 0],
        task_embeddings_2d[:, 1],
        c="blue",
        s=50,
        label="Tasks",
        alpha=0.7,
    )
    for i, name in enumerate(tool_names):
        plt.text(
            tool_embeddings_2d[i, 0],
            tool_embeddings_2d[i, 1],
            name,
            fontsize=9,
            ha="right",
            va="bottom",
            color="darkgreen",
        )
    for i, name in enumerate(task_names):
        plt.text(
            task_embeddings_2d[i, 0],
            task_embeddings_2d[i, 1],
            name,
            fontsize=9,
            ha="right",
            va="bottom",
            color="darkblue",
        )
    # Calculate min and max for both axes (combined)
    x_min = min(np.min(tool_embeddings_2d[:, 0]), np.min(task_embeddings_2d[:, 0]))
    x_max = max(np.max(tool_embeddings_2d[:, 0]), np.max(task_embeddings_2d[:, 0]))
    y_min = min(np.min(tool_embeddings_2d[:, 1]), np.min(task_embeddings_2d[:, 1]))
    y_max = max(np.max(tool_embeddings_2d[:, 1]), np.max(task_embeddings_2d[:, 1]))
    range_frame(plt.gca(), np.array([x_min, x_max]), np.array([y_min, y_max]))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Combined Tools and Tasks Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{task}_{verbosity}_embeddings_combined_pca_plot.pdf")
    plt.savefig(Path(save_dir) / f"{task}_{verbosity}_embeddings_combined_pca_plot.png")
    plt.close()

    # Plot tools only
    plt.figure(figsize=(12, 8))
    pca_tools = PCA(n_components=2)
    tool_embeddings_2d_only = pca_tools.fit_transform(tool_embeddings)
    plt.scatter(
        tool_embeddings_2d_only[:, 0],
        tool_embeddings_2d_only[:, 1],
        c="green",
        s=50,
        alpha=0.7,
    )
    for i, name in enumerate(tool_names):
        plt.text(
            tool_embeddings_2d_only[i, 0],
            tool_embeddings_2d_only[i, 1],
            name,
            fontsize=9,
            ha="right",
            va="bottom",
            color="darkgreen",
        )
    # Calculate min and max for both axes (tools only)
    x_min_tools = np.min(tool_embeddings_2d_only[:, 0])
    x_max_tools = np.max(tool_embeddings_2d_only[:, 0])
    y_min_tools = np.min(tool_embeddings_2d_only[:, 1])
    y_max_tools = np.max(tool_embeddings_2d_only[:, 1])
    range_frame(
        plt.gca(),
        np.array([x_min_tools, x_max_tools]),
        np.array([y_min_tools, y_max_tools]),
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Tools Embeddings Only")
    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / f"{task}_{verbosity}_embeddings_tools_only_pca_plot.pdf"
    )
    plt.savefig(
        Path(save_dir) / f"{task}_{verbosity}_embeddings_tools_only_pca_plot.png"
    )
    plt.close()

    # Plot tasks only
    plt.figure(figsize=(12, 8))
    pca_tasks = PCA(n_components=2)
    task_embeddings_2d_only = pca_tasks.fit_transform(task_embeddings)
    plt.scatter(
        task_embeddings_2d_only[:, 0],
        task_embeddings_2d_only[:, 1],
        c="blue",
        s=50,
        alpha=0.7,
    )
    for i, name in enumerate(task_names):
        plt.text(
            task_embeddings_2d_only[i, 0],
            task_embeddings_2d_only[i, 1],
            name,
            fontsize=9,
            ha="right",
            va="bottom",
            color="darkblue",
        )
    # Calculate min and max for both axes (tasks only)
    x_min_tasks = np.min(task_embeddings_2d_only[:, 0])
    x_max_tasks = np.max(task_embeddings_2d_only[:, 0])
    y_min_tasks = np.min(task_embeddings_2d_only[:, 1])
    y_max_tasks = np.max(task_embeddings_2d_only[:, 1])
    range_frame(
        plt.gca(),
        np.array([x_min_tasks, x_max_tasks]),
        np.array([y_min_tasks, y_max_tasks]),
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Tasks Embeddings Only")
    plt.tight_layout()
    plt.savefig(
        Path(save_dir) / f"{task}_{verbosity}_embeddings_tasks_only_pca_plot.pdf"
    )
    plt.savefig(
        Path(save_dir) / f"{task}_{verbosity}_embeddings_tasks_only_pca_plot.png"
    )
    plt.close()


def plot_all_tasks_combined(verbosity):
    """Plot embeddings from all tasks combined with different markers for each task."""
    tasks = ["spectra_elucidation", "ml", "catalyst", "corral_md"]
    markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond

    all_tool_embeddings = []
    all_task_embeddings = []
    all_tool_names = []
    all_task_names = []
    all_tool_task_labels = []  # to track which task each point belongs to
    all_task_task_labels = []

    # Collect all embeddings from all tasks
    for task_idx, task in enumerate(tasks):
        # Load embeddings
        tool_embeddings = np.load(f"embeddings/{task}_embeddings_{verbosity}.npy")
        tool_names = np.load(f"embeddings/{task}_tool_names_{verbosity}.npy")
        task_embeddings = np.load(f"embeddings/{task}_tasks_embeddings.npy")
        task_names = np.load(f"embeddings/{task}_task_names.npy")

        # Decode bytes if needed
        if tool_names.dtype.type is np.bytes_:
            tool_names = [name.decode("utf-8") for name in tool_names]
        if task_names.dtype.type is np.bytes_:
            task_names = [name.decode("utf-8") for name in task_names]

        # Collect embeddings and metadata
        all_tool_embeddings.append(tool_embeddings)
        all_task_embeddings.append(task_embeddings)
        all_tool_names.extend(tool_names)
        all_task_names.extend(task_names)
        all_tool_task_labels.extend([task_idx] * len(tool_embeddings))
        all_task_task_labels.extend([task_idx] * len(task_embeddings))

    # Concatenate all embeddings
    all_tool_embeddings = np.vstack(all_tool_embeddings)
    all_task_embeddings = np.vstack(all_task_embeddings)
    all_embeddings = np.vstack([all_tool_embeddings, all_task_embeddings])

    # Fit PCA on all combined embeddings
    pca = PCA(n_components=2)
    all_embeddings_2d = pca.fit_transform(all_embeddings)

    # Split back into tool and task embeddings
    tool_embeddings_2d = all_embeddings_2d[: len(all_tool_embeddings)]
    task_embeddings_2d = all_embeddings_2d[len(all_tool_embeddings) :]

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Plot tools for each task with different markers
    for task_idx, task in enumerate(tasks):
        task_tool_mask = np.array(all_tool_task_labels) == task_idx
        task_task_mask = np.array(all_task_task_labels) == task_idx

        # Plot tools
        plt.scatter(
            tool_embeddings_2d[task_tool_mask, 0],
            tool_embeddings_2d[task_tool_mask, 1],
            c="green",
            marker=markers[task_idx],
            s=60,
            label=f"Tools - {task}" if task_idx == 0 else "",
            alpha=0.7,
        )

        # Plot tasks
        plt.scatter(
            task_embeddings_2d[task_task_mask, 0],
            task_embeddings_2d[task_task_mask, 1],
            c="blue",
            marker=markers[task_idx],
            s=60,
            label=f"Tasks - {task}" if task_idx == 0 else "",
            alpha=0.7,
        )

    # Add text labels for tools
    for i, (name, _task_idx) in enumerate(
        zip(all_tool_names, all_tool_task_labels, strict=False)
    ):
        plt.text(
            tool_embeddings_2d[i, 0],
            tool_embeddings_2d[i, 1],
            f"{name}",
            fontsize=8,
            ha="right",
            va="bottom",
            color="darkgreen",
            alpha=0.8,
        )

    # Add text labels for tasks
    for i, (name, _task_idx) in enumerate(
        zip(all_task_names, all_task_task_labels, strict=False)
    ):
        plt.text(
            task_embeddings_2d[i, 0],
            task_embeddings_2d[i, 1],
            f"{name}",
            fontsize=8,
            ha="right",
            va="bottom",
            color="darkblue",
            alpha=0.8,
        )

    # Calculate axis limits
    x_min = min(np.min(tool_embeddings_2d[:, 0]), np.min(task_embeddings_2d[:, 0]))
    x_max = max(np.max(tool_embeddings_2d[:, 0]), np.max(task_embeddings_2d[:, 0]))
    y_min = min(np.min(tool_embeddings_2d[:, 1]), np.min(task_embeddings_2d[:, 1]))
    y_max = max(np.max(tool_embeddings_2d[:, 1]), np.max(task_embeddings_2d[:, 1]))
    range_frame(plt.gca(), np.array([x_min, x_max]), np.array([y_min, y_max]))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"All Tasks Combined - Tools and Tasks Embeddings ({verbosity})")

    # Create custom legend
    import matplotlib.lines as mlines

    legend_elements = []

    # Add tool and task type legends
    legend_elements.append(
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=8,
            label="Tools",
            alpha=0.7,
        )
    )
    legend_elements.append(
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Tasks",
            alpha=0.7,
        )
    )

    # Add task-specific markers
    for task_idx, task in enumerate(tasks):
        legend_elements.append(
            mlines.Line2D(
                [0],
                [0],
                marker=markers[task_idx],
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label=task,
                alpha=0.7,
            )
        )

    plt.legend(handles=legend_elements, loc="best", fontsize=9)
    plt.tight_layout()

    # Save the plot
    save_dir = Path.cwd() / "all_tasks_combined"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    plt.savefig(save_dir / f"all_tasks_{verbosity}_embeddings_combined_pca_plot.pdf")
    plt.savefig(save_dir / f"all_tasks_{verbosity}_embeddings_combined_pca_plot.png")
    plt.close()


if __name__ == "__main__":
    tasks = ["spectra_elucidation", "ml", "catalyst", "corral_md"]
    verbosity_levels = ["brief", "workflow", "full"]

    # Original individual task plots
    for task in tasks:
        for verbosity in verbosity_levels:
            plot_embeddings(task, verbosity=verbosity)

    # New combined plots for all tasks
    for verbosity in verbosity_levels:
        plot_all_tasks_combined(verbosity)
