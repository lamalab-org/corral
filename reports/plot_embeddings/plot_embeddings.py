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
    plt.close()


if __name__ == "__main__":
    tasks = ["spectra_elucidation", "ml", "catalyst", "corral_md"]
    verbosity_levels = ["brief", "workflow", "full"]
    for task in tasks:
        for verbosity in verbosity_levels:
            plot_embeddings(task, verbosity=verbosity)
