import json

import matplotlib.pyplot as plt


def plot_subtask_continuation(json_file_path, save_path="subtask_results.png"):
    # Load the JSON data
    with json_file_path.open("r") as file:
        data = json.load(file)

    # Extract task results
    task_results = data["task_results"]

    # Group tasks and extract subtasks
    grouped_tasks = {}

    for task_name, task_info in task_results.items():
        # Split task name and remove last two parts
        name_parts = task_name.split("_")
        if len(name_parts) >= 2:
            # Remove last two parts to get base task name
            base_task = "_".join(name_parts[:-2])
            # Last part is the subtask
            subtask = name_parts[-1]
        else:
            base_task = task_name
            subtask = "default"

        if base_task not in grouped_tasks:
            grouped_tasks[base_task] = {}

        # Store trials with their trial_ids
        trials = task_info.get("trials", [])
        grouped_tasks[base_task][subtask] = {
            trial.get("trial_id", i): trial.get("score", 0)
            for i, trial in enumerate(trials)
        }

    # Get all unique subtasks and trial IDs
    all_subtasks = set()
    all_trial_ids = set()
    for task_data in grouped_tasks.values():
        all_subtasks.update(task_data.keys())
        for subtask_data in task_data.values():
            all_trial_ids.update(subtask_data.keys())

    # Convert subtasks to integers for proper numerical sorting if they are numeric
    all_subtasks = sorted(
        all_subtasks, key=lambda x: int(x) if str(x).isdigit() else float("inf")
    )
    # Convert trial IDs to integers for proper numerical sorting
    all_trial_ids = sorted(
        all_trial_ids, key=lambda x: int(x) if str(x).isdigit() else float("inf")
    )

    # Create the plot
    num_tasks = len(grouped_tasks)
    num_trials = len(all_trial_ids)
    num_subtasks = len(all_subtasks)

    fig, ax = plt.subplots(
        figsize=(max(12, num_subtasks * 2), max(8, num_tasks * num_trials * 0.5))
    )

    # Plot data
    y_labels = []
    y_pos = 0

    for task_name in sorted(grouped_tasks.keys()):
        task_data = grouped_tasks[task_name]

        for trial_id in all_trial_ids:
            y_labels.append(f"{task_name}_trial_{trial_id}")

            for x, subtask in enumerate(all_subtasks):
                if subtask in task_data and trial_id in task_data[subtask]:
                    score = task_data[subtask][trial_id]
                    color = "green" if score == 1 else "red"
                    ax.scatter(x, y_pos, c=color, s=200, alpha=0.8)
                else:
                    # No data for this subtask/trial combination
                    ax.scatter(x, y_pos, c="lightgray", s=200, alpha=0.5, marker="x")

            y_pos += 1

    # Customize the plot
    ax.set_xlabel("Subtasks", fontsize=16)
    ax.set_ylabel("Task Trials", fontsize=16)
    ax.set_title(
        "Task Trial Results by Subtask (Green = Success, Red = Failure)", fontsize=18
    )

    ax.set_xticks(range(len(all_subtasks)))
    ax.set_xticklabels(all_subtasks, fontsize=12, rotation=45, ha="right")

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)

    ax.grid(True, alpha=0.3)

    # Set limits
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.set_xlim(-0.5, len(all_subtasks) - 0.5)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Score = 1 (Success)"),
        Patch(facecolor="red", label="Score = 0 (Failure)"),
        Patch(facecolor="lightgray", label="No Data"),
    ]
    ax.legend(handles=legend_elements, fontsize=14)

    plt.tight_layout()

    # Save the plot with higher DPI for better quality
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


# Usage example
if __name__ == "__main__":
    # Replace 'your_file.json' with the actual path to your JSON file
    from pathlib import Path

    paths = Path().parent
    for path in paths.glob("*.json"):
        json_file_path = str(path)
        image_name = json_file_path.replace(".json", ".png")
        plot_subtask_continuation(path, save_path=image_name)
