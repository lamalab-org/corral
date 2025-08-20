import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

# First, run your existing processing code to generate the averaged results
path_file = Path("processed_results.json")
with path_file.open() as file:
    data = json.load(file)

all_subtasks = []
for task in data:
    # Only process if chained is True and verbosity_level is 'comprehensive'
    if not (
        task.get("chained") is True and task.get("verbosity_level") == "comprehensive"
    ):
        continue

    source_files = task.get("source_files", [])
    if not source_files:
        raise ValueError(f"Task {task} has no source files.")

    for source_file in source_files:
        if task.get("env") == "Catalyst" or task.get("env") == "ML":
            logger.info(f"Skipping {source_file} as it is not MD or Spectra.")
            continue
        if task.get("env") == "MD":
            with Path(source_file).open("r") as file:
                content = json.load(file)
            subtasks = content.get("task_results", [])
            tasks = []
            names_in_order = []
            for task_name, scores in subtasks.items():
                if not isinstance(scores, dict):
                    raise ValueError(
                        f"Task {task_name} in {source_file} has invalid scores."
                    )
                if "alpha_quartz" in task_name:
                    task_id = task_name.split("_")[2:]
                    name = "_".join(task_id)
                else:
                    task_id = task_name.split("_")[1:]
                    name = "_".join(task_id)
                tasks.append({"name": name, "scores": scores})
                if name not in names_in_order:
                    names_in_order.append(name)

            # Compute average 'average_score' for each unique subtask name
            subtask_scores = []
            for subtask_name in names_in_order:
                avg_scores = [
                    t["scores"].get("average_score")
                    for t in tasks
                    if t["name"] == subtask_name and "average_score" in t["scores"]
                ]
                if avg_scores:
                    subtask_scores.append(np.mean(avg_scores))
                else:
                    subtask_scores.append(None)

            if "melting" in source_file:
                task_name = "melting"
            elif "quenching" in source_file:
                task_name = "quenching"
            elif "surface_energy" in source_file:
                task_name = "surface_energy"
            else:
                task_name = None

            all_subtasks.append(
                {
                    "name": task_name,
                    "agent_type": task.get("agent_type"),
                    "model": task.get("model"),
                    "subtasks": names_in_order,
                    "scores": subtask_scores,
                }
            )
        elif task.get("env") == "Spectra":
            with Path(source_file).open("r") as file:
                content = json.load(file)
            subtasks = content.get("task_results", [])
            tasks = []
            names_in_order = []
            for task_name, scores in subtasks.items():
                if not isinstance(scores, dict):
                    raise ValueError(
                        f"Task {task_name} in {source_file} has invalid scores."
                    )
                task_id = task_name.split("_")[-2:]
                name = "_".join(task_id)
                tasks.append({"name": name, "scores": scores})
                if name not in names_in_order:
                    names_in_order.append(name)

            # Compute average 'average_score' for each unique subtask name
            subtask_scores = []
            for subtask_name in names_in_order:
                avg_scores = [
                    t["scores"].get("average_score")
                    for t in tasks
                    if t["name"] == subtask_name and "average_score" in t["scores"]
                ]
                if avg_scores:
                    subtask_scores.append(np.mean(avg_scores))
                else:
                    subtask_scores.append(None)

            all_subtasks.append(
                {
                    "name": "Spectra",
                    "agent_type": task.get("agent_type"),
                    "model": task.get("model"),
                    "subtasks": names_in_order,
                    "scores": subtask_scores,
                }
            )

    for source_file in source_files:
        if task.get("env") != "ML" and task.get("env") != "Catalyst":
            logger.info(
                f"Skipping {source_file} as it is not for ML or Catalyst environment."
            )
            continue
        with Path(source_file).open("r") as file:
            content = json.load(file)
        subtasks = content.get("task_results", [])
        tasks = []
        names_in_order = []
        for task_name, scores in subtasks.items():
            if not isinstance(scores, dict):
                raise ValueError(
                    f"Task {task_name} in {source_file} has invalid scores."
                )
            if "batch_retrieve" in task_name:
                cleaned_task_name = (
                    task_name.replace("oxide_", "")
                    .replace("nitride_", "")
                    .replace("sulphide_", "")
                )
            else:
                cleaned_task_name = task_name

            name = cleaned_task_name
            tasks.append({"name": name, "scores": scores})
            if name not in names_in_order:
                names_in_order.append(name)

            # Compute average 'average_score' for each unique subtask name
        subtask_scores = []
        for subtask_name in names_in_order:
            avg_scores = [
                t["scores"].get("average_score")
                for t in tasks
                if t["name"] == subtask_name and "average_score" in t["scores"]
            ]
            if avg_scores:
                subtask_scores.append(np.mean(avg_scores))
            else:
                subtask_scores.append(None)

            _name = "ML" if task.get("env") == "ML" else "Catalyst"

            all_subtasks.append(
                {
                    "name": _name,
                    "agent_type": task.get("agent_type"),
                    "model": task.get("model"),
                    "subtasks": names_in_order,
                    "scores": subtask_scores,
                }
            )


# Consistency check: for all entries with the same name, subtasks must be identical and scores length must match subtasks length
subtasks_by_name = defaultdict(list)
for entry in all_subtasks:
    subtasks_by_name[entry["name"]].append(entry)

for name, entries in subtasks_by_name.items():
    if not entries:
        continue
    first_subtasks = entries[0]["subtasks"]
    for entry in entries:
        if entry["subtasks"] != first_subtasks:
            raise ValueError(
                f"Inconsistent subtasks for name '{name}': {entry['subtasks']} != {first_subtasks}"
            )
        if len(entry["scores"]) != len(entry["subtasks"]):
            raise ValueError(
                f"Scores length does not match subtasks length for name '{name}': {len(entry['scores'])} != {len(entry['subtasks'])}"
            )

# NEW: Average scores for entries with the same name, agent_type, and model
grouped_entries = defaultdict(list)
for entry in all_subtasks:
    key = (entry["name"], entry["agent_type"], entry["model"])
    grouped_entries[key].append(entry)

averaged_subtasks = []
for key, entries in grouped_entries.items():
    name, agent_type, model = key

    if len(entries) == 1:
        # Only one entry, no averaging needed
        averaged_subtasks.append(entries[0])
    else:
        # Multiple entries, need to average scores element-wise
        first_entry = entries[0]
        subtasks = first_entry["subtasks"]

        # Collect all score lists
        all_scores = [entry["scores"] for entry in entries]

        # Average element-wise
        averaged_scores = []
        for i in range(len(subtasks)):
            # Get all scores at position i, filtering out None values
            scores_at_i = [scores[i] for scores in all_scores if scores[i] is not None]

            if scores_at_i:
                averaged_scores.append(np.mean(scores_at_i))
            else:
                averaged_scores.append(None)

        averaged_entry = {
            "name": name,
            "agent_type": agent_type,
            "model": model,
            "subtasks": subtasks,
            "scores": averaged_scores,
        }
        averaged_subtasks.append(averaged_entry)

with Path("averaged_results_subtasks.json").open("w") as f:
    json.dump(averaged_subtasks, f, indent=4)
