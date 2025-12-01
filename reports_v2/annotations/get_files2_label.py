"""Collect all accepted question files and map them to their corresponding report files,
extracting relevant metadata such as model, environment, agent type, verbosity, task ID, level, and trial information."""

import json
import re
from pathlib import Path

from loguru import logger

dir_path = Path(__file__).parent.parent.parent / "accepted_questions"
reports_base_path = Path(__file__).parent.parent.parent / "reports_v2"

files = {}
for file in dir_path.rglob("*.json"):
    file_path = str(file)

    # Extract level from the path using regex
    level_match = re.search(r"level_(\d+)", file_path)
    if not level_match:
        logger.warning(f"No level found in path: {file_path}")
        continue

    level = int(level_match.group(1))

    # Extract model from the path (e.g., gpt-4o, claude_sonnet_45)
    model_match = re.search(r"accepted_questions/([^/]+)/", file_path)
    model = model_match.group(1) if model_match else None

    # Extract environment from the path (e.g., catalyst, afm, md, etc.)
    env_match = re.search(r"accepted_questions/[^/]+/([^/]+)/", file_path)
    environment = env_match.group(1) if env_match else None

    # Extract agent type from the directory name (e.g., ReActAgent, ConcreteAgent)
    agent_match = re.search(r"agent_logs-([^-]+)-", file_path)
    agent_type = agent_match.group(1) if agent_match else None

    # Extract filename and timestamp
    filename = Path(file_path).name
    timestamp_match = re.search(r"_(\d{8}_\d{6})\.json$", filename)
    file_timestamp = timestamp_match.group(1) if timestamp_match else None

    # Extract task_id from filename (e.g., make_5_lvl3 from make_5_lvl3_20251110_122448.json)
    task_id_match = re.search(r"([^/]+)_\d{8}_\d{6}\.json$", filename)
    task_id = task_id_match.group(1) if task_id_match else None

    # Extract verbosity from the run directory path (last part after agent_logs-)
    # Pattern: agent_logs-{Agent}-{model}-{date}-{verbosity}
    agent_dir_match = re.search(r"(agent_logs-[^/]+)/", file_path)
    verbosity = None
    if agent_dir_match:
        agent_dir_name = agent_dir_match.group(1)
        # Verbosity is the last component after splitting by '-'
        verbosity = agent_dir_name.split("-")[-1]

    # Try to find the corresponding report file in reports_v2
    report_file_path = file_path.replace("accepted_questions", "reports_v2")
    report_exists = Path(report_file_path).exists()

    # Raise error if report file doesn't exist
    if not report_exists:
        raise FileNotFoundError(
            f"Report file not found for accepted question:\n"
            f"  Expected: {report_file_path}\n"
            f"  Original: {file_path}"
        )

    # Initialize trial info as None
    trial_info = None

    if (
        report_exists
        and model
        and environment
        and agent_type
        and verbosity
        and task_id
        and file_timestamp
    ):
        # Find the score file using glob pattern since there are multiple naming conventions
        # Normalize agent type for filename matching
        agent_name_map = {
            "ReActAgent": ["react", "React"],
            "ToolCallingAgent": ["toolcalling", "tool_calling", "Tool_calling"],
            "ConcreteAgent": ["concrete", "Concrete"],
        }

        # Search for score files in the tasks directory
        tasks_dir = reports_base_path / model / environment / f"level_{level}" / "tasks"

        if not tasks_dir.exists():
            raise FileNotFoundError(
                f"Tasks directory not found:\n"
                f"  Path: {tasks_dir}\n"
                f"  For file: {file_path}"
            )

        # Look for score files (not individual trial logs)
        possible_agents = agent_name_map.get(agent_type, [agent_type.lower()])
        score_file_path = None

        # For MD environment, extract task category from task_id
        # e.g., "sio2_quenching_3" -> "quenching", "aluminum_surface_energy_3" -> "surface_energy"
        task_category = None
        if environment == "md" and task_id:
            for category in ["melting", "quenching", "surface_energy"]:
                if category in task_id:
                    task_category = category
                    break

        for agent_variant in possible_agents:
            # Try multiple patterns
            patterns = []
            if task_category:
                # If we have a task category (for MD), use it in the pattern
                patterns.append(f"*{agent_variant}*{task_category}*{verbosity}*.json")
            # Also try pattern without task category as fallback
            patterns.append(f"*{agent_variant}*{verbosity}*.json")

            for pattern in patterns:
                matches = list(tasks_dir.glob(pattern))
                # Filter out agent_logs directories and wandb files
                matches = [
                    m
                    for m in matches
                    if "agent_logs-" not in str(m) and "wandb" not in str(m)
                ]

                if matches:
                    # If multiple matches, try to find the most specific one
                    if len(matches) == 1:
                        score_file_path = matches[0]
                        break
                    else:
                        # Prefer files that match both agent and verbosity (and task category if applicable)
                        for match in matches:
                            match_ok = (
                                agent_variant.lower() in match.name.lower()
                                and verbosity in match.name
                            )
                            if task_category:
                                match_ok = match_ok and task_category in match.name
                            if match_ok:
                                score_file_path = match
                                break
                        if score_file_path:
                            break

            if score_file_path:
                break

        if not score_file_path:
            raise FileNotFoundError(
                f"Score file not found:\n"
                f"  Directory: {tasks_dir}\n"
                f"  For file: {file_path}\n"
                f"  Model: {model}, Agent: {agent_type}, Env: {environment}, "
                f"Level: {level}, Verbosity: {verbosity}\n"
                f"  Available files: {list(tasks_dir.glob('*.json'))}"
            )

        try:
            with score_file_path.open() as f:
                score_data = json.load(f)

            # Get trials for this task_id
            if "task_results" not in score_data:
                raise KeyError(
                    f"'task_results' not found in score file: {score_file_path}"
                )

            if task_id not in score_data["task_results"]:
                raise KeyError(
                    f"Task ID '{task_id}' not found in score file: {score_file_path}\n"
                    f"  Available tasks: {list(score_data['task_results'].keys())}"
                )

            task_trials = score_data["task_results"][task_id].get("trials", [])

            if not task_trials:
                raise ValueError(
                    f"No trials found for task '{task_id}' in score file: {score_file_path}"
                )

            # Get all log files for this task from reports_v2 to map by timestamp
            report_dir = Path(report_file_path).parent
            task_log_files = sorted(
                report_dir.glob(f"{task_id}_*.json"),
                key=lambda x: re.search(r"_(\d{8}_\d{6})\.json$", x.name).group(1),
            )

            # Find the index of the current file by timestamp
            trial_found = False
            for idx, log_file in enumerate(task_log_files):
                log_timestamp = re.search(r"_(\d{8}_\d{6})\.json$", log_file.name)
                if log_timestamp and log_timestamp.group(1) == file_timestamp:
                    # Map to trial (trials are 1-indexed in trial_id but 0-indexed in list)
                    if idx >= len(task_trials):
                        raise IndexError(
                            f"Trial index {idx} out of range for task '{task_id}':\n"
                            f"  Found {len(task_log_files)} log files but only {len(task_trials)} trials\n"
                            f"  File: {file_path}\n"
                            f"  Score file: {score_file_path}"
                        )
                    trial_info = task_trials[idx]
                    trial_info["matched_by_timestamp"] = file_timestamp
                    trial_found = True
                    break

            if not trial_found:
                available_timestamps = [
                    re.search(r"_(\d{8}_\d{6})\.json$", f.name).group(1)
                    for f in task_log_files
                ]
                raise ValueError(
                    f"Could not map file to trial by timestamp:\n"
                    f"  File: {file_path}\n"
                    f"  Timestamp: {file_timestamp}\n"
                    f"  Available timestamps: {available_timestamps}"
                )
        except Exception as e:
            logger.error(f"Error processing score file {score_file_path}: {e}")
            raise

    if file_path in files:
        logger.warning(f"Duplicate file found: {file_path}")
    else:
        files[file_path] = {
            "level": level,
            "model": model,
            "environment": environment,
            "agent_type": agent_type,
            "verbosity": verbosity,
            "task_id": task_id,
            "timestamp": file_timestamp,
            "report_exists": report_exists,
            "trial_info": trial_info,
        }

output_path = Path(__file__).parent / "data" / "all_files2_label.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w") as f:
    json.dump(files, f, indent=4)

logger.info(f"Processed {len(files)} files")
logger.info(
    f"Files with trial info: {sum(1 for f in files.values() if f['trial_info'] is not None)}"
)
logger.info(f"Output saved to: {output_path}")
