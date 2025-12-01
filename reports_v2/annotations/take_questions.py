"""
It randomly samples question files from different models and tasks, ensuring a balanced representation.
It then copies the selected files to a designated directory while preserving their directory structure.
"""

import os
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pymongo import MongoClient  # <- new

load_dotenv(".env", override=True)

LIMIT = 160

# ---- MongoDB config (adjust to your environment) ----
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "Corral")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "First-traces")


def file_id_from_path(path: Path) -> str:
    """
    Map a question file path to the MongoDB fileId.
    Currently: use the file's stem (filename without extension).

    If your fileId is something else (e.g., stored inside the JSON),
    change this function accordingly.
    """
    return path.stem


def take_questions(env) -> list[Path]:
    root_path = Path(__file__).parent.parent.parent / "reports_v2"

    # Dictionary to store model -> task_id -> list of file paths
    model_task_files_dict = {}

    # Collect files from all models
    for model in root_path.iterdir():
        if "claude" not in model.name and "gpt" not in model.name:
            continue
        if model.is_dir():
            env_path = model / env
            if not env_path.exists():
                continue

            # Dictionary to store task_id -> list of file paths for this model
            task_files_dict = {}

            for level in env_path.iterdir():
                if level.is_dir():
                    task_path = level / "tasks"
                    if task_path.exists():
                        for logs_dir in task_path.iterdir():
                            if "workflow" not in logs_dir.name:
                                continue
                            if logs_dir.is_dir():
                                for json_file in logs_dir.glob("*.json"):
                                    task_id_parts = json_file.stem.split("_")[:-2]
                                    task_id = "_".join(task_id_parts)

                                    # Add file to the task's list
                                    if task_id not in task_files_dict:
                                        task_files_dict[task_id] = []
                                    task_files_dict[task_id].append(json_file)

            if task_files_dict:
                model_task_files_dict[model.name] = task_files_dict

    if not model_task_files_dict:
        return []

    # Hierarchical sampling with round-robin across models
    accepted_tasks: list[Path] = []

    # Determine max files per task across all models
    max_files_per_task = 0
    for task_files_dict in model_task_files_dict.values():
        if task_files_dict:
            max_files = max(len(files) for files in task_files_dict.values())
            max_files_per_task = max(max_files_per_task, max_files)

    # Sample round-robin: for each file index, go through all models and all tasks
    for file_index in range(max_files_per_task):
        if len(accepted_tasks) >= LIMIT:
            break

        for model_name in sorted(model_task_files_dict.keys()):  # Sort for consistency
            if len(accepted_tasks) >= LIMIT:
                break

            task_files_dict = model_task_files_dict[model_name]

            for files in task_files_dict.values():
                if len(accepted_tasks) >= LIMIT:
                    break

                # Get files not yet selected for this task
                remaining_files = [f for f in files if f not in accepted_tasks]

                if remaining_files and file_index < len(files):
                    # For the first iteration, sample randomly
                    # For subsequent iterations, sample from remaining files
                    if file_index == 0:
                        sampled_file = random.choice(files)
                    else:
                        sampled_file = random.choice(remaining_files)

                    if sampled_file not in accepted_tasks:
                        accepted_tasks.append(sampled_file)

    return accepted_tasks


def copy_questions():
    envs = [
        # "catalyst",
        # "md",
        # "ml",
        # "resistor",
        # "retrosynthesis",
        "afm"
    ]
    all_accepted_tasks: list[Path] = []
    for env in envs:
        accepted_tasks = take_questions(env)
        all_accepted_tasks.extend(accepted_tasks)

        root_path = Path(__file__).parent
        reports_v2_path = root_path / "reports_v2"
        dest_base_path = root_path / "accepted_questions"

        for question in accepted_tasks:
            # Get the relative path from reports_v2
            relative_path = question.relative_to(reports_v2_path)
            # Create the destination path preserving the structure
            dest_file = dest_base_path / relative_path
            # Create parent directories if they don't exist
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            # Copy the file
            with question.open("r") as src, dest_file.open("w") as dst:
                dst.write(src.read())

    return all_accepted_tasks


# ---- New: MongoDB integration ----
def get_annotator_file_map(accepted_tasks: list[Path]) -> dict[str, list[Path]]:
    """
    Given the list of accepted task files, query MongoDB and return
    annotator -> list of Paths (those accepted files that have annotations).
    """
    if not accepted_tasks:
        return {}

    # Map fileId -> Path for quick lookup
    fileid_to_path: dict[str, Path] = {}
    for p in accepted_tasks:
        fid = file_id_from_path(p) + ".json"
        fileid_to_path[fid] = p

    file_ids = list(fileid_to_path.keys())

    client = MongoClient(MONGO_URI)
    coll = client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]

    # Find all documents for these fileIds
    cursor = coll.find({"fileId": {"$in": file_ids}})

    annotator_to_paths: dict[str, set[Path]] = defaultdict(set)

    for doc in cursor:
        annotator = doc.get("annotator", "UNKNOWN")
        fid = doc.get("fileId")
        if not fid:
            continue
        path = fileid_to_path.get(fid)
        if path:
            annotator_to_paths[annotator].add(path)

    client.close()

    # Convert sets to sorted lists for determinism
    return {
        annotator: sorted(paths, key=lambda p: str(p.stem + ".json"))
        for annotator, paths in annotator_to_paths.items()
    }


def main():
    accepted_tasks = copy_questions()
    if accepted_tasks:
        logger.info("Copied accepted tasks.")

        # Query MongoDB for annotations on the accepted files
        annotator_file_map = get_annotator_file_map(accepted_tasks)

        if not annotator_file_map:
            logger.info("No annotations found in MongoDB for accepted files.")
        else:
            logger.info("Annotations per annotator for accepted files:")
            for annotator, paths in annotator_file_map.items():
                logger.info(f"Annotator '{annotator}' labeled {len(paths)} file(s):")
                for p in paths:
                    parts = p.parts[-6:]  # Show last 5 parts for brevity
                    logger.info(f"  - {'/'.join(parts)}")
    else:
        logger.info("No accepted tasks found.")


if __name__ == "__main__":
    main()
