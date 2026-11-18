"""Filter exported annotation records into the analysis-ready dataset.

This script loads the raw annotation dump created from MongoDB, keeps only the
annotators used in the study, removes catalyst annotations from Martino,
deduplicates submissions by choosing the latest record for each `fileId`, and
enriches the surviving entries with environment, level, model, and scaffold
metadata derived from the accepted question directories.
"""

import json
import re
from datetime import datetime
from pathlib import Path

from loguru import logger


def parse_date(date_obj: dict | None) -> datetime | None:
    """Parse a MongoDB Extended JSON date value.

    Args:
        date_obj: Date payload from the exported annotation dump.

    Returns:
        A timezone-aware `datetime` when the payload contains a `$date`
        field, otherwise `None`.
    """
    if isinstance(date_obj, dict) and "$date" in date_obj:
        date_str = date_obj["$date"]
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"
        return datetime.fromisoformat(date_str)
    return None


def build_file_metadata_map(
    accepted_questions_path: Path,
) -> dict[str, dict[str, str | int]]:
    """Index accepted question files by their exported `fileId`.

    Args:
        accepted_questions_path: Root directory containing accepted question
            data grouped by model, environment, and level.

    Returns:
        A mapping from file name to metadata describing the originating
        environment, task level, model, and scaffold.
    """
    file_metadata = {}

    for model_folder in ["claude_sonnet_45", "gpt-4o"]:
        model_path = accepted_questions_path / model_folder
        if not model_path.exists():
            continue

        for env_path in model_path.iterdir():
            if not env_path.is_dir():
                continue
            env = env_path.name

            for level_path in env_path.iterdir():
                if not level_path.is_dir():
                    continue
                level_match = re.match(r"level_(\d+)", level_path.name)
                if not level_match:
                    continue
                level = int(level_match.group(1))

                for json_file in level_path.rglob("*.json"):
                    file_id = json_file.name
                    path_str = str(json_file)
                    if "ReActAgent" in path_str:
                        scaffold = "react"
                    elif "ToolCallingAgent" in path_str:
                        scaffold = "tool_calling"
                    else:
                        raise ValueError(f"Unknown scaffold for file: {json_file}")
                    if file_id not in file_metadata:
                        file_metadata[file_id] = {
                            "env": env,
                            "level": level,
                            "model": model_folder,
                            "scaffold": scaffold,
                        }

    return file_metadata


def get_catalyst_file_ids(accepted_questions_path: Path) -> set[str]:
    """Collect file identifiers associated with catalyst tasks.

    Args:
        accepted_questions_path: Root directory containing accepted question
            data grouped by model and environment.

    Returns:
        The set of JSON file names that appear under the catalyst environment.
    """
    catalyst_file_ids = set()

    for model_folder in ["claude_sonnet_45", "gpt-4o"]:
        catalyst_path = accepted_questions_path / model_folder / "catalyst"
        if catalyst_path.exists():
            for json_file in catalyst_path.rglob("*.json"):
                catalyst_file_ids.add(json_file.name)

    return catalyst_file_ids


def main() -> None:
    """Create the filtered annotation dataset used by downstream analysis.

    Returns:
        None.
    """
    input_path = (
        Path(__file__).parent / "results" / "data" / "corral_annotations_dump.json"
    )
    output_path = (
        Path(__file__).parent / "results" / "data" / "filtered_annotations.json"
    )
    accepted_questions_path = Path(__file__).parent.parent / "questions_to_annotate"
    logger.info(f"Accepted questions path: {accepted_questions_path}")

    file_metadata = build_file_metadata_map(accepted_questions_path)
    logger.info(
        f"Built metadata for {len(file_metadata)} files from questions_to_annotate"
    )

    catalyst_file_ids = get_catalyst_file_ids(accepted_questions_path)
    logger.info(f"Found {len(catalyst_file_ids)} catalyst files to exclude for Martino")

    logger.info(f"Loading data from {input_path}...")
    with input_path.open() as f:
        data = json.load(f)

    logger.info(f"Total entries in original file: {len(data)}")

    # Filter by annotators (Nawaf or Martino)
    # Also exclude Martino + catalyst entries
    valid_annotators = {"Nawaf", "Martino"}
    filtered_entries = []
    catalyst_dropped = 0

    for entry in data:
        annotator = entry.get("annotator")
        if annotator not in valid_annotators:
            continue

        if annotator == "Martino" and entry.get("fileId") in catalyst_file_ids:
            catalyst_dropped += 1
            continue

        filtered_entries.append(entry)

    logger.info(f"Entries by Nawaf or Martino: {len(filtered_entries)}")
    logger.info(f"Martino catalyst entries dropped: {catalyst_dropped}")

    file_id_map = {}

    for entry in filtered_entries:
        file_id = entry.get("fileId")
        if not file_id:
            continue

        submitted_at = entry.get("submittedAt")
        entry_date = parse_date(submitted_at) if submitted_at else None

        if file_id not in file_id_map:
            file_id_map[file_id] = {"entry": entry, "date": entry_date}
        else:
            existing_date = file_id_map[file_id]["date"]
            if entry_date is not None and (
                existing_date is None or entry_date > existing_date
            ):
                file_id_map[file_id] = {"entry": entry, "date": entry_date}

    final_entries = []
    metadata_added = 0
    metadata_missing = 0

    for item in file_id_map.values():
        entry = item["entry"]
        file_id = entry.get("fileId")

        if file_id in file_metadata:
            meta = file_metadata[file_id]
            entry["env"] = meta["env"]
            entry["level"] = meta["level"]
            entry["model"] = meta["model"]
            entry["scaffold"] = meta["scaffold"]
            metadata_added += 1
            final_entries.append(entry)
        else:
            metadata_missing += 1

    logger.info(f"Unique fileIds after deduplication: {len(final_entries)}")
    logger.info(
        f"Entries with metadata: {metadata_added}, dropped (no metadata): {metadata_missing}"
    )

    logger.info(f"Saving filtered data to {output_path}...")
    with output_path.open("w") as f:
        json.dump(final_entries, f, indent=4)

    logger.info("Done!")

    annotator_counts = {}
    for entry in final_entries:
        annotator = entry.get("annotator", "Unknown")
        annotator_counts[annotator] = annotator_counts.get(annotator, 0) + 1

    logger.info("\nAnnotator breakdown in final data:")
    for annotator, count in sorted(annotator_counts.items()):
        logger.info(f"  {annotator}: {count}")

    env_counts = {}
    for entry in final_entries:
        env = entry.get("env", "Unknown")
        env_counts[env] = env_counts.get(env, 0) + 1

    logger.info("\nEnvironment breakdown in final data:")
    for env, count in sorted(env_counts.items(), key=lambda x: (x[0] is None, x[0])):
        logger.info(f"  {env}: {count}")


if __name__ == "__main__":
    main()
