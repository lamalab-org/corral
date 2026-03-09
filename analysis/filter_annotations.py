"""
Script to filter annotations from corral_dump.json.
- Only keeps entries where annotator is "Nawaf" or "Martino"
- Drops entries where annotator is "Martino" and the file belongs to "catalyst" environment
- For duplicate fileIds, keeps the entry with the latest submittedAt
- Adds env, level, and model metadata based on accepted_questions folder structure
"""

# ruff: noqa: T201

import json
import re
from datetime import datetime
from pathlib import Path


def parse_date(date_obj):
    """Parse MongoDB date format to datetime object."""
    if isinstance(date_obj, dict) and "$date" in date_obj:
        date_str = date_obj["$date"]
        # Handle ISO format with Z suffix
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"
        return datetime.fromisoformat(date_str)
    return None


def build_file_metadata_map(accepted_questions_path):
    """
    Build a mapping from fileId to metadata (env, level, model).
    Structure: accepted_questions/<model>/<env>/<level>/tasks/<agent_folder>/<file.json>
    """
    file_metadata = {}

    for model_folder in ["claude_sonnet_45", "gpt-4o"]:
        model_path = accepted_questions_path / model_folder
        if not model_path.exists():
            continue

        # Iterate over env folders (afm, catalyst, md, ml, resistor, retrosynthesis, spectra, wetlab)
        for env_path in model_path.iterdir():
            if not env_path.is_dir():
                continue
            env = env_path.name

            # Iterate over level folders (level_1, level_2, level_3)
            for level_path in env_path.iterdir():
                if not level_path.is_dir():
                    continue
                level_match = re.match(r"level_(\d+)", level_path.name)
                if not level_match:
                    continue
                level = int(level_match.group(1))

                # Find all JSON files recursively
                for json_file in level_path.rglob("*.json"):
                    file_id = json_file.name
                    # Determine scaffold from path
                    path_str = str(json_file)
                    if "ReActAgent" in path_str:
                        scaffold = "react"
                    elif "ToolCallingAgent" in path_str:
                        scaffold = "tool_calling"
                    else:
                        raise ValueError(f"Unknown scaffold for file: {json_file}")
                    # If same file exists in multiple models, we might overwrite
                    # but typically we want to track which model it came from
                    if file_id not in file_metadata:
                        file_metadata[file_id] = {
                            "env": env,
                            "level": level,
                            "model": model_folder,
                            "scaffold": scaffold,
                        }

    return file_metadata


def get_catalyst_file_ids(accepted_questions_path):
    """Get all fileIds that belong to the catalyst environment."""
    catalyst_file_ids = set()

    # Search in both claude_sonnet_45 and gpt-4o folders
    for model_folder in ["claude_sonnet_45", "gpt-4o"]:
        catalyst_path = accepted_questions_path / model_folder / "catalyst"
        if catalyst_path.exists():
            # Recursively find all JSON files
            for json_file in catalyst_path.rglob("*.json"):
                catalyst_file_ids.add(json_file.name)

    return catalyst_file_ids


def main():
    # Define paths
    input_path = (
        Path(__file__).parent / "results" / "data" / "corral_annotations_dump.json"
    )
    output_path = (
        Path(__file__).parent / "results" / "data" / "filtered_annotations.json"
    )
    accepted_questions_path = Path(__file__).parent.parent / "accepted_questions"

    # Build file metadata map (env, level, model)
    file_metadata = build_file_metadata_map(accepted_questions_path)
    print(f"Built metadata for {len(file_metadata)} files from accepted_questions")

    # Get catalyst file IDs to exclude for Martino
    catalyst_file_ids = get_catalyst_file_ids(accepted_questions_path)
    print(f"Found {len(catalyst_file_ids)} catalyst files to exclude for Martino")

    # Load the data
    print(f"Loading data from {input_path}...")
    with input_path.open() as f:
        data = json.load(f)

    print(f"Total entries in original file: {len(data)}")

    # Filter by annotators (Nawaf or Martino)
    # Also exclude Martino + catalyst entries
    valid_annotators = {"Nawaf", "Martino"}
    filtered_entries = []
    catalyst_dropped = 0

    for entry in data:
        annotator = entry.get("annotator")
        if annotator not in valid_annotators:
            continue

        # If Martino and file is from catalyst, skip
        if annotator == "Martino" and entry.get("fileId") in catalyst_file_ids:
            catalyst_dropped += 1
            continue

        filtered_entries.append(entry)

    print(f"Entries by Nawaf or Martino: {len(filtered_entries)}")
    print(f"Martino catalyst entries dropped: {catalyst_dropped}")

    # Group by fileId and keep the one with latest submittedAt
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
            # Keep the one with the latest submittedAt
            # If no date available, prefer the new one if it has a date
            if entry_date is not None and (
                existing_date is None or entry_date > existing_date
            ):
                file_id_map[file_id] = {"entry": entry, "date": entry_date}

    # Extract the final entries and enrich with metadata
    # Only keep entries that have metadata (env, level, model)
    final_entries = []
    metadata_added = 0
    metadata_missing = 0

    for item in file_id_map.values():
        entry = item["entry"]
        file_id = entry.get("fileId")

        # Add env, level, model from metadata - skip if not found
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
            # Skip entries without metadata

    print(f"Unique fileIds after deduplication: {len(final_entries)}")
    print(
        f"Entries with metadata: {metadata_added}, dropped (no metadata): {metadata_missing}"
    )

    # Save to output file
    print(f"Saving filtered data to {output_path}...")
    with output_path.open("w") as f:
        json.dump(final_entries, f, indent=4)

    print("Done!")

    # Print summary of annotators in final data
    annotator_counts = {}
    for entry in final_entries:
        annotator = entry.get("annotator", "Unknown")
        annotator_counts[annotator] = annotator_counts.get(annotator, 0) + 1

    print("\nAnnotator breakdown in final data:")
    for annotator, count in sorted(annotator_counts.items()):
        print(f"  {annotator}: {count}")

    # Print summary of envs
    env_counts = {}
    for entry in final_entries:
        env = entry.get("env", "Unknown")
        env_counts[env] = env_counts.get(env, 0) + 1

    print("\nEnvironment breakdown in final data:")
    for env, count in sorted(env_counts.items(), key=lambda x: (x[0] is None, x[0])):
        print(f"  {env}: {count}")


if __name__ == "__main__":
    main()
