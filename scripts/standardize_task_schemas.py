"""Standardize all task/subtask JSON files to a common schema.

Transforms:
  - Dict-based files → Array-based (top-level always a list)
  - Dict keys become `id` field on each entry
  - `scoring_fn` → `scoring_function`
  - Group A (spectra, retrosynthesis): adds `description` from `input.prompt`
  - Adds a `uuid` to every task/subtask entry
"""

import json
import sys
import uuid
from pathlib import Path

TASKS_ROOT = Path("tasks")
SKIP_ENVS = {"samplemath"}

# Group A environments use array format with scoring_fn and input.prompt
GROUP_A_ENVS = {"spectra_elucidation", "retrosynthesis"}


def generate_uuid() -> str:
    return str(uuid.uuid4())


def standardize_group_a(entries: list[dict]) -> list[dict]:
    """Standardize Group A (spectra, retrosynthesis) entries."""
    result = []
    for entry in entries:
        e = dict(entry)

        # Add uuid if missing
        if "uuid" not in e:
            e["uuid"] = generate_uuid()

        # Add description from input.prompt if missing
        if "description" not in e and "input" in e and isinstance(e["input"], dict):
            e["description"] = e["input"].get("prompt", "")

        # Rename scoring_fn → scoring_function
        if "scoring_fn" in e and "scoring_function" not in e:
            e["scoring_function"] = e.pop("scoring_fn")

        result.append(e)
    return result


def standardize_group_b(data: dict | list) -> list[dict]:
    """Standardize Group B (catalyst, ml, resistor, corral_md, afm) entries.

    Converts dict-based format to array-based.
    """
    if isinstance(data, list):
        # Already array format
        entries = data
    else:
        # Dict format: {"task_key": {task_data}, ...} → [{...with id}, ...]
        entries = []
        for key, value in data.items():
            e = dict(value)
            if "id" not in e:
                e["id"] = key
            entries.append(e)

    result = []
    for entry in entries:
        e = dict(entry)
        if "uuid" not in e:
            e["uuid"] = generate_uuid()
        result.append(e)
    return result


def process_file(fpath: Path, env_name: str) -> list[dict]:
    """Load and standardize a single JSON file."""
    with fpath.open() as f:
        data = json.load(f)

    if env_name in GROUP_A_ENVS:
        if not isinstance(data, list):
            # Shouldn't happen for Group A, but handle gracefully
            data = [data]
        return standardize_group_a(data)
    else:
        return standardize_group_b(data)


def main() -> int:
    if not TASKS_ROOT.is_dir():
        sys.stdout.write("tasks/ directory not found\n")
        return 1

    changed = 0
    for env_dir in sorted(TASKS_ROOT.iterdir()):
        if not env_dir.is_dir() or env_dir.name in SKIP_ENVS:
            continue

        environments_dir = env_dir / "environments"
        if not environments_dir.is_dir():
            continue

        for level_dir in sorted(environments_dir.iterdir()):
            if not level_dir.is_dir() or not level_dir.name.startswith("level_"):
                continue

            for subdir_name in ["tasks_json", "subtasks_json"]:
                subdir = level_dir / subdir_name
                if not subdir.is_dir():
                    continue

                for fpath in sorted(subdir.glob("*.json")):
                    with fpath.open() as f:
                        original = f.read()

                    entries = process_file(fpath, env_dir.name)
                    new_content = json.dumps(entries, indent=2) + "\n"

                    if new_content != original:
                        with fpath.open("w") as f:
                            f.write(new_content)
                        changed += 1
                        sys.stdout.write(f"  updated {fpath.relative_to(TASKS_ROOT)}\n")

    sys.stdout.write(f"\n{changed} files updated.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
