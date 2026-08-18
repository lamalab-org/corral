"""Unified task loader for CORRAL environments.

Loads standardized task/subtask configs from either:
  - HuggingFace: jablonkagroup/corral-environment-tasks (subset per env-level-type)
  - Local JSON directory: tasks/<env>/environments/level_<N>/{tasks_json,subtasks_json}/

All JSON files follow the standardized array schema:
  [{uuid, id, name, description, tools, scoring_function, submission_format, ...}]

Usage:
    from corral.utils.task_loader import load_task_entries

    # From HuggingFace
    entries = load_task_entries(environment="catalyst", level=1, task_type="task")

    # From local directory
    entries = load_task_entries(local_dir="tasks/catalyst/environments/level_1/tasks_json")
"""

import json
from pathlib import Path

from corral.logging import logger

HF_REPO = "jablonkagroup/corral-environment-tasks"


def load_task_entries(
    *,
    environment: str | None = None,
    level: int = 1,
    task_type: str = "task",
    local_dir: str | Path | None = None,
) -> list[dict]:
    """Load standardized task entries from HuggingFace or a local directory.

    Exactly one source must be specified: either (environment + level + task_type)
    for HuggingFace, or local_dir for a local JSON directory.

    Args:
        environment: Environment name (e.g. "catalyst", "spectra_elucidation").
        level: Level number (default 1).
        task_type: "task" or "subtask" (default "task").
        local_dir: Path to a local directory containing JSON files.

    Returns:
        List of task entry dicts with standardized fields.
    """
    if local_dir is not None:
        return _load_from_local(Path(local_dir))

    if environment is not None:
        return _load_from_hf(environment, level, task_type)

    raise ValueError(
        "Specify either 'environment' (for HF) or 'local_dir' (for local)."
    )


def load_task_entries_from_env_package(
    env_package_path: str | Path,
    level: int = 1,
    subtask: bool = False,
) -> list[dict]:
    """Load task entries using the standard directory layout relative to an env package.

    Resolves: <env_package>/../../environments/level_{N}/{tasks_json|subtasks_json}/

    Args:
        env_package_path: Path to the environment's Python package directory
                          (e.g. Path(__file__).parent for an env.py).
        level: Level number.
        subtask: If True, load from subtasks_json instead of tasks_json.

    Returns:
        List of task entry dicts.
    """
    pkg = Path(env_package_path).resolve()
    subdir = "subtasks_json" if subtask else "tasks_json"
    json_dir = pkg.parent.parent / "environments" / f"level_{level}" / subdir

    if not json_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {json_dir}")

    return _load_from_local(json_dir)


def _load_from_local(directory: Path) -> list[dict]:
    """Load all JSON files from a local directory and return flattened entries."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    entries: list[dict] = []
    for fpath in sorted(directory.glob("*.json")):
        with fpath.open() as f:
            data = json.load(f)

        if isinstance(data, list):
            entries.extend(data)
        else:
            entries.append(data)

    logger.info(f"Loaded {len(entries)} entries from {directory}")
    return entries


def _load_from_hf(environment: str, level: int, task_type: str) -> list[dict]:
    """Load task entries from HuggingFace dataset subset."""
    subset = f"{environment}-level_{level}-{task_type}"

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install 'datasets' to load from HuggingFace: pip install datasets"
        ) from e

    logger.info(f"Loading subset '{subset}' from {HF_REPO}")
    ds = load_dataset(HF_REPO, name=subset, split="train")

    entries = []
    for row in ds:
        if "full_json" in row:
            entries.append(json.loads(row["full_json"]))
        else:
            entries.append(dict(row))

    logger.info(f"Loaded {len(entries)} entries from HF subset '{subset}'")
    return entries
