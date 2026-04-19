"""Pre-commit hook to validate that environment task configs follow the standard convention.

Expected structure:
  tasks/<env>/environments/level_<N>/tasks_json/<task>.json
  tasks/<env>/environments/level_<N>/subtasks_json/<task>.json

Rules:
  1. Each environment must have an `environments/` directory with `level_<N>/` subdirs
  2. Each level must contain `tasks_json/` (required) and optionally `subtasks_json/`
  3. Only `tasks_json/` and `subtasks_json/` directories are allowed inside a level
  4. All files inside tasks_json/ and subtasks_json/ must be .json files
  5. Each .json file must contain valid JSON
"""

import json
import re
import sys
from pathlib import Path

TASKS_ROOT = Path("tasks")

SKIP_ENVS: set[str] = set()

LEVEL_PATTERN = re.compile(r"^level_\d+$")
ALLOWED_SUBDIRS = {"tasks_json", "subtasks_json"}


def check_environment(env_path: Path) -> list[str]:
    """Validate a single environment directory. Returns list of error messages."""
    errors = []
    env_name = env_path.name

    environments_dir = env_path / "environments"
    if not environments_dir.is_dir():
        errors.append(f"{env_name}: missing 'environments/' directory")
        return errors

    # Check that environments/ contains only level_N directories
    # (allow visuals_and_helpers or other non-level dirs to coexist)
    level_dirs = []
    for child in sorted(environments_dir.iterdir()):
        if not child.is_dir():
            continue
        if LEVEL_PATTERN.match(child.name):
            level_dirs.append(child)

    if not level_dirs:
        errors.append(f"{env_name}: no level_N directories found in environments/")
        return errors

    for level_dir in level_dirs:
        level_name = level_dir.name

        # Check subdirectories
        subdirs = {d.name for d in level_dir.iterdir() if d.is_dir()}
        invalid_subdirs = subdirs - ALLOWED_SUBDIRS
        if invalid_subdirs:
            errors.append(
                f"{env_name}/{level_name}: invalid subdirectories: {invalid_subdirs}. "
                f"Only {ALLOWED_SUBDIRS} are allowed."
            )

        if "tasks_json" not in subdirs:
            errors.append(
                f"{env_name}/{level_name}: missing required 'tasks_json/' directory"
            )

        # Validate JSON files in each subdir
        for subdir_name in ALLOWED_SUBDIRS:
            subdir = level_dir / subdir_name
            if not subdir.is_dir():
                continue

            json_files = list(subdir.iterdir())
            if not json_files:
                errors.append(
                    f"{env_name}/{level_name}/{subdir_name}: directory is empty"
                )
                continue

            for fpath in sorted(json_files):
                if fpath.is_dir():
                    errors.append(
                        f"{env_name}/{level_name}/{subdir_name}: "
                        f"unexpected subdirectory '{fpath.name}'"
                    )
                    continue

                if fpath.suffix != ".json":
                    errors.append(
                        f"{env_name}/{level_name}/{subdir_name}/{fpath.name}: "
                        f"not a .json file"
                    )
                    continue

                try:
                    with fpath.open() as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    errors.append(
                        f"{env_name}/{level_name}/{subdir_name}/{fpath.name}: "
                        f"invalid JSON: {e}"
                    )

    return errors


def main() -> int:
    if not TASKS_ROOT.is_dir():
        return 0

    all_errors = []
    for env_dir in sorted(TASKS_ROOT.iterdir()):
        if not env_dir.is_dir():
            continue
        if env_dir.name in SKIP_ENVS:
            continue
        if (env_dir / "environments").is_dir():
            all_errors.extend(check_environment(env_dir))

    if all_errors:
        sys.stdout.write("Environment convention check failed:\n")
        for error in all_errors:
            sys.stdout.write(f"  ✗ {error}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
