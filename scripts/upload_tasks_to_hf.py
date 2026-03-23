"""Upload all standardized task/subtask configs to HuggingFace as a dataset.

Repo: jablonkagroup/corral-environment-tasks
Subsets: {env}-level_{N}-task, {env}-level_{N}-subtask

Each row in a subset is one task or subtask entry (from the JSON arrays).

Usage:
    uv run python scripts/upload_tasks_to_hf.py          # dry-run (print summary)
    uv run python scripts/upload_tasks_to_hf.py --push    # push to HF
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

# Load HF token from .env file
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

HF_TOKEN = os.environ.get("HF_TOKEN")

TASKS_ROOT = Path("tasks")
HF_REPO = "jablonkagroup/corral-environment-tasks"
SKIP_ENVS = {"samplemath"}

SUBDIR_TO_TYPE = {
    "tasks_json": "task",
    "subtasks_json": "subtask",
}


def collect_subsets() -> dict[str, list[dict]]:
    """Walk all environments and collect entries grouped by subset name."""
    subsets: dict[str, list[dict]] = {}

    for env_dir in sorted(TASKS_ROOT.iterdir()):
        if not env_dir.is_dir() or env_dir.name in SKIP_ENVS:
            continue

        environments_dir = env_dir / "environments"
        if not environments_dir.is_dir():
            continue

        env_name = env_dir.name

        for level_dir in sorted(environments_dir.iterdir()):
            if not level_dir.is_dir() or not level_dir.name.startswith("level_"):
                continue

            level_name = level_dir.name

            for subdir_name, type_suffix in SUBDIR_TO_TYPE.items():
                subdir = level_dir / subdir_name
                if not subdir.is_dir():
                    continue

                subset_name = f"{env_name}-{level_name}-{type_suffix}"
                rows = []

                for fpath in sorted(subdir.glob("*.json")):
                    with fpath.open() as f:
                        entries = json.load(f)

                    if not isinstance(entries, list):
                        entries = [entries]

                    for entry in entries:
                        row = {
                            "uuid": entry.get("uuid", ""),
                            "id": entry.get("id", ""),
                            "name": entry.get("name", ""),
                            "description": entry.get("description", ""),
                            "tools": json.dumps(entry.get("tools", [])),
                            "scoring_function": str(entry.get("scoring_function", "")),
                            "submission_format": entry.get("submission_format", ""),
                            "environment": env_name,
                            "level": level_name,
                            "type": type_suffix,
                            "source_file": str(fpath.relative_to(TASKS_ROOT)),
                            "full_json": json.dumps(entry),
                        }
                        rows.append(row)

                if rows:
                    subsets[subset_name] = rows

    return subsets


def rows_to_parquet(rows: list[dict], path: Path) -> None:
    """Convert rows to a parquet file."""
    columns = {}
    for key in rows[0]:
        columns[key] = [r[key] for r in rows]
    table = pa.table(columns)
    pq.write_table(table, path)


def main() -> int:
    push = "--push" in sys.argv

    subsets = collect_subsets()

    sys.stdout.write(f"Collected {len(subsets)} subsets:\n")
    total_rows = 0
    for name, rows in sorted(subsets.items()):
        sys.stdout.write(f"  {name}: {len(rows)} entries\n")
        total_rows += len(rows)
    sys.stdout.write(f"  Total: {total_rows} entries\n\n")

    if not push:
        sys.stdout.write("Dry run. Use --push to upload to HuggingFace.\n")
        return 0

    if not HF_TOKEN:
        sys.stdout.write("Error: HF_TOKEN not found. Set it in .env or environment.\n")
        return 1

    api = HfApi(token=HF_TOKEN)

    api.create_repo(
        repo_id=HF_REPO,
        repo_type="dataset",
        exist_ok=True,
    )

    # Upload a clean README first
    readme = (
        "---\nlicense: apache-2.0\n---\n"
        "# CORRAL Environment Tasks\n\n"
        "Standardized task and subtask configurations for the CORRAL benchmark.\n\n"
        "Each subset follows the naming pattern: "
        "`{environment}-level_{N}-{task|subtask}`\n"
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="dataset",
    )

    # Upload each subset as a parquet file under the HF dataset layout
    with tempfile.TemporaryDirectory() as tmpdir:
        for subset_name, rows in sorted(subsets.items()):
            parquet_path = Path(tmpdir) / f"{subset_name}.parquet"
            rows_to_parquet(rows, parquet_path)

            # HF datasets convention: {config_name}/train-00000-of-00001.parquet
            remote_path = f"{subset_name}/train-00000-of-00001.parquet"

            sys.stdout.write(f"Uploading '{subset_name}' ({len(rows)} rows)... ")
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=remote_path,
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            sys.stdout.write("done.\n")

    sys.stdout.write(
        f"\nAll subsets uploaded to https://huggingface.co/datasets/{HF_REPO}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
