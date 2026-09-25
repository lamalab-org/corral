"""Level-specific blank submission examples, separate from evaluator policy."""

from __future__ import annotations

import json
from pathlib import Path

from corral.workspace import confine_workspace_path

EXAMPLE_DIRECTORY = "submission_examples"
TEMPLATE_ROOT = Path(__file__).with_name("submission_templates")


def load_example(task_number: int, *, level: int = 2) -> dict:
    """Read fresh objects so callers cannot mutate a later task's examples."""
    if type(task_number) is not int or task_number not in range(1, 11):
        raise ValueError("task_number must be an integer from 1 to 10")
    if type(level) is not int or level not in (1, 2):
        raise ValueError("level must be 1 or 2")
    root = TEMPLATE_ROOT / f"level_{level}"
    return json.loads((root / f"task_{task_number}.json").read_text())


def _json_block(value: object) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"


def example_files(task_number: int, *, level: int = 2) -> dict[str, str]:
    """Files copied into a task workspace."""
    example = load_example(task_number, level=level)
    templates = {"manifest.json": example["manifest"], **example["files"]}
    guide = (
        f"# Task {task_number}: submission examples\n\n"
        "These are blank templates, not completed evidence. Copy the files you use "
        "to your output directory, fill them with actual data, and update the manifest "
        "paths. Only link files used in your calculation; do not submit unused "
        "alternative examples. File names are illustrative; keep the artifact-role "
        "keys that identify their contents. All linked paths must stay in the workspace.\n\n"
        "## Files\n\n"
        + "\n".join(f"- [{name}]({name})" for name in templates)
        + "\n\n## Filling the templates\n\n"
        "Replace nulls, empty strings, placeholder keys, and example records with "
        "your actual results and paths. Include every file used by the calculation "
        "and list the code in manifest.scripts. Keep all paths in the task workspace.\n\n"
        "Use finite numbers and the units named by the template. NumPy arrays must "
        "load without pickle. Structure data may use ASE .traj, extended XYZ, or "
        "JSON frames with symbols, positions, cell, and pbc. Follow the task "
        "description for all scientific requirements.\n"
    )
    if level == 2:
        guide += (
            "\nAngle-bracketed type and shape placeholders describe the data to insert: "
            "replace <number[frames, 216, 3]> with a numeric array, not a string. "
            "Dimensions with the same name must agree across arrays; frame and atom "
            "orders must stay aligned. A one-record trajectory example shows the "
            "fields on EACH frame; repeat it for every retained state required by "
            "the task. ASE .traj and Extended XYZ may carry the same data: save time "
            "and stage IDs in frame info, momenta as atomic arrays, and energies/forces "
            "as stored calculator results.\n\n"
            "Files ending in .example.json are optional alternative layouts. Point "
            "the same manifest artifact role to whichever layout you use. Files "
            "ending in .layout.json describe array names and shapes inside a binary "
            "file; create the real numeric file and do not submit its layout as data. "
            "Resolve arrays_file and lammps_data paths relative to the referring JSON "
            "file, and also list every referenced file in manifest.artifacts. "
            "Linked LAMMPS data files must retain atom IDs, charges, and Velocities.\n"
        )
    return {
        "README.md": guide,
        **{
            name: json.dumps(value, indent=2, ensure_ascii=False) + "\n"
            for name, value in templates.items()
        },
    }


def example_prompt(task_number: int, *, workspace: bool, level: int = 2) -> str:
    """Show the manifest first; load supporting layouts on demand in a workspace."""
    example = load_example(task_number, level=level)
    prompt = "Blank manifest example:\n\n" + _json_block(example["manifest"]) + "\n"
    if workspace:
        return prompt + (
            f"\nBlank copies and linked-file templates are in /workspace/{EXAMPLE_DIRECTORY}/. "
            f"Read /workspace/{EXAMPLE_DIRECTORY}/README.md before writing your output files.\n"
        )
    # Unbound environments have no filesystem in which to expose the templates.
    files = example_files(task_number, level=level)
    prompt += "\n" + files["README.md"]
    for name, value in example["files"].items():
        prompt += f"\n### {name}\n\n" + _json_block(value) + "\n"
    return prompt


def seed_examples(workspace: str | Path, task_number: int, *, level: int = 2) -> None:
    """Seed before the initial snapshot, preserving any existing user edits."""
    root = Path(workspace)
    directory = confine_workspace_path(root, EXAMPLE_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in example_files(task_number, level=level).items():
        path = confine_workspace_path(root, directory / name)
        if not path.exists():
            path.write_text(content)
