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
        "\nBefore final submission, call validate_submission with your completed "
        "manifest path, for example /workspace/output/manifest.json. It returns "
        "format errors with file or field locations and suggested repairs. Fix "
        "the errors and call it again; review warnings about evidence that was "
        "not checked. A valid format does not establish scientific correctness.\n"
    )
    if task_number >= 3:
        guide += (
            "\nRecord calculator options in settings.model_settings, for example "
            '{"default_dtype": "float64", "device": "cuda", "dispersion": false} '
            "using the values actually used. settings.calculator may be a class-name "
            "string such as mace.calculators.MACECalculator. Top-level options and "
            "calculator or calculator_args objects are also accepted; repeated "
            "options must agree. Precision aliases are dtype, precision, and "
            "model_dtype.\n"
        )
    if level == 2 and task_number == 3:
        guide += (
            "For the separate teacher, student, and MD calculators, use "
            "teacher_model_settings, student_model_settings, or md_model_settings; "
            "these describe separate calculators and do not inherit generic options.\n"
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
    if task_number == 4:
        guide += (
            "\nRecord force-constant corrections in the order applied. Supported names "
            "include cubic_symmetry (all 48 FCC cubic point-group operations), "
            "pair_symmetry, and acoustic_sum_rule. Retain the raw displaced structures "
            "and forces so the derivative and each correction can be reconstructed.\n"
        )
    if task_number == 6:
        guide += (
            "\nUse equilibration or nvt as the thermal-trace stage label for equilibration. "
            "Use production or nve for an NVE production stage when requested. "
            "Save the initialized state and the equilibration endpoint with momenta. "
            "The thermal trace must include step, time_fs, temperature_K, "
            "density_g_cm3, and measured pressure with units in the column name "
            "(pressure_GPa, pressure_bar, or pressure_atm).\n"
        )
    if task_number == 7:
        guide += (
            "\nSave time_fs and stage in every trajectory frame. The stages.json production "
            "interval names frame indices [start_index, exclusive_end_index], not MD "
            "step numbers. Retain the stage's initial state and its final restartable "
            "state, and distinguish an extra boundary frame from production samples.\n"
        )
    if task_number == 8:
        guide += (
            "\nFor reproducible training distortions, start a fresh NumPy default_rng(seed) "
            "or Generator(PCG64(seed)) and record its exact name, library, and seed in "
            "generation.train. Draw one normal array of shape (64, 3) for each geometry "
            "in generation_index order, with the requested sigma, before using this "
            "generator for other random draws. Keep generation_index and row IDs aligned "
            "with the saved structures.\n"
        )
    if task_number == 9:
        guide += (
            "\nFor every requested run, save its initial state before integration as the "
            "first equilibration frame, then the equilibration endpoint. Save a boundary "
            "state shared with production, all 200 chronological production frames, and "
            "their cumulative time_fs and momenta. Opening an ASE Trajectory in write "
            "mode does not write a frame: call write(atoms) before closing it.\n"
        )
    if task_number == 10:
        guide += (
            "\nA stage starts before its equilibration. Keep stage start/end times distinct "
            "from production_start_time_fs and production_end_time_fs. Save the full "
            "thermal trace from stage start through stage end; label each row and frame "
            "with its stage ID and cumulative time_fs.\n"
        )
        if level == 2:
            guide += (
                "Save all eight stages with distinct IDs, including the two 300 K stages. "
                "boundary_states must contain each stage's start and end, in order "
                "(16 frames). Save both production-window endpoints for each stage.\n"
                "When trusted execution is required, call run_verified_md with a JSON "
                "configuration containing random_seed (an integer); optional timestep_fs "
                "(default 1), friction_fs (default 0.01), and default_dtype (float64 or "
                "float32); and stages in target order 300, 400, 500, 600, 700, 800, 900, "
                "300 K. Each stage has exactly id, target_temperature_K, "
                "equilibration_steps, production_steps, sample_interval, and "
                "equilibration_interval. Use positive integer step and sampling counts; "
                "sampling intervals must divide their stage durations, and retain at "
                "least two equilibration samples. There may be at most 1,000,000 total "
                "steps. This tool runs the simulation and saves restartable states, "
                "trajectories, settings, thermal traces, and a verification receipt. "
                "Use its generated artifacts without rewriting them, preserve the "
                "returned IDs as manifest.provenance.run_id, "
                "manifest.provenance.action_id, and optionally "
                "manifest.provenance.release_id. Top-level run_id, action_id, "
                "and release_id are also accepted; duplicate IDs must agree. "
                "Replace the template's null IDs with the receipt strings. If "
                "trusted execution is not required and was not used, omit the "
                "provenance object. Then compute the requested analysis and "
                "results from those artifacts.\n"
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
