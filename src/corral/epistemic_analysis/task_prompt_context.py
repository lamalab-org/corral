"""Render task instructions without constructing a workspace or loading scorers."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

ENVIRONMENT_FILES = {
    "psychometrics": "src/corral_psychometrics/env.py",
    "spectra_elucidation": "spectra_elucidation/env.py",
    "wetlab": "wetlab/env.py",
    "corral_md": "src/corral_md/env.py",
    "resistor_network": "src/resistor_network/env.py",
    "retrosynthesis": "retrosynthesis/env.py",
}
RENDERERS = {
    "psychometrics": "_task_prompt",
    "spectra_elucidation": "_spectra_prompt",
    "wetlab": "get_task_prompt",
    "corral_md": "_md_task_prompt",
    "resistor_network": "_default_task_prompt",
    "retrosynthesis": "_retro_prompt",
}


def _read_source(path, reference=None):
    raw = path.read_bytes()
    signature = hashlib.sha256(raw).hexdigest()
    if reference is not None and signature != reference["sha256"]:
        raise ValueError(f"Frozen task configuration has changed: {path}")
    return raw.decode(), {
        **(reference or {}),
        "path": str(path.resolve()),
        "sha256": signature,
    }


def _renderer(source, filename, name):
    """Load only the repository's pure prompt function and literal constants.

    Importing an environment also loads its scorers, simulators and data stores.
    The prompt functions need only task attributes, so extract the actual function
    instead of duplicating its template or executing environment setup.
    """
    tree = ast.parse(source, filename=filename)
    functions = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    if len(functions) != 1:
        raise ValueError(f"Expected one task prompt renderer {name} in {filename}")
    namespace = {"event": lambda *_args, **_kwargs: None}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.isupper():
                with suppress(ValueError, TypeError):
                    namespace[target.id] = ast.literal_eval(node.value)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            functions[0],
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[name]


def task_prompt(source, row, tasks_root, core_environment):
    """Use the pinned task source when present, otherwise the local tasks tree."""
    environment = row["environment"]
    if environment not in RENDERERS:
        raise ValueError(f"No task prompt renderer configured for {environment}")
    refs = row.get("validation_tool_inventory", {}).get("source_evidence", [])
    expected = f"tasks/{environment}/environments/level_{row['level']}/tasks_json/"
    task_refs = [r for r in refs if expected in r["path"]]
    provenance = []
    if task_refs:
        if len(task_refs) != 1:
            raise ValueError("Ambiguous task definition provenance")
        ref = task_refs[0]
        relative = Path(ref["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Expected a relative frozen configuration path")
        if row.get("pr_head_sha") and ref.get("git_commit") != row["pr_head_sha"]:
            raise ValueError("Task configuration differs from pinned report revision")
        text, evidence = _read_source(source.parent / relative, ref)
        tasks = [t for t in json.loads(text) if t["id"] == row["task_id"]]
        provenance.append(evidence)
        env_relative = f"tasks/{environment}/{ENVIRONMENT_FILES[environment]}"
        env_refs = [r for r in refs if r["path"].endswith(env_relative)]
        if len(env_refs) != 1:
            raise ValueError("Pinned task definition lacks its prompt renderer source")
        env_ref = env_refs[0]
        if env_ref.get("git_commit") != ref.get("git_commit"):
            raise ValueError("Task definition and prompt renderer revisions differ")
        renderer_text, renderer_ref = _read_source(
            source.parent / env_ref["path"], env_ref
        )
    else:
        directory = (
            tasks_root
            / environment
            / "environments"
            / f"level_{row['level']}"
            / "tasks_json"
        )
        tasks = []
        for path in sorted(directory.glob("*.json")):
            text, evidence = _read_source(path)
            matches = [t for t in json.loads(text) if t["id"] == row["task_id"]]
            if matches:
                tasks.extend(matches)
                provenance.append(evidence)
        renderer_text, renderer_ref = _read_source(
            tasks_root / environment / ENVIRONMENT_FILES[environment]
        )
    if len(tasks) != 1:
        raise ValueError(f"Expected one definition of task {row['task_id']}")
    task = tasks[0]
    dependencies = task.get("input_from_tasks") or task.get("input", {}).get(
        "input_from_task"
    )
    if dependencies:
        raise ValueError("Chained task prompts require the recorded dependency inputs")
    # Follow the task loaders' distinction between visible input.prompt and hidden
    # input.samples/output.target/scoring_params. Never pass hidden fields to code.
    description = (
        task["input"]["prompt"]
        if environment in {"wetlab", "spectra_elucidation", "retrosynthesis"}
        else task["description"]
    )
    initial = task.get("initial_input", {})
    if environment == "psychometrics":
        initial = {"public_inputs": initial.get("public_inputs", [])}
    elif environment == "spectra_elucidation" and "initial_input" not in task:
        initial = {"work_dir": "/workspace"}
    task_number = re.search(r"_task_(\d+)$", row["task_id"])
    public_task = SimpleNamespace(
        description=description,
        submission_format=task.get("submission_format", ""),
        initial_input=initial,
        input_map={},
        scoring_fn=SimpleNamespace(
            task_number=int(task_number[1]) if task_number else None
        ),
    )
    env = SimpleNamespace(
        current_task=public_task,
        task_id=row["task_id"],
        workspace_path="/workspace",
        group_tasks={},
        resolve_inputs=lambda _state: {},
        normalize_public_paths=lambda prompt: prompt,
    )
    if environment in {"corral_md", "resistor_network"}:
        core_text, core_ref = _read_source(core_environment)
        default = _renderer(core_text, core_ref["path"], "_default_task_prompt")
        env._default_task_prompt = lambda state: default(env, state)
        provenance.append(core_ref)
        if environment == "resistor_network":
            provenance.append(renderer_ref)
            return default(env, None), provenance
    renderer = _renderer(renderer_text, renderer_ref["path"], RENDERERS[environment])
    prompt = renderer(env, None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Task renderer returned an empty or non-text prompt")
    provenance.append(renderer_ref)
    return prompt, provenance
