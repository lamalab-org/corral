"""Public format validation must help repair submissions without grading them."""

import json
from contextlib import nullcontext

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from corral_md.env import MolecularDynamicsEnvironment, create_environments
from corral_md.score import WorkflowScorer
from corral_md.submission_examples import example_files, load_example
from corral_md.submission_validation import (
    build_validate_submission_tool,
    validate_submission,
)

from corral.core.action import Action
from corral.core.environment import Toolset
from corral.core.state import (
    ActionState,
    EnvironmentState,
    ExecutionState,
    RuntimeState,
    TaskState,
)
from corral.core.task import TaskDefinition
from corral.core.tool import WorkspaceAccess
from corral.core.transition import execute_action
from corral.runtime import permissions


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return name


def structure(root, name="frames.traj", count=1):
    atoms = Atoms("Ag2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.set_momenta(np.zeros((2, 3)))
    atoms.calc = SinglePointCalculator(atoms, energy=123.0, forces=np.zeros((2, 3)))
    write(root / name, [atoms] * count)
    return name


def submission(root, number=3):
    (root / "run.py").write_text("raise RuntimeError('must never execute')\n")
    settings = load_example(number)["files"]["settings.json"]
    if number == 3:
        settings = {
            "teacher_model": "teacher.model",
            "dispersion": True,
            "energy_unit": "eV",
            "force_unit": "eV/Angstrom",
        }
        artifacts = {"dataset": structure(root)}
    else:
        artifacts = {"extra": save(root, "data.json", {"value": 1})}
    manifest = {
        "settings": save(root, "settings.json", settings),
        "artifacts": artifacts,
        "scripts": ["run.py"],
    }
    save(root, "manifest.json", manifest)
    return manifest, settings


def validate(root, number=3, level=1):
    return validate_submission("/workspace/manifest.json", root, number, level=level)


def codes(report):
    return {item["code"] for item in report["errors"]}


def test_valid_format_is_read_only_and_does_not_grade_or_run_scripts(
    tmp_path, monkeypatch
):
    submission(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(
        WorkflowScorer, "evaluate", lambda *_a, **_k: pytest.fail("must not grade")
    )
    report = validate(tmp_path)
    assert report["valid"], report
    assert "score" not in report
    # Deliberately unphysical energy and incomplete scientific dataset above:
    # format validity must not become a scientific pass/fail oracle.
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_nested_teacher_metadata_gets_specific_repairable_errors(tmp_path):
    _, settings = submission(tmp_path)
    settings["teacher_model"] = {"path_used": "teacher.model"}
    settings["calculator"] = {"dispersion": settings.pop("dispersion")}
    save(tmp_path, "settings.json", settings)
    report = validate(tmp_path)
    assert {"type", "missing_field"} <= codes(report)
    errors = {item["location"]: item for item in report["errors"]}
    assert errors["settings.teacher_model"]["suggestion"]
    assert "settings.dispersion" in errors
    settings["teacher_model"] = "teacher.model"
    settings["dispersion"] = settings.pop("calculator")["dispersion"]
    save(tmp_path, "settings.json", settings)
    assert validate(tmp_path)["valid"]


@pytest.mark.parametrize("level", [1, 2])
def test_calculator_metadata_aliases_and_conflicts_are_reported_consistently(
    tmp_path, level
):
    _, settings = submission(tmp_path)
    settings.update(
        calculator="mace.calculators.MACECalculator",
        default_dtype="float64",
        device="cuda",
    )
    save(tmp_path, "settings.json", settings)
    assert validate(tmp_path, level=level)["valid"]
    settings["model_settings"] = {"dtype": "float32"}
    save(tmp_path, "settings.json", settings)
    report = validate(tmp_path, level=level)
    assert not report["valid"]
    error = next(
        item for item in report["errors"] if item["code"] == "calculator_settings"
    )
    assert error["location"] == "settings.model_settings.dtype"
    assert "Conflicts with settings.default_dtype" in error["message"]
    settings["model_settings"]["dtype"] = "float64"
    save(tmp_path, "settings.json", settings)
    assert validate(tmp_path, level=level)["valid"]


@pytest.mark.parametrize("value", ["192 degrees of freedom", True, None, 192.5])
def test_degrees_of_freedom_requires_an_integer(tmp_path, value):
    _, settings = submission(tmp_path, 6)
    settings["md"]["temperature_dof"] = value
    save(tmp_path, "settings.json", settings)
    report = validate(tmp_path, 6)
    assert any(
        item["code"] == "type" and item["location"] == "settings.md.temperature_dof"
        for item in report["errors"]
    )


def test_directory_npz_and_missing_files_are_distinct_errors(tmp_path):
    manifest, _ = submission(tmp_path)
    (tmp_path / "raw").mkdir()
    np.savez(tmp_path / "structures.npz", positions=np.zeros((1, 2, 3)))
    manifest["artifacts"] = {
        "dataset": "structures.npz",
        "raw": "raw",
        "missing": "absent.csv",
    }
    save(tmp_path, "manifest.json", manifest)
    report = validate(tmp_path)
    assert {"directory_link", "structure_format", "file_path"} <= codes(report)


@pytest.mark.parametrize("level", [1, 2])
def test_trajectory_metadata_wrapper_returns_repairable_error(tmp_path, level):
    manifest, _ = submission(tmp_path, 6)
    np.savez(tmp_path / "samples.npz", positions=np.zeros((2, 64, 3)))
    manifest["artifacts"].update(
        nve_trajectory=save(
            tmp_path,
            "trajectory.json",
            {"arrays_file": "samples.npz", "symbols": ["Al"] * 64, "stage": "nve"},
        ),
        nve_samples="samples.npz",
    )
    save(tmp_path, "manifest.json", manifest)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    validator = build_validate_submission_tool(tmp_path, 6, level=level)
    report = json.loads(validator.execute(manifest_path="/workspace/manifest.json"))
    assert not report["valid"]
    error = next(
        item
        for item in report["errors"]
        if item["location"] == "manifest.artifacts.nve_trajectory"
    )
    assert error["code"] == "structure_data"
    assert "ASE JSON database" in error["message"]
    assert "JSON frames" in error["suggestion"]
    assert "NPZ" in error["suggestion"]
    assert str(tmp_path) not in json.dumps(report)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

    # A corrected trajectory clears this error without accepting the wrapper.
    manifest["artifacts"]["nve_trajectory"] = structure(tmp_path)
    save(tmp_path, "manifest.json", manifest)
    repaired = json.loads(validator.execute(manifest_path="/workspace/manifest.json"))
    assert not any(item["code"] == "structure_data" for item in repaired["errors"])


def test_ase_json_database_remains_a_supported_structure_format(tmp_path):
    manifest, _ = submission(tmp_path)
    from ase.io import read

    write(tmp_path / "dataset.json", read(tmp_path / "frames.traj"), format="json")
    manifest["artifacts"]["dataset"] = "dataset.json"
    save(tmp_path, "manifest.json", manifest)
    assert validate(tmp_path)["valid"]


def test_missing_required_artifact_is_an_error(tmp_path):
    manifest, _ = submission(tmp_path)
    manifest["artifacts"] = {"wrong_role": "frames.traj"}
    save(tmp_path, "manifest.json", manifest)
    assert "missing_artifact" in codes(validate(tmp_path))


@pytest.mark.parametrize("value", [{}, [], {"file": "frames.traj"}, ["frames.traj"]])
def test_single_artifact_role_requires_a_file_path(tmp_path, value):
    manifest, _ = submission(tmp_path)
    manifest["artifacts"]["dataset"] = value
    save(tmp_path, "manifest.json", manifest)
    assert "artifact_link_type" in codes(validate(tmp_path))


def test_null_settings_are_not_a_valid_document(tmp_path):
    submission(tmp_path)
    save(tmp_path, "settings.json", None)
    assert "settings_object" in codes(validate(tmp_path))


def test_supported_role_alias_is_accepted(tmp_path):
    manifest, _ = submission(tmp_path)
    manifest["artifacts"]["labeled_dataset"] = manifest["artifacts"].pop("dataset")
    save(tmp_path, "manifest.json", manifest)
    assert validate(tmp_path)["valid"]


def test_missing_stored_labels_are_reported_without_calculating_them(tmp_path):
    submission(tmp_path)
    write(tmp_path / "frames.traj", Atoms("Ag2", positions=[[0, 0, 0], [1, 0, 0]]))
    report = validate(tmp_path)
    assert "stored_fields" in codes(report)
    assert "energy, forces" in next(
        item["message"] for item in report["errors"] if item["code"] == "stored_fields"
    )


def test_nested_data_references_are_confined_and_must_be_linked(tmp_path):
    manifest, _ = submission(tmp_path)
    manifest["artifacts"]["extra"] = save(
        tmp_path, "index.json", {"arrays_file": "values.npy"}
    )
    np.save(tmp_path / "values.npy", np.zeros((2, 3)))
    save(tmp_path, "manifest.json", manifest)
    assert "unlisted_reference" in codes(validate(tmp_path))
    manifest["artifacts"]["arrays"] = "values.npy"
    save(tmp_path, "manifest.json", manifest)
    assert validate(tmp_path)["valid"]
    save(tmp_path, "index.json", {"arrays_file": "../secret.npy"})
    assert "file_path" in codes(validate(tmp_path))


def test_relative_sibling_files_work_and_paths_do_not_leak(tmp_path):
    submission(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "settings": "../settings.json",
        "scripts": ["../run.py"],
        "artifacts": {"dataset": "../frames.traj"},
    }
    save(output, "manifest.json", manifest)
    report = validate_submission("/workspace/output/manifest.json", tmp_path, 3)
    assert report["valid"], report
    assert str(tmp_path) not in json.dumps(report)


@pytest.mark.parametrize(
    "link",
    [
        "../secret.json",
        "/etc/passwd",
        "/workspace/link.json",
        "/workspace/resources/secret.json",
    ],
)
def test_links_cannot_escape_or_follow_symlinks(tmp_path, link):
    manifest, _ = submission(tmp_path)
    (tmp_path / "link.json").symlink_to(tmp_path.parent / "secret.json")
    manifest["artifacts"]["extra"] = link
    save(tmp_path, "manifest.json", manifest)
    report = validate(tmp_path)
    assert not report["valid"]
    assert "file_path" in codes(report)


@pytest.mark.parametrize(
    "content",
    ['{"artifacts":{},"artifacts":{}}', '{"value":NaN}', '{"value":1e999}', "not JSON"],
)
def test_invalid_json_returns_diagnostics(tmp_path, content):
    (tmp_path / "manifest.json").write_text(content)
    assert "json" in codes(validate(tmp_path))


def test_object_arrays_are_rejected_without_executing_pickle(tmp_path):
    manifest, _ = submission(tmp_path)
    np.save(tmp_path / "unsafe.npy", np.array([{"value": 1}], dtype=object))
    manifest["artifacts"]["extra"] = "unsafe.npy"
    save(tmp_path, "manifest.json", manifest)
    assert "array_file" in codes(validate(tmp_path))


def test_potential_include_needs_a_recognized_role(tmp_path):
    manifest, _ = submission(tmp_path, 2)
    (tmp_path / "input.in").write_text("include /workspace/pot.mod\n")
    (tmp_path / "pot.mod").write_text("pair_style buck/coul/long 8 12\n")
    manifest["artifacts"].update(lammps_inputs=["input.in"], potential_file="pot.mod")
    save(tmp_path, "manifest.json", manifest)
    assert "include_link" in codes(validate(tmp_path, 2))
    manifest["artifacts"]["potential"] = manifest["artifacts"].pop("potential_file")
    save(tmp_path, "manifest.json", manifest)
    assert "include_link" not in codes(validate(tmp_path, 2))


@pytest.mark.parametrize("level", [1, 2])
def test_equilibration_stage_label_is_explained(tmp_path, level):
    manifest, _ = submission(tmp_path, 6)
    manifest["artifacts"]["thermal_trace"] = save(
        tmp_path, "trace.json", [{"stage": "nvt_equilibration", "time_fs": 0}]
    )
    save(tmp_path, "manifest.json", manifest)
    assert "stage_label" in codes(validate(tmp_path, 6, level))
    save(
        tmp_path,
        "trace.json",
        [
            {"stage": "equilibration", "time_fs": 0},
            {"stage": "production", "time_fs": 10},
        ],
    )
    assert "stage_label" not in codes(validate(tmp_path, 6, level))


def test_eight_stage_boundaries_and_all_trace_windows(tmp_path):
    manifest, settings = submission(tmp_path, 10)
    settings["stages"] = [
        {
            "id": f"stage-{i}",
            "target_temperature_K": t,
            "start_time_fs": 20 * i,
            "end_time_fs": 20 * (i + 1),
            "production_start_time_fs": 20 * i + 10,
            "production_end_time_fs": 20 * (i + 1),
        }
        for i, t in enumerate([300, 400, 500, 600, 700, 800, 900, 300])
    ]
    trace = [
        {"stage": stage["id"], "time_fs": stage[key]}
        for stage in settings["stages"]
        for key in ("start_time_fs", "end_time_fs")
    ]
    manifest["artifacts"]["boundary_states"] = structure(tmp_path, count=16)
    manifest["artifacts"]["thermal_trace"] = save(tmp_path, "trace.json", trace)
    save(tmp_path, "manifest.json", manifest)
    save(tmp_path, "settings.json", settings)
    checked = {"stage_count", "boundary_count", "trace_coverage"}
    assert not checked.intersection(codes(validate(tmp_path, 10, 2)))
    trace[-2]["time_fs"] += 1
    save(tmp_path, "trace.json", trace)
    assert "trace_coverage" in codes(validate(tmp_path, 10, 2))
    structure(tmp_path, count=15)
    assert "boundary_count" in codes(validate(tmp_path, 10, 2))


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize(
    ("number", "needle"),
    [
        (4, "cubic_symmetry"),
        (6, "equilibration or nvt"),
        (7, "exclusive_end_index"),
        (8, "default_rng(seed)"),
        (9, "before integration"),
        (10, "before its equilibration"),
    ],
)
def test_shared_submission_guidance(level, number, needle):
    assert needle in example_files(number, level=level)["README.md"]


def test_numeric_template_defaults_accept_floats(tmp_path):
    _, settings = submission(tmp_path, 6)
    settings["md"]["target_temperature_K"] = 300.0
    settings["md"]["initial_temperature_K"] = 300.0
    save(tmp_path, "settings.json", settings)
    assert not any(
        item["code"] == "type" and item["location"].endswith("temperature_K")
        for item in validate(tmp_path, 6)["errors"]
    )


def test_stage_boundary_count_and_missing_initial_trace(tmp_path):
    manifest, settings = submission(tmp_path, 10)
    stage = {
        "id": "300K",
        "target_temperature_K": 300,
        "start_time_fs": 0,
        "end_time_fs": 20,
        "production_start_time_fs": 10,
        "production_end_time_fs": 20,
    }
    settings["stages"] = [stage, dict(stage, id="extra")]
    manifest["artifacts"]["boundary_states"] = structure(tmp_path, count=3)
    manifest["artifacts"]["thermal_trace"] = save(
        tmp_path,
        "trace.json",
        [{"stage": "300K", "time_fs": 11}, {"stage": "300K", "time_fs": 20}],
    )
    save(tmp_path, "manifest.json", manifest)
    save(tmp_path, "settings.json", settings)
    assert {"boundary_count", "stage_count"} <= codes(validate(tmp_path, 10))
    settings["stages"] = [stage]
    save(tmp_path, "settings.json", settings)
    assert "trace_coverage" in codes(validate(tmp_path, 10))


@pytest.mark.parametrize("level", [1, 2])
def test_every_task_has_a_bound_read_only_validator(level, tmp_path):
    environments = create_environments(str(tmp_path), level=level)
    for number in range(1, 11):
        env = environments[f"level_{level}_task_{number}"]
        selected = env.tools["validate_submission"]
        assert selected.trusted
        assert selected.workspace_access == WorkspaceAccess.READ
        assert set(selected.params_json_schema["properties"]) == {"manifest_path"}
        report = json.loads(selected.execute(manifest_path="/workspace/missing.json"))
        assert report["task"] == f"level_{level}_task_{number}"
        assert not report["valid"]
        assert "validate_submission" in example_files(number, level=level)["README.md"]


def test_workspace_rebinding_does_not_read_another_trial(tmp_path):
    roots = [tmp_path / name for name in ("one", "two")]
    validators = []
    for root in roots:
        root.mkdir()
        submission(root)
        environment = MolecularDynamicsEnvironment(
            "md",
            TaskDefinition(
                name="md",
                description="md",
                tools=[],
                submission_format="",
                scoring_fn=WorkflowScorer(3, level=1),
            ),
            workspace_path=str(root),
            toolset=Toolset(workspace_factory=None),
        )
        validators.append(environment.tools["validate_submission"])
    (roots[0] / "frames.traj").unlink()
    assert not json.loads(
        validators[0].execute(manifest_path="/workspace/manifest.json")
    )["valid"]
    assert json.loads(validators[1].execute(manifest_path="/workspace/manifest.json"))[
        "valid"
    ]


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize(
    ("metadata", "invalid"),
    [
        ({}, True),
        ({"run_id": "run-1"}, True),
        ({"run_id": "run-1", "action_id": "action-1"}, False),
        ({"provenance": {"run_id": "run-1", "action_id": "action-1"}}, False),
    ],
)
def test_validator_uses_current_task_provenance_requirement(
    tmp_path, required, metadata, invalid
):
    from corral_md.workflow_scoring.verification import ModalVerifier

    manifest, _ = submission(tmp_path, 10)
    manifest.update(metadata)
    save(tmp_path, "manifest.json", manifest)
    environment = MolecularDynamicsEnvironment(
        "md",
        TaskDefinition(
            name="md",
            description="md",
            tools=[],
            submission_format="",
            scoring_fn=WorkflowScorer(
                10, level=2, verifier=ModalVerifier(require_provenance=required)
            ),
        ),
        workspace_path=str(tmp_path),
        toolset=Toolset(workspace_factory=None),
    )
    report = json.loads(
        environment.tools["validate_submission"].execute(
            manifest_path="/workspace/manifest.json"
        )
    )
    assert ("provenance_identifiers" in codes(report)) is (required and invalid)


def test_validator_rejects_conflicting_provenance(tmp_path):
    manifest, _ = submission(tmp_path)
    manifest.update(run_id="one", provenance={"run_id": "two"})
    save(tmp_path, "manifest.json", manifest)
    report = validate(tmp_path)
    assert "provenance_identifiers" in codes(report)
    assert any("Conflicting" in error["message"] for error in report["errors"])


@pytest.mark.parametrize("restricted", [False, True])
def test_runtime_validation_keeps_trial_open(tmp_path, monkeypatch, restricted):
    environment = create_environments(str(tmp_path / "definitions"))[
        "level_1_task_3"
    ].for_task("validation-trial", workspace_path=str(tmp_path / "trial"))
    root = tmp_path / "trial"
    root.mkdir(exist_ok=True)
    submission(root)
    started = environment.initial_event(execution_id="validation-trial")
    action = Action(
        id="validate",
        name="validate_submission",
        arguments={"manifest_path": "/workspace/manifest.json"},
    )
    state = ExecutionState(
        through_commit_hash="0" * 64,
        execution_id="validation-trial",
        branch_id="main",
        task=TaskState(environment=started.environment_metadata),
        environment=EnvironmentState(values=started.environment),
        workspace=started.workspace,
        runtime=RuntimeState(),
        actions={action.id: ActionState(action=action, requested_by_run_id="agent")},
    )
    monkeypatch.setattr(permissions, "enabled", lambda: restricted)
    # Exercise restricted tool dispatch without configuring the independent
    # OS snapshot service in this unit test's temporary directory.
    monkeypatch.setattr(permissions, "snapshot_source", nullcontext)
    monkeypatch.setattr(
        permissions,
        "run_worker",
        lambda *_a, **_k: pytest.fail("validator must use the confined reader"),
    )
    monkeypatch.setattr(
        WorkflowScorer, "evaluate", lambda *_a, **_k: pytest.fail("must not grade")
    )
    result = execute_action(
        environment,
        state,
        action,
    )
    assert result.success, result.observation
    assert json.loads(result.observation)["valid"]
    assert state.submission is None
