"""The shipped definitions use offline graders and expose blank submissions."""

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from corral_md import env
from corral_md.score import PendingReviewError, WorkflowScorer
from corral_md.submission_examples import example_files, load_example

from corral.core.environment import Toolset
from corral.core.state import ExecutionState, RuntimeState
from corral.evaluation import TaskScorer


def _state(submission):
    return ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="offline-check",
        branch_id="main",
        runtime=RuntimeState(status="submitted"),
        submission=submission,
    )


@pytest.mark.parametrize("number", range(1, 11))
def test_shipped_workflow_contract_and_empty_evidence(number, tmp_path):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_2/tasks_json/task_{number}.json", str(tmp_path)
    )[f"level_2_task_{number}"]
    assert isinstance(task.scoring_fn, WorkflowScorer)
    assert task.scoring_fn.task_number == number
    current = SimpleNamespace(
        current_task=task,
        task_id=f"level_2_task_{number}",
        workspace_path=str(tmp_path),
        resolve_inputs=lambda _state: {},
    )
    prompt = env._md_task_prompt(current, _state("{}"))
    manifest = json.loads(re.search(r"```json\n(.*?)\n```", prompt, re.S)[1])
    assert manifest == load_example(number)["manifest"]
    assert "Level_2" not in prompt
    assert "Level 2" not in prompt
    assert "submission_examples/README.md" in prompt
    assert "Narrative reports are optional" not in prompt
    assert set(load_example(number)) == {"manifest", "files"}
    result = TaskScorer(task, workspace=tmp_path).evaluate(_state("{}"))
    assert result.score == 0
    assert result.metrics == {"score": 0}
    assert result.metadata["workflow_evaluation"]["score"] == 0
    assert "manifest_and_artifacts" in result.feedback
    report = task.scoring_fn.evaluate("{}")
    assert report["task_number"] == number
    assert sum(check["points"] for check in report["checks"]) == 100
    assert any(check["status"] == "unverified" for check in report["checks"])
    assert result.scorer_version == "corral_md.score:WorkflowScorer"
    assert "version" not in report
    assert report["status"] == "complete"


@pytest.mark.parametrize("number", range(1, 11))
def test_shipped_level1_contract_uses_only_level1_scorer(number, tmp_path):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_1/tasks_json/task_{number}.json", str(tmp_path)
    )[f"level_1_task_{number}"]
    assert isinstance(task.scoring_fn, WorkflowScorer)
    assert task.scoring_fn.task_number == number
    assert task.scoring_fn.level == 1
    current = SimpleNamespace(
        current_task=task,
        task_id=f"level_1_task_{number}",
        workspace_path=str(tmp_path),
        resolve_inputs=lambda _state: {},
    )
    prompt = env._md_task_prompt(current, _state("{}"))
    manifest = json.loads(re.search(r"```json\n(.*?)\n```", prompt, re.S)[1])
    assert manifest == load_example(number, level=1)["manifest"]
    assert "Level_1" not in prompt
    assert "Level 1" not in prompt
    assert "Leave the Level 2 parts empty" not in prompt
    report = task.scoring_fn.evaluate("{}")
    assert report["score"] == 0
    assert report["level"] == 1
    assert report["possible_points"] == 100
    assert sum(check["points"] for check in report["checks"]) == 100


@pytest.mark.parametrize("number", range(1, 11))
@pytest.mark.parametrize("level", [1, 2])
def test_examples_are_readable_and_survive_workspace_restoration(
    number, level, tmp_path
):
    task_id = f"level_{level}_task_{number}"
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_{level}/tasks_json/task_{number}.json",
        str(tmp_path),
    )[task_id]
    environment = env.MolecularDynamicsEnvironment(
        task_id,
        replace(task, tools=[]),
        base_work_dir=str(tmp_path),
        task_execution_id="examples",
        toolset=Toolset(workspace_factory=env._md_file_tools),
    )
    started = environment.initial_event(execution_id="examples")
    root = Path(environment.workspace_path)
    read_file = environment.toolset.resolve(task, str(root))["read_file"]
    assert {path.name for path in (root / "submission_examples").iterdir()} == set(
        example_files(number, level=level)
    )
    for name, expected in example_files(number, level=level).items():
        relative = f"submission_examples/{name}"
        assert relative in started.workspace.files
        assert read_file.execute(path="/workspace/" + relative) == expected
        if name == "README.md":
            assert f"Level {level}" not in expected
        if name.endswith(".json"):
            json.loads(expected)
    # Edited templates are part of the run's snapshot, not regenerated on resume.
    manifest = root / "submission_examples/manifest.json"
    edited = json.loads(manifest.read_text())
    edited["scripts"] = ["my_analysis.py"]
    manifest.write_text(json.dumps(edited))
    environment._ensure_seed_directories()
    assert json.loads(manifest.read_text()) == edited
    saved = environment.capture_workspace(started.workspace)
    shutil.rmtree(root)
    root.mkdir()
    environment.prepare_workspace(saved)
    assert json.loads(manifest.read_text()) == edited
    assert (root / "submission_examples/settings.json").is_file()


def test_level1_examples_omit_later_workflow_fields():
    examples = {number: load_example(number, level=1) for number in range(1, 11)}
    assert "diffusion_data" not in examples[1]["manifest"]["artifacts"]
    assert "reheating_end" not in examples[2]["files"]["boundary_states.json"]
    assert "trained_checkpoint" not in examples[3]["manifest"]["artifacts"]
    assert "bands" not in examples[4]["manifest"]["artifacts"]
    assert "fit" not in examples[5]["manifest"].get("results", {})
    assert "spectrum" not in examples[6]["manifest"]["artifacts"]
    assert "pressure_fit" not in examples[7]["files"]["settings.json"]
    assert "id_test" not in examples[8]["files"]["regression_data.json"]
    assert set(examples[9]["manifest"]["artifacts"]["runs"]) == {"main"}
    assert "heat_capacity" not in examples[10]["manifest"]["results"]


@pytest.mark.parametrize("level", [1, 2])
def test_unbound_prompt_includes_linked_file_examples(level, tmp_path):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_{level}/tasks_json/task_1.json", str(tmp_path)
    )[f"level_{level}_task_1"]
    environment = env.MolecularDynamicsEnvironment("task", task)
    prompt = env._md_task_prompt(environment, _state("{}"))
    examples = [
        json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", prompt, re.S)
    ]
    bundle = load_example(1, level=level)
    assert examples == [bundle["manifest"], *bundle["files"].values()]
    assert ("unwrapped_positions_A" in prompt) == (level == 2)
    assert "Fields and data layouts" not in prompt


def test_missing_manifest_uses_existing_resolution_error(tmp_path):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_1.json", str(tmp_path)
    )["level_2_task_1"]
    with pytest.raises(FileNotFoundError, match="missing"):
        TaskScorer(task, workspace=tmp_path).evaluate(_state("missing.json"))


def test_shared_scorer_propagates_pending_instead_of_recording_zero(
    tmp_path, monkeypatch
):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_10.json", str(tmp_path)
    )["level_2_task_10"]
    monkeypatch.setattr(
        task.scoring_fn.module,
        "evaluate",
        lambda _e, r: r.check(
            "consistent_structural_return_comparison",
            90,
            None,
            "Independent metric review",
        ),
    )
    (tmp_path / "settings.json").write_text('{"model": "MACE-MP-0"}')
    (tmp_path / "run.py").write_text("raise RuntimeError('must never run')")
    (tmp_path / "data.json").write_text("[1]")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "results": {"value": 1},
                "settings": "settings.json",
                "scripts": ["run.py"],
                "artifacts": {"data": "data.json"},
            }
        )
    )
    state = _state(str(manifest))
    before = state.model_dump()
    with pytest.raises(PendingReviewError) as pending:
        TaskScorer(task, workspace=tmp_path).evaluate(state)
    assert pending.value.report["score"] is None
    assert pending.value.report["status"] == "pending_review"
    assert state.model_dump() == before


def test_partial_manifest_survives_workspace_restoration(tmp_path):
    former = tmp_path / "former"
    restored = tmp_path / "restored"
    outputs = restored / "results"
    outputs.mkdir(parents=True)
    (outputs / "settings.json").write_text(
        json.dumps({"model": "MACE-MP-0", "units": "eV"})
    )
    (outputs / "report.json").write_text(
        json.dumps({"limitations": "Calculation incomplete"})
    )
    (outputs / "run.py").write_text("raise RuntimeError('must never run')")
    (outputs / "data.json").write_text("[1, 2, 3]")
    manifest = {
        "results": {"status": "incomplete", "units": "eV"},
        "settings": str(former / "results/settings.json"),
        "report": str(former / "results/report.json"),
        "scripts": [str(former / "results/run.py")],
        "artifacts": {"data": str(former / "results/data.json")},
    }
    (outputs / "manifest.json").write_text(json.dumps(manifest))
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_1.json", str(former)
    )["level_2_task_1"]
    result = TaskScorer(task, workspace=restored).evaluate(
        _state(str(former / "results/manifest.json"))
    )
    assert result.score == 0
    assert result.metrics == {"score": result.score}
    assert result.metadata["workflow_evaluation"]["score"] == 0
    assert "reported_results" in result.feedback
    assert not former.exists()
