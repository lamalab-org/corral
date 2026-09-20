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
    assert "submission_examples/README.md" in prompt
    assert "Narrative reports are optional" not in prompt
    assert set(load_example(number)) == {"manifest", "files"}
    result = TaskScorer(task, workspace=tmp_path).evaluate(_state("{}"))
    assert result.score == 0
    assert result.metrics == {"score": 0}
    assert result.metadata == {}
    assert result.feedback is None
    report = task.scoring_fn.evaluate("{}")
    assert report["task_number"] == number
    assert sum(check["points"] for check in report["checks"]) == 100
    assert any(check["status"] == "unverified" for check in report["checks"])
    assert result.scorer_version == "corral_md.score:WorkflowScorer"
    assert report["version"] == "6"
    assert report["status"] == "complete"
    assert all("requirement" in check for check in report["checks"] if check["points"])


@pytest.mark.parametrize("number", range(1, 11))
def test_examples_are_readable_and_survive_workspace_restoration(number, tmp_path):
    task_id = f"level_2_task_{number}"
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_2/tasks_json/task_{number}.json", str(tmp_path)
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
    for name, expected in example_files(number).items():
        relative = f"submission_examples/{name}"
        assert relative in started.workspace.files
        assert read_file.execute(path="/workspace/" + relative) == expected
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


def test_unbound_prompt_includes_linked_file_examples(tmp_path):
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_1.json", str(tmp_path)
    )["level_2_task_1"]
    environment = env.MolecularDynamicsEnvironment("task", task)
    prompt = env._md_task_prompt(environment, _state("{}"))
    examples = [
        json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", prompt, re.S)
    ]
    bundle = load_example(1)
    assert examples == [bundle["manifest"], *bundle["files"].values()]
    assert "unwrapped_positions_A" in prompt
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
    state = _state("{}")
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
    assert 0 < result.score <= 0.1
    assert result.metrics == {"score": result.score}
    assert result.metadata == {}
    assert result.feedback is None
    assert not former.exists()
