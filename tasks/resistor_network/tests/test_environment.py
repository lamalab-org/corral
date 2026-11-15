from pathlib import Path

import pytest
from resistor_network.env import load_tasks_from_json

from corral.evaluation.scorer import _resolve_submission
from corral.runtime.environment_loader import load_environment_group

TASKS_ROOT = Path(__file__).resolve().parents[1] / "environments"


@pytest.mark.parametrize("level", [1, 2])
def test_runtime_work_directory_creates_isolated_trial_workspaces(tmp_path, level):
    environments = load_environment_group(
        "resistor_network",
        env_kwargs={"level": level, "work_dir": str(tmp_path)},
    )
    template = environments["task_0"]
    first = template.for_task("first-trial")
    second = template.for_task("second-trial")

    assert template.base_work_dir == str(tmp_path)
    assert template.workspace_path is None
    assert first.workspace_path != second.workspace_path
    for environment in (first, second):
        workspace = Path(environment.workspace_path)
        assert workspace.is_dir()
        assert workspace.parent == tmp_path
        assert "simulate_circuit_resistance" in environment.tools
        assert "validate_circuit_topology" in environment.tools
        assert not {
            "estimate_resistor_values",
            "propose_simple_topology",
            "validate_measurements",
        } & set(environment.tools)
        assert "read_file" in environment.tools
        assert "write_file" in environment.tools
    Path(first.workspace_path, "trial.txt").write_text("first")
    assert not Path(second.workspace_path, "trial.txt").exists()


def test_runtime_default_work_directory_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv("CORRAL_WORK_DIR", str(tmp_path))
    environments = load_environment_group("resistor_network")
    environment = environments["task_0"].for_task("default-workspace")

    assert Path(environment.workspace_path).parent == tmp_path


@pytest.mark.parametrize("level", [1, 2])
def test_prose_answers_are_scored_zero_instead_of_resolved_as_files(tmp_path, level):
    tasks = load_tasks_from_json(TASKS_ROOT / f"level_{level}" / "tasks_json")
    prose = "I am computing all simulated effective resistances first. " * 20

    for task in tasks.values():
        assert task.resolve_answer is False
        answer = _resolve_submission(task, prose, tmp_path)
        assert answer == prose
        assert task.scoring_fn(answer) == 0.0
