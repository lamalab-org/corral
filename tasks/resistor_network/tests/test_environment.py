from pathlib import Path

import pytest

from corral.runtime.environment_loader import load_environment_group


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
        assert "calculate_series_resistance" in environment.tools
        assert "read_file" in environment.tools
        assert "write_file" in environment.tools
    Path(first.workspace_path, "trial.txt").write_text("first")
    assert not Path(second.workspace_path, "trial.txt").exists()


def test_runtime_default_work_directory_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv("CORRAL_WORK_DIR", str(tmp_path))
    environments = load_environment_group("resistor_network")
    environment = environments["task_0"].for_task("default-workspace")

    assert Path(environment.workspace_path).parent == tmp_path
