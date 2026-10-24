import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest
from corral_md import env
from hatchling.builders.wheel import WheelBuilder

from corral.core.environment import Toolset
from corral.core.state import ExecutionState, RuntimeState


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("level_1/tasks_json", 10),
        ("level_2/tasks_json", 10),
    ],
)
def test_all_shipped_tasks_load_from_unrelated_cwd(
    source, count, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    tasks = env.load_tasks_from_json(env.PACKAGE_DATA_ROOT / source)
    assert len(tasks) == count
    assert all(callable(task.scoring_fn) for task in tasks.values())
    assert all("work_dir" not in task.initial_input for task in tasks.values())
    expected_level = int(source.split("/")[0].removeprefix("level_"))
    assert all(task.scoring_fn.level == expected_level for task in tasks.values())
    for task in tasks.values():
        assert "validate_submission" not in task.submission_format
        assert task.scoring_fn.evaluate("{}")["score"] == 0


@pytest.mark.usefixtures("md_assets")
def test_wheel_loads_both_levels_without_editable_source(tmp_path):
    task_root = Path(__file__).resolve().parents[1]
    wheel = next(WheelBuilder(str(task_root)).build(directory=str(tmp_path)))
    installed = tmp_path / "installed"
    with ZipFile(wheel) as archive:
        archive.extractall(installed)

    # Reuse installed dependencies, but remove the editable package and run
    # outside the repository so task data cannot leak in from the checkout.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
from pathlib import Path

installed, source = map(Path, sys.argv[1:])
sys.path = [str(installed)] + [
    path for path in sys.path if Path(path).resolve() != source
]
from corral_md import env

assert Path(env.__file__).is_relative_to(installed)
assert env.PACKAGE_DATA_ROOT.is_relative_to(installed)
for level in (1, 2):
    environments = env.create_environments(
        work_dir=str(installed.parent / "work"), level=level
    )
    assert len(environments) == 10
    for environment in environments.values():
        simulation = environment.tools["run_lammps"]
        assert environment.local_mode
        assert not simulation.trusted
        assert "corral_action_id" not in simulation.hidden_args
        assert "run_verified_md" not in environment.tools
        assert "validate_submission" not in environment.tools
        task = environment.current_task
        assert "validate_submission" not in task.submission_format
        assert task.scoring_fn.evaluate("{}")["score"] == 0
    assert not list(Path(env.__file__).parent.glob("*_recording.py"))
""",
            str(installed),
            str(task_root / "src"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_removed_subtasks_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "PACKAGE_DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        env,
        "build_environments",
        lambda *_args, **_kwargs: pytest.fail("No removed subtasks should be built"),
    )

    with pytest.raises(ValueError, match="no subtasks; select level 1 or 2"):
        env.create_environments(
            work_dir=str(tmp_path / "work"), subtask_level=True, level=1
        )


@pytest.mark.parametrize("level", ["level_1", "level_2"])
def test_expansion_prompt_constructs_its_aluminum_cell(level):
    path = env.PACKAGE_DATA_ROOT / level / "tasks_json/task_7.json"
    description = json.loads(path.read_text())[0]["description"]
    assert (
        r"Start with a conventional FCC unit cell of Aluminum (a = 4.05 \u00c5) "
        r"and replicate it into a 3 \u00d7 3 \u00d7 3 supercell." in description
    )
    assert "supplied periodic 108-atom FCC aluminum cell" not in description


def test_regression_prompt_requires_agent_written_calculations():
    path = env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_8.json"
    description = json.loads(path.read_text())[0]["description"]
    assert "with your own code" in description
    assert "freeze the final and learning-curve pipelines" in description


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("number", range(1, 11))
def test_agent_workspace_and_prompt_do_not_supply_recording_helpers(
    tmp_path, level, number
):
    task_id = f"level_{level}_task_{number}"
    task = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_{level}/tasks_json/task_{number}.json"
    )[task_id]
    environment = env.MolecularDynamicsEnvironment(
        task_id,
        replace(task, tools=[]),
        workspace_path=str(tmp_path),
        base_work_dir=str(tmp_path),
        toolset=Toolset(workspace_factory=None),
    )
    environment._ensure_seed_directories()
    assert not any(path.is_file() for path in tmp_path.rglob("*"))
    state = ExecutionState(
        through_commit_hash="a" * 64,
        execution_id=task_id,
        branch_id="main",
        runtime=RuntimeState(status="submitted"),
        submission="{}",
    )
    prompt = environment.get_task_prompt(state)
    assert "Recorder" not in prompt
    assert "_recording.py" not in prompt


@pytest.mark.parametrize(
    ("name", "params", "error"),
    [
        ("unknown_scorer", {}, "not found in the registry"),
        ("check_level2_workflow", {}, "task_number"),
        (
            "check_level2_workflow",
            {"task_number": 1, "target": None},
            "unexpected keyword argument 'target'",
        ),
    ],
)
@pytest.mark.usefixtures("md_assets")
def test_invalid_tasks_fail_before_environments_are_built(
    name, params, error, tmp_path, monkeypatch
):
    source = tmp_path / "level_1/tasks_json"
    source.mkdir(parents=True)
    task_file = source / "invalid.json"
    task_file.write_text(
        json.dumps(
            [
                {
                    "id": "invalid_task",
                    "scoring_function": name,
                    "scoring_params": params,
                }
            ]
        )
    )
    monkeypatch.setattr(env, "PACKAGE_DATA_ROOT", tmp_path)

    def unexpected_build(*_args, **_kwargs):
        pytest.fail("Environments must not be built for invalid tasks")

    monkeypatch.setattr(env, "build_environments", unexpected_build)
    with pytest.raises(ValueError, match=error) as exc:
        env.create_environments(work_dir=str(tmp_path / "work"))
    assert "invalid_task" in str(exc.value)
    assert str(task_file) in str(exc.value)


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.usefixtures("md_assets")
def test_all_tasks_require_direct_json_answers(tmp_path, level, local):
    tasks = env.load_tasks_from_json(
        env.PACKAGE_DATA_ROOT / f"level_{level}/tasks_json", local=local
    )
    for task in tasks.values():
        assert task.submission_format.startswith("Return JSON:\n\n```json\n")
        example = re.search(r"```json\n(.*?)\n```", task.submission_format, re.S)
        assert example is not None
        answer = example[1]
        assert isinstance(json.loads(answer)["artifacts"], dict)
        resolved = task.submission_resolver(answer, tmp_path)
        expected = json.loads(answer.replace("/workspace/", f"{tmp_path}/"))
        assert json.loads(resolved) == expected
        if local:
            invalid = task.submission_resolver("/workspace/manifest.json", tmp_path)
            assert json.loads(invalid)["invalid_submission"]
        else:
            with pytest.raises(ValueError, match="JSON object directly"):
                task.submission_resolver("/workspace/manifest.json", tmp_path)
