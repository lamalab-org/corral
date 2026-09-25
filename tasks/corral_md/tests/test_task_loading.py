import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from corral_md import env
from hatchling.builders.wheel import WheelBuilder


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
    tasks = env.load_tasks_from_json(env.PACKAGE_DATA_ROOT / source, str(tmp_path))
    assert len(tasks) == count
    assert all(callable(task.scoring_fn) for task in tasks.values())
    expected_level = int(source.split("/")[0].removeprefix("level_"))
    assert all(task.scoring_fn.level == expected_level for task in tasks.values())


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
from corral_md.submission_examples import load_example
from corral_md.workflow_scoring.lammps_checks import supplied_cu32_cell

assert Path(env.__file__).is_relative_to(installed)
assert env.PACKAGE_DATA_ROOT.is_relative_to(installed)
for level in (1, 2):
    environments = env.create_environments(
        work_dir=str(installed.parent / "work"), level=level
    )
    assert len(environments) == 10
    for number in range(1, 11):
        assert load_example(number, level=level)["manifest"]
assert len(supplied_cu32_cell()) == 32
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


def test_removed_subtasks_create_no_environments(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "PACKAGE_DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        env,
        "build_environments",
        lambda *args, **kwargs: pytest.fail("No removed subtasks should be built"),
    )

    environments = env.create_environments(
        work_dir=str(tmp_path / "work"), subtask_level=True, level=1
    )

    assert environments == {}


@pytest.mark.parametrize("level", ["level_1", "level_2"])
def test_expansion_prompt_constructs_its_aluminum_cell(level):
    path = env.PACKAGE_DATA_ROOT / level / "tasks_json/task_7.json"
    description = json.loads(path.read_text())[0]["description"]
    assert (
        "conventional cubic FCC cell with lattice parameter 4.05 Å 3 x 3 x 3"
        in description
    )
    assert "supplied periodic 108-atom FCC aluminum cell" not in description


def test_regression_prompt_requires_replayable_generation_script():
    path = env.PACKAGE_DATA_ROOT / "level_2/tasks_json/task_8.json"
    description = json.loads(path.read_text())[0]["description"]
    assert "generation script needed to replay the structures" in description


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

    def unexpected_build(*args, **kwargs):
        pytest.fail("Environments must not be built for invalid tasks")

    monkeypatch.setattr(env, "build_environments", unexpected_build)
    with pytest.raises(ValueError, match=error) as exc:
        env.create_environments(work_dir=str(tmp_path / "work"))
    assert "invalid_task" in str(exc.value)
    assert str(task_file) in str(exc.value)
