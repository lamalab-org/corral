import json

import pytest
from corral_md import env


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("level_1/tasks_json", 10),
        ("level_2/tasks_json", 10),
    ],
)
def test_all_shipped_tasks_load_from_unrelated_cwd(source, count, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks = env.load_tasks_from_json(env.PACKAGE_DATA_ROOT / source, str(tmp_path))
    assert len(tasks) == count
    assert all(callable(task.scoring_fn) for task in tasks.values())


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
