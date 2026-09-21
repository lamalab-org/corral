import pytest

from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.runtime import environment_loader


def _environment(task_id: str) -> Environment:
    return Environment(
        task_id,
        TaskDefinition(
            name=task_id,
            description=f"Run {task_id}",
            tools=[],
            scoring_fn=lambda answer: float(bool(answer)),
            submission_format={"answer": "string"},
            resolve_answer=False,
        ),
        toolset=Toolset(workspace_factory=None),
    )


def test_factory_kwargs_fill_only_declared_parameters(tmp_path):
    def factory(*, work_dir, level, subtask=False, custom=None):
        del work_dir, level, subtask, custom

    kwargs = environment_loader._factory_kwargs(
        factory,
        initial={"custom": 3},
        work_dir=tmp_path,
        level=2,
        subtasks=True,
        config_parameter=None,
        task_config=None,
    )

    assert kwargs == {
        "work_dir": str(tmp_path),
        "level": 2,
        "subtask": True,
        "custom": 3,
    }


def test_environment_kwargs_configure_registered_factory(monkeypatch, tmp_path):
    calls = []

    def factory(*, work_dir, level, subtask_level=False, custom=None):
        calls.append((work_dir, level, subtask_level, custom))
        return _environment("registered-task")

    monkeypatch.setattr(environment_loader, "_load_factory", lambda spec: factory)
    work_dir = tmp_path / "workspace"

    environments = environment_loader.load_environment_group(
        "corral_md",
        env_kwargs={
            "work_dir": str(work_dir),
            "level": 2,
            "subtasks": True,
            "custom": 3,
        },
        repository_root=tmp_path,
    )

    assert list(environments) == ["registered-task"]
    assert calls == [(str(work_dir.resolve()), 2, True, 3)]


def test_unknown_environment_lists_registered_names():
    try:
        environment_loader.load_environment_group("unknown")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected an unknown environment to fail")

    assert "unknown environment 'unknown'" in message
    assert ", ".join(environment_loader.ENVIRONMENT_NAMES) in message


@pytest.mark.parametrize("level", [1, 2])
def test_stargazer_factory_selects_its_level_bank(monkeypatch, tmp_path, level):
    calls = []

    def factory(*, work_dir, level, selector_path=None):
        calls.append((work_dir, level, selector_path))
        return {"stargazer-task": _environment("stargazer-task")}

    def load_factory(specification):
        assert specification == "stargazer.env:create_environments"
        return factory

    monkeypatch.setattr(environment_loader, "_load_factory", load_factory)
    work_dir = tmp_path / "workspace"

    environments = environment_loader.load_environment_group(
        "stargazer",
        env_kwargs={"level": level, "work_dir": str(work_dir)},
        repository_root=tmp_path,
    )

    assert list(environments) == ["stargazer-task"]
    assert calls == [(str(work_dir.resolve()), level, None)]
    assert "stargazer" in environment_loader.ENVIRONMENT_NAMES
    assert (
        environment_loader.ENVIRONMENT_PRESETS["stargazer"].source_dir
        == "tasks/stargazer/src"
    )


@pytest.mark.parametrize("config_option", ["selector_path", "task_config"])
def test_stargazer_custom_selector_config(monkeypatch, tmp_path, config_option):
    calls = []

    def factory(*, work_dir, level, selector_path=None):
        calls.append(selector_path)
        return {"selected-task": _environment("selected-task")}

    monkeypatch.setattr(environment_loader, "_load_factory", lambda spec: factory)
    selector = tmp_path / "selector.json"
    selector.write_text("[]")

    environments = environment_loader.load_environment_group(
        "stargazer",
        env_kwargs={config_option: str(selector), "level": 1},
        repository_root=tmp_path,
    )

    assert list(environments) == ["selected-task"]
    assert calls == [str(selector.resolve())]
