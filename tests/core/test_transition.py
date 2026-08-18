"""Architecture tests for the core stateless Environment API."""

from __future__ import annotations

from pathlib import Path

import pytest

from corral.core import (
    Action,
    State,
    ToolExecutionResult,
    execute_action,
    propose_action,
)
from corral.core.environment import Environment, Toolset
from corral.core.task import EnvironmentSetup, TaskDefinition
from corral.core.tool import tool
from corral.core.tool_catalog import ToolCatalogSnapshot


def _task(*, setup_fn=None) -> TaskDefinition:
    return TaskDefinition(
        name="counter",
        description="increment the counter",
        tools=["increment"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={"answer": "string"},
        setup_fn=setup_fn,
        resolve_answer=False,
    )


@tool(hidden_args=["counter"])
def increment(counter: int) -> ToolExecutionResult:
    """Increment the hidden counter."""
    value = counter + 1
    return ToolExecutionResult(
        content=str(value),
        environment={
            "hidden_arguments": {"counter": value},
            "jobs": {},
            "values": {},
        },
    )


def _environment(*, setup_fn=None, work_dir: Path | None = None) -> Environment:
    return Environment(
        "counter",
        _task(setup_fn=setup_fn),
        base_work_dir=str(work_dir or ""),
        toolset=Toolset(
            pool={"increment": increment},
            workspace_factory=None,
        ),
        task_execution_id="execution" if work_dir is not None else None,
    )


def _configured_state(environment: Environment, counter: int = 0) -> State:
    state = environment.initial_state()
    return state.fork(
        environment={
            **dict(state.environment),
            "hidden_arguments": {"counter": counter},
        }
    )


def test_environment_does_not_own_state() -> None:
    environment = _environment()

    assert not hasattr(environment, "state")
    assert not hasattr(environment, "call_tool")
    assert not hasattr(environment, "runtime_id")
    assert not hasattr(environment, "for_trial")
    assert hasattr(environment, "for_task")


def test_initial_state_is_ready_for_the_action_agent() -> None:
    environment = _environment()

    state = environment.initial_state()

    assert "increment the counter" in state.metadata.task["prompt"]
    catalog = ToolCatalogSnapshot.model_validate(
        state.metadata.environment["tool_catalog"]
    )
    assert [tool["function"]["name"] for tool in catalog.tools] == [
        "increment",
        "submit_answer",
    ]


def test_execute_action_forks_without_mutating_parent() -> None:
    environment = _environment()
    initial = _configured_state(environment, counter=4)
    action = Action(name="increment", arguments={}, id="increment-1")
    proposed = propose_action(initial, action)

    completed = execute_action(environment, proposed, action)

    assert initial.revision == 1
    assert initial.environment["hidden_arguments"] == {"counter": 4}
    assert proposed.revision == 2
    assert proposed.pending_action == action
    assert completed.revision == 3
    assert completed.pending_action is None
    assert completed.environment["hidden_arguments"] == {"counter": 5}
    assert completed.messages[-1]["content"] == "5"
    assert completed.usage.tool_calls == 1


def test_same_environment_definition_has_no_cross_execution_contamination() -> None:
    environment = _environment()
    first = _configured_state(environment, counter=1)
    second = _configured_state(environment, counter=10)

    first_action = Action(name="increment", arguments={}, id="first")
    second_action = Action(name="increment", arguments={}, id="second")
    first_result = execute_action(
        environment, propose_action(first, first_action), first_action
    )
    second_result = execute_action(
        environment, propose_action(second, second_action), second_action
    )

    assert first_result.environment["hidden_arguments"] == {"counter": 2}
    assert second_result.environment["hidden_arguments"] == {"counter": 11}


def test_execute_action_requires_a_persisted_pending_action() -> None:
    environment = _environment()
    state = _configured_state(environment)
    action = Action(name="increment", arguments={})

    with pytest.raises(ValueError, match="pending action"):
        execute_action(environment, state, action)


def test_setup_returns_explicit_hidden_namespace() -> None:
    def setup(_environment: Environment, _state: State) -> EnvironmentSetup:
        return EnvironmentSetup(
            hidden_arguments={"counter": 7},
            values={"instrument": "ready"},
            status="configured",
        )

    environment = _environment(setup_fn=setup)
    initial = environment.initial_state()

    configured, status = environment.configure(initial)

    assert status == "configured"
    assert configured.environment["hidden_arguments"] == {"counter": 7}
    assert configured.environment["values"] == {"instrument": "ready"}
    assert initial.environment["hidden_arguments"] == {}


def test_workspace_changes_are_committed_to_the_child_state(tmp_path: Path) -> None:
    workspace: Path | None = None

    @tool
    def write_report() -> str:
        """Write a report into the bound materialization."""
        assert workspace is not None
        (workspace / "report.txt").write_text("done", encoding="utf-8")
        return "written"

    task = TaskDefinition(
        name="writer",
        description="write",
        tools=["write_report"],
        scoring_fn=lambda _answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    environment = Environment(
        "counter",
        task,
        base_work_dir=str(tmp_path),
        toolset=Toolset(
            pool={"write_report": write_report},
            workspace_factory=None,
        ),
        task_execution_id="execution",
    )
    workspace = Path(environment.workspace_path or "")
    initial = environment.initial_state()
    action = Action(name="write_report", arguments={}, id="write-1")

    completed = execute_action(environment, propose_action(initial, action), action)

    assert initial.workspace.files == {}
    assert completed.workspace.revision == 1
    assert completed.workspace.files["report.txt"].created_by_action == "write-1"


def test_workspace_names_are_unambiguous_and_cannot_be_path_injected(
    tmp_path: Path,
) -> None:
    task = TaskDefinition(
        name="task",
        description="task",
        tools=[],
        scoring_fn=lambda _answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    first = Environment(
        "a_b",
        task,
        base_work_dir=str(tmp_path),
        task_execution_id="c",
        toolset=Toolset(workspace_factory=None),
    )
    second = Environment(
        "a",
        task,
        base_work_dir=str(tmp_path),
        task_execution_id="b_c",
        toolset=Toolset(workspace_factory=None),
    )
    injected = Environment(
        "task",
        task,
        base_work_dir=str(tmp_path),
        task_execution_id="../../sibling",
        toolset=Toolset(workspace_factory=None),
    )

    paths = {
        Path(first.workspace_path or ""),
        Path(second.workspace_path or ""),
        Path(injected.workspace_path or ""),
    }
    assert len(paths) == 3
    assert all(path.parent == tmp_path.resolve() for path in paths)


def test_existing_workspace_symlink_is_rejected(tmp_path: Path) -> None:
    task = TaskDefinition(
        name="task",
        description="task",
        tools=[],
        scoring_fn=lambda _answer: 1.0,
        submission_format={},
        resolve_answer=False,
    )
    kwargs = {
        "base_work_dir": str(tmp_path),
        "task_execution_id": "execution",
        "toolset": Toolset(workspace_factory=None),
    }
    original = Environment("task", task, **kwargs)
    workspace = Path(original.workspace_path or "")
    workspace.rmdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    workspace.symlink_to(sibling, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        Environment("task", task, **kwargs)
