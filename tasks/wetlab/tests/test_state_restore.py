from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import anyio
import pytest
from wetlab.engine import ChemicalSystemSpec, WetlabEngine, WetlabState
from wetlab.env import QualitativeAnalysisEnvironment, QualitativeAnalysisTask
from wetlab.tools import create_tools

from corral.agents.session import AgentSession
from corral.core import Action, ActorRef, AgentStarted, CommitRequest, ExecutionState
from corral.core.environment import Toolset
from corral.core.task import InputRef
from corral.orchestration.registry import RuntimeRegistry
from corral.persistence import SQLiteCommitStore, WorkspaceManager
from corral.runtime.environment_loader import load_environment_group


@pytest.fixture()
def anyio_backend():
    return "asyncio"


async def _start_session(environment, store, *, dependency_outputs=None):
    execution_id = store.execution_id
    runtime = ActorRef(kind="runtime", actor_id="corral", run_id="runtime")
    actor = ActorRef(kind="agent", actor_id="agent_0", run_id="agent")

    async def append(request_id, event):
        head = await store.head("main")
        await store.append(
            CommitRequest(
                request_id=request_id,
                branch_id="main",
                based_on_hash=head.hash if head else None,
                author=runtime,
                event=event,
            )
        )

    await append(
        "started",
        await anyio.to_thread.run_sync(
            partial(
                environment.initial_event,
                execution_id=execution_id,
                dependency_outputs=dependency_outputs,
            )
        ),
    )
    await append(
        "configured",
        await anyio.to_thread.run_sync(
            environment.configure, await store.materialize("main")
        ),
    )
    await append(
        "agent-started",
        AgentStarted(agent_run_id=actor.run_id, agent_id=actor.actor_id),
    )
    return AgentSession(
        environment,
        await store.materialize("main"),
        actor=actor,
        runtime_actor=runtime,
        state_store=store,
        max_iterations=10,
    )


def _environment(work_dir: str = "") -> QualitativeAnalysisEnvironment:
    task = QualitativeAnalysisTask(
        name="wetlab-state-test",
        description="Run a deterministic wetlab experiment.",
        tools=["mix_two_solutions", "measure_pH"],
        scoring_fn=lambda _answer: 1.0,
        submission_format="number",
        task_sys="K S(+6)",
        sample_list=[
            {
                "label": "sample",
                "vol": 20,
                "composition": {"K+": 0.1, "HSO4-": 0.1},
            }
        ],
        reagent_set="Basic",
        additional_reagents=[],
    )
    return QualitativeAnalysisEnvironment(
        "wetlab-state-test",
        task,
        base_work_dir=work_dir,
        toolset=Toolset(
            pool=create_tools(),
            workspace_factory=None,
        ),
        group_tasks={"wetlab-state-test": task},
    )


def _wetlab_payload(state: ExecutionState) -> dict:
    return dict(state.environment.values["hidden_arguments"]["wetlab"])


def test_loader_binds_wetlab_scratch_per_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CORRAL_WORK_DIR", str(tmp_path))
    environments = load_environment_group("wetlab", env_kwargs={"level": 1})
    task_id = "qualysis_lvl1_01"
    template = environments[task_id]
    assert template.base_work_dir == str(tmp_path)
    assert template.workspace_path is None
    registry = RuntimeRegistry(agents={}, environments=environments)
    try:
        first = registry.environment(task_id, "trial-1")
        second = registry.environment(task_id, "trial-2")
        assert first.workspace_path != second.workspace_path
        for environment in (first, second):
            assert isinstance(environment, QualitativeAnalysisEnvironment)
            assert Path(environment.workspace_path).parent == tmp_path
            assert environment.toolset.workspace_factory is None
            assert environment.tools.keys() == template.tools.keys()
            assert "write_file" not in environment.tools
        Path(first.workspace_path, "notes.txt").write_text("scratch only")
        assert list(Path(second.workspace_path).iterdir()) == []
    finally:
        registry.close()


def test_structured_inventory_round_trip_is_exact() -> None:
    spec = ChemicalSystemSpec(elements="Ag Cl N")
    engine = WetlabEngine(spec)
    silver_nitrate = engine.stock_solution(
        {"Ag+": 0.1, "NO3-": 0.1},
        description="AgNO3 0.1 M",
    )
    hydrochloric_acid = engine.stock_solution(
        {"H+": 0.1, "Cl-": 0.1},
        description="HCl 0.1 M",
    )
    test_solution = 5 * silver_nitrate + 5 * hydrochloric_acid
    test_solution.equilibrate()
    assert test_solution.has_precipitate
    original = engine.snapshot(
        {
            "AgNO3": silver_nitrate,
            "HCl": hydrochloric_acid,
            "test": test_solution,
        }
    )

    serialized = original.to_json()
    restored_state = WetlabState.from_dict(json.loads(serialized))
    restored_engine = WetlabEngine(restored_state.chemical_system)
    restored_inventory = restored_engine.restore(restored_state)

    assert restored_engine.snapshot(restored_inventory).to_json() == serialized


def test_incompatible_chemistry_checkpoint_fails_loudly() -> None:
    incompatible = replace(
        ChemicalSystemSpec(elements="K S(+6)"),
        reaktoro_version="incompatible-test-version",
    )

    with pytest.raises(RuntimeError, match="Reaktoro version does not match"):
        WetlabEngine(incompatible)


@pytest.mark.anyio()
async def test_action_replay_and_checkpoint_restore_are_deterministic(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    private = tmp_path / "corral-state"
    private.mkdir(mode=0o700)
    manager = WorkspaceManager(artifact_root=private / "artifacts")
    environment = _environment(str(workspace_root)).for_task("wetlab-replay")
    environment.workspace_manager = manager
    Path(environment.workspace_path, "notes.txt").write_text("scratch only")
    action = Action(
        id="mix-1",
        name="mix_two_solutions",
        arguments={
            "test_label": "sample_koh",
            "sol1_label": "sample",
            "sol1_vol": 2,
            "sol2_label": "KOH",
            "sol2_vol": 1,
        },
        actor_id="agent_0",
    )

    database = private / "replay.sqlite3"
    async with SQLiteCommitStore(database, "wetlab-replay") as store:
        session = await _start_session(environment, store)
        initial = _wetlab_payload(await store.materialize("main"))
        replay_session = await session.fork_branch(branch_id="replay")
        first_result = await session.execute(action)
        replay_result = await replay_session.execute(
            Action(**{**action.model_dump(), "id": "mix-replay"})
        )
        assert first_result.success and replay_result.success
        assert first_result.result == replay_result.result
        first = await store.materialize("main")
        replay = await store.materialize("replay")
        assert _wetlab_payload(first) == _wetlab_payload(replay)
        assert _wetlab_payload(first) != initial
        assert set(first.workspace.files) == {"notes.txt"}
        assert {path.name for path in Path(environment.workspace_path).iterdir()} == {
            "notes.txt"
        }

    # Reopen the durable commit store and reconstruct a fresh environment/session.
    # Continuation must use the inventory produced by the committed mix.
    continuation = Action(
        id="measure-2",
        name="measure_pH",
        arguments={"label": "sample_koh"},
        actor_id="agent_0",
    )
    async with SQLiteCommitStore(database, "wetlab-replay") as restored_store:
        checkpoint = await restored_store.materialize("main")
        assert checkpoint == first
        restored_environment = _environment(
            str(tmp_path / "restored-workspace")
        ).for_task("wetlab-replay")
        restored_environment.workspace_manager = manager
        await manager.materialize(
            checkpoint.workspace, restored_environment.workspace_path
        )
        assert (
            Path(restored_environment.workspace_path, "notes.txt").read_text()
            == "scratch only"
        )
        restored_session = AgentSession(
            restored_environment,
            checkpoint,
            actor=session.actor,
            runtime_actor=session.runtime_actor,
            state_store=restored_store,
            max_iterations=10,
        )
        result = await restored_session.execute(continuation)
        assert result.success
        continued = await restored_store.materialize("main")
        assert _wetlab_payload(continued) == _wetlab_payload(checkpoint)
        assert set(continued.workspace.files) == {"notes.txt"}


@pytest.mark.anyio()
async def test_chained_task_receives_inventory_through_dependency_output(
    tmp_path,
) -> None:
    upstream_environment = _environment()
    submit = Action(
        id="submit-1",
        name="submit_answer",
        arguments={"answer": "K+"},
        actor_id="agent_0",
    )
    async with SQLiteCommitStore(tmp_path / "upstream.sqlite3", "upstream") as store:
        session = await _start_session(upstream_environment, store)
        result = await session.execute(submit)
        assert result.success
        submitted = await store.materialize("main")
    output = upstream_environment.get_task_output(submitted)
    assert output is not None

    upstream_task = upstream_environment.current_task
    downstream_task = QualitativeAnalysisTask(
        name="wetlab-state-test-next",
        description="Continue the experiment.",
        tools=["measure_pH"],
        scoring_fn=lambda _answer: 1.0,
        submission_format="number",
        input_map={"wetlab-state-test": InputRef("wetlab-state-test")},
        task_sys="K S(+6)",
        sample_list=[
            {
                "label": "sample",
                "vol": 20,
                "composition": {"K+": 0.1, "HSO4-": 0.1},
            }
        ],
        reagent_set="Basic",
        additional_reagents=[],
        initial=False,
    )
    downstream_environment = QualitativeAnalysisEnvironment(
        "wetlab-state-test-next",
        downstream_task,
        toolset=Toolset(pool=create_tools(), workspace_factory=None),
        group_tasks={
            "wetlab-state-test": upstream_task,
            "wetlab-state-test-next": downstream_task,
        },
    )
    async with SQLiteCommitStore(
        tmp_path / "downstream.sqlite3", "downstream"
    ) as store:
        await _start_session(
            downstream_environment,
            store,
            dependency_outputs={"wetlab-state-test": output},
        )
        configured = await store.materialize("main")

    assert _wetlab_payload(configured) == output.metadata["wetlab"]
