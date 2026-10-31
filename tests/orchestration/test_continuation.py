"""Independent trials must restore the selected point, never the parent's head."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from corral.agents.schema import AgentOutcome
from corral.core import Action
from corral.core.environment import Environment, EnvironmentSetup
from corral.core.task import TaskDefinition
from corral.observability import NoOpObserver
from corral.orchestration import ContinuationSource, RunTaskInput, RuntimeRegistry
from corral.persistence import ShardedCommitStore
from corral.persistence.continuation import import_checkpoint, read_checkpoint
from corral.run import CorralRunner, execute_task


@pytest.fixture
def anyio_backend():
    return "asyncio"


class CheckpointAgent:
    def __init__(self):
        self.seen = []

    async def run_session(self, session):
        if session.state.runtime.metadata.get("lineage"):
            self.seen.append((session.state, session.messages, session.workspace))
            assert session.previous_state is None
            assert session.previous_evaluation is None
            text = session.execution_id
        else:
            await session.record_message({"role": "user", "content": "original prompt"})
            await session.set_agent_state("algorithm", {"iteration": 3})
            written = await session.execute(
                Action(
                    name="write_file",
                    arguments={"path": "/workspace/note.txt", "content": "before"},
                )
            )
            assert written.success
            self.checkpoint = session.state
            text = "after"
        written = await session.execute(
            Action(
                name="write_file",
                arguments={"path": "/workspace/note.txt", "content": text},
            )
        )
        assert written.success
        submitted = await session.execute(
            Action(name="submit_answer", arguments={"answer": "finished"})
        )
        assert submitted.success
        return AgentOutcome(status="completed", answer="finished")


@pytest.fixture
async def trial(tmp_path):
    setup_calls = []

    def setup(_environment, _state):
        setup_calls.append(True)
        return EnvironmentSetup(hidden_arguments={"budget_used": 7 + len(setup_calls)})

    environment = Environment(
        "task",
        TaskDefinition(
            name="task",
            description="write notes",
            tools=[],
            scoring_fn=lambda _: 1.0,
            submission_format={"answer": "string"},
            resolve_answer=False,
            allow_previous_attempt_context=True,
            setup_fn=setup,
        ),
        base_work_dir=str(tmp_path / "workspaces"),
    )
    store = ShardedCommitStore(tmp_path / "state")
    agent = CheckpointAgent()
    registry = RuntimeRegistry(
        agents={"agent": agent},
        environments={"task": environment},
        workspace_manager_factory=store.workspace_manager,
    )
    request = RunTaskInput(
        execution_id="parent",
        task_id="task",
        environment_id="task",
        agent_id="agent",
        max_iterations=10,
    )
    final = await execute_task(
        task=request, state_store=store, registry=registry, observer=NoOpObserver()
    )
    assert final.status == "submitted", final.error
    source = ContinuationSource(
        trial=str(store.execution_dir("parent")),
        commit_hash=agent.checkpoint.through_commit_hash,
        message="Try again.",
    )
    await store.close_execution("parent")
    yield store, registry, agent, request, source, final
    registry.close()
    await store.aclose()


def fingerprints(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.anyio
async def test_four_independent_children_restore_history_and_leave_source_unchanged(
    trial,
):
    store, registry, agent, request, source, _ = trial
    before = fingerprints(Path(source.trial))
    runner = CorralRunner(
        registry,
        environments={"task": registry._environments["task"]},
        agent_id="agent",
        state_store=store,
        observer=NoOpObserver(),
    )
    report = await runner.run(
        "children",
        task_ids=["task"],
        trials_per_task=4,
        evaluate=False,
        continuation_by_task={"task": source},
    )
    assert all(result.error_message is None for result in report.all_results)
    assert len(agent.seen) == 4
    for state, messages, workspace in agent.seen:
        run = next(
            run for run in state.agent_runs.values() if run.parent_run_id is None
        )
        assert run.run_id not in agent.checkpoint.agent_runs
        assert run.algorithm_state == {"algorithm": {"iteration": 3}}
        assert messages[:-1] == next(iter(agent.checkpoint.conversations.values()))
        assert messages[-1] == {"role": "user", "content": "Try again."}
        assert state.environment == agent.checkpoint.environment
        assert state.workspace.files == agent.checkpoint.workspace.files
        assert state.runtime.metadata[
            "inherited_usage"
        ] == agent.checkpoint.usage.model_dump(mode="json")
        assert state.usage.tool_calls == 0
        assert Path(workspace, "note.txt").read_text() == state.execution_id
    assert len({workspace for _, _, workspace in agent.seen}) == 4
    assert fingerprints(Path(source.trial)) == before


@pytest.mark.anyio
async def test_import_is_durable_without_source_and_message_is_not_duplicated(trial):
    store, registry, agent, request, source, _ = trial
    child = replace(request, execution_id="child", continuation=source)
    await import_checkpoint(child, store, store.workspace_manager("child"))
    shutil.move(source.trial, source.trial + "-moved")
    result = await execute_task(
        task=child, state_store=store, registry=registry, observer=NoOpObserver()
    )
    repeated = await execute_task(
        task=child, state_store=store, registry=registry, observer=NoOpObserver()
    )
    assert result.commit_hash == repeated.commit_hash
    state = await store.for_execution("child").materialize("main")
    assert (
        sum(
            message.get("content") == "Try again."
            for messages in state.conversations.values()
            for message in messages
        )
        == 1
    )
    assert len(agent.seen) == 1
    with pytest.raises(Exception, match="different checkpoint"):
        await import_checkpoint(
            replace(child, continuation=replace(source, message="changed")),
            store,
            store.workspace_manager("child"),
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "boundary",
    [
        "execution.started",
        "task.configured",
        "agent.started",
        "agent.turn_recorded",
        "tool.started",
        "submission.accepted",
        "execution.completed",
    ],
)
async def test_continuation_at_lifecycle_boundaries(trial, boundary):
    store, registry, agent, request, source, _ = trial
    parent = store.for_execution("parent")
    commits = [commit async for commit in parent.iter_commits("main")]
    selected = next(commit for commit in commits if commit.event.type == boundary)
    await store.close_execution("parent")
    child = replace(
        request,
        execution_id="boundary-child",
        continuation=replace(source, commit_hash=selected.hash),
    )
    result = await execute_task(
        task=child, state_store=store, registry=registry, observer=NoOpObserver()
    )
    assert result.status == "submitted", result.error
    assert len(agent.seen) == 1
    assert agent.seen[0][0].submission is None


@pytest.mark.anyio
async def test_missing_blob_and_wrong_task_fail_before_publishing_child(trial):
    store, registry, agent, request, source, _ = trial
    child = replace(request, execution_id="bad", continuation=source)
    with pytest.raises(ValueError, match="task does not match"):
        await import_checkpoint(
            replace(child, task_id="other"), store, store.workspace_manager("bad")
        )
    ref = next(iter(agent.checkpoint.workspace.files.values()))
    Path(source.trial, "artifacts", ref.sha256[:2], ref.sha256).unlink()
    with pytest.raises(Exception, match="was not found"):
        await import_checkpoint(child, store, store.workspace_manager("bad"))
    assert await store.for_execution("bad").head("main") is None


@pytest.mark.anyio
async def test_missing_commit_is_not_replaced_by_latest(trial):
    _, _, _, _, source, _ = trial
    with pytest.raises(Exception, match="does not belong"):
        await read_checkpoint(replace(source, commit_hash="0" * 64))


@pytest.mark.anyio
async def test_reusing_source_identity_does_not_even_overwrite_request(trial):
    store, registry, _, request, source, _ = trial
    before = fingerprints(Path(source.trial))
    with pytest.raises(ValueError, match="new execution ID"):
        await execute_task(
            task=replace(request, continuation=source),
            state_store=store,
            registry=registry,
        )
    assert fingerprints(Path(source.trial)) == before


@pytest.mark.anyio
async def test_completed_submission_is_accepted_without_reexecuting_tool(trial):
    store, registry, agent, request, source, _ = trial
    commits = [c async for c in store.for_execution("parent").iter_commits("main")]
    cutoff = next(
        c
        for c in commits
        if c.event.type == "tool.completed"
        and c.event.action_id
        == next(
            a.event.action_id for a in commits if a.event.type == "submission.accepted"
        )
    )
    child = replace(
        request,
        execution_id="submission-child",
        continuation=replace(source, commit_hash=cutoff.hash),
    )
    result = await execute_task(
        task=child, state_store=store, registry=registry, observer=NoOpObserver()
    )
    assert result.status == "submitted"
    assert agent.seen == []
    child_commits = [
        c async for c in store.for_execution(child.execution_id).iter_commits("main")
    ]
    assert not any(c.event.type == "tool.started" for c in child_commits)


@pytest.mark.anyio
async def test_cli_continuation_inherits_runtime_configuration(trial, monkeypatch):
    from dataclasses import asdict

    from corral import cli
    from corral.orchestration import (
        AgentRuntimeDefinition,
        EnvironmentRuntimeDefinition,
    )

    _, _, _, request, source, _ = trial
    saved = replace(
        request,
        agent_runtime=AgentRuntimeDefinition(
            name="tool-calling",
            model="test-model",
            options={"system_prompt": "saved prompt"},
        ),
        environment_runtime=EnvironmentRuntimeDefinition(
            name="psychometrics", options={"level": 1}
        ),
    )
    Path(source.trial, "launch-request.json").write_text(json.dumps(asdict(saved)))
    seen = []

    async def launch(args):
        seen.append(args)
        return 0

    monkeypatch.setattr(cli, "run_benchmark", launch)
    args = cli.build_parser().parse_args(
        [
            "continue",
            "--source-trial",
            source.trial,
            "--commit",
            source.commit_hash,
            "--trials",
            "4",
            "--sandbox",
            "local",
            "--message",
            "feedback",
        ]
    )
    assert await args._handler(args) == 0
    assert seen[0].agent == "tool-calling"
    assert seen[0].environment == "psychometrics"
    assert seen[0].env_kwargs == {"level": 1}
    assert seen[0].agent_kwargs == {"system_prompt": "saved prompt"}
    assert seen[0].continuation_by_task["task"].message == "feedback"
    assert seen[0].trials == 4


@pytest.mark.anyio
async def test_nonterminal_branch_of_finished_trial_can_be_selected(trial):
    store, _, _, _, source, _ = trial
    parent = store.for_execution("parent")
    await parent.create_branch(branch_id="experiment", from_hash=source.commit_hash)
    await store.close_execution("parent")
    event = await read_checkpoint(replace(source, branch_id="experiment"))
    assert event.source.branch_id == "experiment"
    assert event.source.through_commit_hash == source.commit_hash


@pytest.mark.anyio
@pytest.mark.parametrize("live", ["subagent", "job"])
async def test_live_process_state_is_rejected_before_import(trial, live):
    from corral.core import (
        ActorRef,
        AgentSpawned,
        CommitRequest,
        EnvironmentOperation,
        TaskConfigured,
    )

    store, _, agent, _, source, _ = trial
    parent = store.for_execution("parent")
    await parent.create_branch(branch_id="live", from_hash=source.commit_hash)
    if live == "subagent":
        run = next(iter(agent.checkpoint.agent_runs.values()))
        actor = ActorRef(kind="agent", actor_id=run.actor_id, run_id=run.run_id)
        event = AgentSpawned(
            child_run_id="live-child",
            child_actor_id="child",
            context_cutoff_hash=source.commit_hash,
            handoff="Continue this analysis.",
        )
    else:
        actor = ActorRef(kind="runtime", actor_id="corral", run_id="runtime:parent")
        event = TaskConfigured(
            status="job started",
            environment_operations=(
                EnvironmentOperation(
                    operation="set",
                    path=("jobs",),
                    value={"job": {"status": "running"}},
                ),
            ),
            expected_environment_revision=agent.checkpoint.environment.revision,
        )
    commit = await parent.bind(actor, branch_id="live").append(
        CommitRequest(
            request_id="live",
            branch_id="live",
            based_on_hash=source.commit_hash,
            author=actor,
            event=event,
        )
    )
    await store.close_execution("parent")
    with pytest.raises(ValueError, match="active subagents|live background jobs"):
        await read_checkpoint(
            replace(source, branch_id="live", commit_hash=commit.hash)
        )


@pytest.mark.anyio
async def test_cli_rejects_overlapping_source_and_output_before_writing(
    trial, monkeypatch, tmp_path
):
    from corral import cli

    _, registry, agent, _, source, _ = trial
    args = cli.build_parser().parse_args(
        [
            "bench",
            "--agent",
            "tool-calling",
            "--environment",
            "samplemath",
            "--task",
            "task",
            "--sandbox",
            "local",
            "--run-id",
            "overlap",
            "--output-dir",
            source.trial,
        ]
    )
    args.continuation_by_task = {"task": source}
    before = fingerprints(Path(source.trial))
    monkeypatch.setattr(
        cli, "load_environment_group", lambda *_args, **_kwargs: registry._environments
    )
    monkeypatch.setattr(
        cli,
        "_configured_agent",
        lambda *_args, **_kwargs: ("tool-calling", agent, "test"),
    )
    with pytest.raises(ValueError, match="separate output run directory"):
        await cli.run_benchmark(args)
    assert fingerprints(Path(source.trial)) == before
