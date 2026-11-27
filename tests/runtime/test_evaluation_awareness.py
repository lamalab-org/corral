"""Evaluation notices travel through the normal task/session boundary."""

import json
from dataclasses import asdict
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter

from corral.agents.schema import AgentOutcome
from corral.core.action import submit_answer_action
from corral.core.environment import Environment, Toolset
from corral.core.task import TaskDefinition
from corral.observability import NoOpObserver
from corral.orchestration import RunTaskInput, RuntimeRegistry
from corral.persistence import SQLiteCommitStore
from corral.run import CorralRunner
from corral.runtime import TaskRuntime
from corral.runtime.agent_worker import _snapshot
from corral.runtime.evaluation_awareness import EVALUATION_AWARENESS_TEMPLATE


@pytest.fixture
def anyio_backend():
    return "asyncio"


def environment(prompt):
    return Environment(
        "notice-task",
        TaskDefinition(
            name="notice-task",
            description="Solve the task.",
            tools=[],
            prompt_fn=lambda _env, _state: prompt,
            scoring_fn=lambda answer: 1.0,
            submission_format={},
            resolve_answer=False,
        ),
        toolset=Toolset(workspace_factory=None),
    )


class PromptProbe:
    def __init__(self, original, mode):
        self.original = original
        self.mode = mode

    async def run_session(self, session):
        delivered = session.prompt
        assert delivered == session.prompt  # reading again must not add another notice
        assert session.state.task.metadata["prompt"] == self.original
        assert _snapshot(session)["prompt"] == delivered
        if self.mode == "off":
            assert delivered == self.original
        else:
            notice = EVALUATION_AWARENESS_TEMPLATE.format(
                iteration_limit=session.iteration_limit
            )
            if isinstance(self.original, str):
                assert delivered == f"{notice}\n\n{self.original}"
            else:
                assert delivered == [{"type": "text", "text": notice}, *self.original]
        self.scaffold = dict(session.state.task.scaffold)
        await session.execute(submit_answer_action("done"))
        return AgentOutcome(status="completed", answer="done")


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["off", "on"])
@pytest.mark.parametrize("budget", [None, 7])
@pytest.mark.parametrize(
    "prompt",
    [
        "Original task instructions.",
        [
            {"type": "text", "text": "Inspect this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    ],
)
async def test_runner_delivers_notice_without_changing_task_prompt(
    tmp_path, mode, budget, prompt
):
    agent = PromptProbe(prompt, mode)
    env = environment(prompt)
    registry = RuntimeRegistry(
        agents={"agent": agent}, environments={"notice-task": env}
    )
    async with SQLiteCommitStore(tmp_path / "run.sqlite3") as store:
        runner = CorralRunner(
            registry,
            environments={"notice-task": env},
            max_iterations=budget,
            state_store=store,
            observer=NoOpObserver(),
        )
        report = await runner.run(
            "awareness", evaluation_awareness=mode, evaluate=False
        )
        assert report.all_results[0].error_message is None
        metadata = report.metadata["benchmark"]
        assert metadata["evaluation_awareness"] == mode
        assert agent.scaffold["evaluation_awareness"] == mode
        expected = EVALUATION_AWARENESS_TEMPLATE.format(iteration_limit=budget or 50)
        assert agent.scaffold["evaluation_awareness_notice"] == (
            expected if mode == "on" else None
        )
    registry.close()


class BudgetProbe:
    def __init__(self, expected, child=False):
        self.expected = expected
        self.child = child

    async def run_session(self, session):
        prompt = session.prompt
        notice = EVALUATION_AWARENESS_TEMPLATE.format(iteration_limit=self.expected)
        assert prompt.count(notice) == 1
        assert f"iteration limit is {self.expected}." in prompt
        if self.child:
            assert "Subagent handoff:" in prompt
            return AgentOutcome(status="completed", answer="child done")
        child = await session.spawn_subagent(
            BudgetProbe(3, child=True), handoff="Investigate.", max_iterations=3
        )
        outcome = await session.wait_for_subagent(child)
        assert outcome.status == "completed"
        delegate = await session.run_delegate(DelegateProbe(), max_iterations=4)
        assert delegate.status == "iteration_limit"
        assert "iteration limit is 9." in session.prompt
        await session.execute(submit_answer_action("done"))
        return AgentOutcome(status="completed", answer="done")


class DelegateProbe:
    async def run_session(self, session):
        assert "iteration limit is 4." in session.prompt
        return AgentOutcome(status="iteration_limit", error="delegate probe stopped")


@pytest.mark.anyio
async def test_children_and_delegates_get_their_effective_budgets(tmp_path):
    async with SQLiteCommitStore(tmp_path / "children.sqlite3") as store:
        state = await TaskRuntime(store, NoOpObserver()).run(
            BudgetProbe(9),
            environment("Original instructions."),
            execution_id="children",
            max_iterations=9,
            started_at=datetime.now(timezone.utc),
            scaffold_metadata={"evaluation_awareness": "on"},
        )
        assert state.submission == "done", state.runtime.metadata


@pytest.mark.anyio
async def test_resuming_same_trial_cannot_change_the_condition(tmp_path):
    env = environment("Original instructions.")
    async with SQLiteCommitStore(tmp_path / "resume.sqlite3") as store:
        runtime = TaskRuntime(store, NoOpObserver())
        kwargs = {
            "execution_id": "resume",
            "max_iterations": 7,
            "started_at": datetime.now(timezone.utc),
        }
        await runtime.run(
            PromptProbe("Original instructions.", "on"),
            env,
            scaffold_metadata={"evaluation_awareness": "on"},
            **kwargs,
        )
        with pytest.raises(ValueError, match="start a new trial or continuation"):
            await runtime.run(
                PromptProbe("Original instructions.", "off"),
                env,
                scaffold_metadata={"evaluation_awareness": "off"},
                **kwargs,
            )


@pytest.mark.parametrize("mode", ["off", "on"])
def test_docker_request_roundtrip_keeps_condition(mode):
    request = RunTaskInput(
        execution_id="trial",
        task_id="task",
        environment_id="env",
        agent_id="agent",
        evaluation_awareness=mode,
    )
    restored = TypeAdapter(RunTaskInput).validate_json(json.dumps(asdict(request)))
    assert restored.evaluation_awareness == mode


@pytest.mark.parametrize("value", ["", "yes", True, None])
def test_invalid_awareness_is_rejected(value):
    with pytest.raises(ValueError, match="evaluation_awareness"):
        RunTaskInput(
            execution_id="trial",
            task_id="task",
            environment_id="env",
            agent_id="agent",
            evaluation_awareness=value,
        )


class DockerAwarenessProbe:
    def __init__(self, mode, guardrails):
        self.mode = mode
        self.guardrails = guardrails

    async def run_session(self, session):
        from pathlib import Path

        prompt = session.prompt
        assert isinstance(prompt, list)
        assert prompt[-1]["type"] == "image_url"
        notice = EVALUATION_AWARENESS_TEMPLATE.format(iteration_limit=11)
        notices = [
            part
            for part in prompt
            if part.get("type") == "text" and notice in part["text"]
        ]
        assert len(notices) == (1 if self.mode == "on" else 0)
        if notices:
            assert "iteration limit is 11." in notices[0]["text"]
        try:
            Path("/opt/corral/pyproject.toml").read_text()
            readable = True
        except (PermissionError, FileNotFoundError):
            readable = False
        assert readable == (self.guardrails == "off")
        await session.execute(submit_answer_action("done"))
        return AgentOutcome(status="completed", answer="done")


def create_docker_registry(request):
    import os

    return RuntimeRegistry(
        agents={
            "probe": DockerAwarenessProbe(
                request.evaluation_awareness, os.environ["CORRAL_TEST_GUARDRAILS"]
            )
        },
        environments={
            "probe": Environment(
                "probe",
                TaskDefinition(
                    name="probe",
                    description="Inspect the task.",
                    tools=[],
                    prompt_fn=lambda _env, _state: [
                        {"type": "text", "text": "Original task instructions."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                    scoring_fn=lambda answer: 1.0,
                    submission_format={},
                    resolve_answer=False,
                ),
                base_work_dir="/workspace",
                task_execution_id="probe",
            )
        },
    )


@pytest.mark.anyio
async def test_real_docker_awareness_is_independent_of_guardrails():
    import os
    from pathlib import Path

    if os.environ.get("CORRAL_EVALUATION_TESTS") != "1":
        pytest.skip("requires a fresh prepared Docker container")
    from corral.orchestration import DockerSandboxSpec, SandboxProfile
    from corral.orchestration.internal import run_task_from_files

    guardrails = os.environ["CORRAL_TEST_GUARDRAILS"]
    # The standalone PR works on main; the combination is checked after merging
    # the independent guardrail PR into a disposable validation worktree.
    options = (
        {"guardrails": guardrails}
        if "guardrails" in DockerSandboxSpec.__dataclass_fields__
        else {}
    )
    assert guardrails == "on" or options, "guardrails off requires the guardrail PR"
    request = RunTaskInput(
        execution_id="awareness-probe",
        task_id="probe",
        environment_id="probe",
        agent_id="probe",
        max_iterations=11,
        evaluation_awareness=os.environ["CORRAL_TEST_AWARENESS"],
        sandbox=SandboxProfile(
            mode="docker",
            docker=DockerSandboxSpec(
                registry_module="tests.runtime.test_evaluation_awareness:create_docker_registry",
                **options,
            ),
        ),
    )
    root = Path("/corral-state")
    root.mkdir(mode=0o700, exist_ok=True)
    request_file = root / "request.json"
    request_file.write_text(json.dumps(asdict(request)))
    result_file = root / "result.json"
    assert await run_task_from_files(request_file, result_file) == 0
    result = json.loads(result_file.read_text())
    assert result["submission"] == "done", result
