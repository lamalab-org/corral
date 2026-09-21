from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from stargazer.audit import DEFAULT_DATA_ROOT, reference_submission
from stargazer.env import (
    SUBMISSION_FORMAT,
    StargazerEnvironment,
    _configure_trial,
    _task_prompt,
    create_environments,
)
from stargazer.models import load_task
from stargazer.score import make_stargazer_scorer, score_execution
from stargazer.tools import create_tools

from corral.agents.session import AgentSession
from corral.agents.tool_calling import ToolCallingAgent
from corral.core import Action, ActorRef, AgentStarted, CommitRequest
from corral.core.environment import Toolset
from corral.core.task import TaskDefinition
from corral.evaluation import TaskScorer
from corral.persistence import SQLiteCommitStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def environment(simple_task):
    task = TaskDefinition(
        name="Stargazer test",
        description="Infer the planetary system.",
        tools=["PythonREPL", "submit_action"],
        scoring_fn=make_stargazer_scorer(simple_task),
        state_scoring_fn=score_execution,
        submission_format=SUBMISSION_FORMAT,
        scoring_inputs={"benchmark_task": simple_task},
        prompt_fn=_task_prompt,
        setup_fn=_configure_trial,
        resolve_answer=False,
    )
    return StargazerEnvironment(
        simple_task.task_id,
        task,
        toolset=Toolset(pool=create_tools(), workspace_factory=None),
    )


def _resume_session(environment, state, store):
    return AgentSession(
        environment,
        state,
        actor=ActorRef(kind="agent", actor_id="agent_0", run_id="agent"),
        runtime_actor=ActorRef(kind="runtime", actor_id="corral", run_id="runtime"),
        state_store=store,
        max_iterations=20,
    )


async def _start_session(environment, store):
    async def append(request_id, event):
        head = await store.head("main")
        await store.append(
            CommitRequest(
                request_id=request_id,
                branch_id="main",
                based_on_hash=head.hash if head else None,
                author=ActorRef(kind="runtime", actor_id="corral", run_id="runtime"),
                event=event,
            )
        )

    await append(
        "started",
        await asyncio.to_thread(
            environment.initial_event, execution_id=store.execution_id
        ),
    )
    await append(
        "configured",
        await asyncio.to_thread(environment.configure, await store.materialize("main")),
    )
    await append(
        "agent-started", AgentStarted(agent_run_id="agent", agent_id="agent_0")
    )
    return _resume_session(environment, await store.materialize("main"), store)


async def _execute(session, name, **arguments):
    result = await session.execute(
        Action(id=uuid4().hex, name=name, arguments=arguments, actor_id="agent_0")
    )
    assert result.success, result
    return result.result


def _submission_state(state):
    return state.environment.values["hidden_arguments"]["submission_session"]


def test_official_levels_have_fixed_reference_valid_synthetic_banks(tmp_path):
    levels = {
        level: create_environments(level=level, work_dir=tmp_path / f"level_{level}")
        for level in (1, 2)
    }

    assert {level: len(environments) for level, environments in levels.items()} == {
        1: 10,
        2: 10,
    }
    task_ids = [task_id for environments in levels.values() for task_id in environments]
    assert len(task_ids) == len(set(task_ids)) == 20

    expected_source_counts = {level: {"synthetic": 10} for level in (1, 2)}
    expected_difficulty_counts = {
        1: {5: 4, 6: 3, 7: 3},
        2: {8: 4, 9: 3, 10: 3},
    }
    for level, environments in levels.items():
        source_counts: dict[str, int] = {}
        difficulty_counts: dict[int, int] = {}
        for environment in environments.values():
            task = environment.current_task.scoring_inputs["benchmark_task"]
            source_counts[task.source] = source_counts.get(task.source, 0) + 1
            difficulty_counts[task.truth_difficulty] = (
                difficulty_counts.get(task.truth_difficulty, 0) + 1
            )
        assert source_counts == expected_source_counts[level]
        assert difficulty_counts == expected_difficulty_counts[level]


@pytest.mark.parametrize("level", [3, "3", "real"])
def test_unavailable_levels_are_rejected(tmp_path, level):
    with pytest.raises(ValueError, match="level must be 1 or 2"):
        create_environments(level=level, work_dir=tmp_path)


def test_selected_rv_only_records_keep_observations_and_truth(tmp_path):
    manifest = json.loads(
        (DEFAULT_DATA_ROOT / "selection_manifest.json").read_text(encoding="utf-8")
    )
    rv_only_ids = {
        1: {"seed15_diff5", "seed64_diff6", "seed43_diff7"},
        2: {
            "seed1_diff8",
            "seed17_diff8",
            "seed101_diff9",
            "seed93_diff9",
            "seed82_diff10",
        },
    }
    for level, expected in rv_only_ids.items():
        environments = create_environments(level=level, work_dir=tmp_path / str(level))
        rows = manifest["levels"][str(level)]["tasks"]
        assert {row["task_id"] for row in rows} == set(environments)
        assert expected <= set(environments)
        for task_id in expected:
            path = DEFAULT_DATA_ROOT / "synthetic" / f"{task_id}.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            task = load_task(path, source="synthetic")
            assert raw["meta"]["rv_semantics"].startswith("rv_only")
            # RV-only data must bypass the legacy REBOUND transformation.
            assert asdict(task.observations) == {
                key: tuple(values) for key, values in raw["observations"].items()
            }
            assert task.star_mass_sun == raw["config"]["star"]["M_star_sun"]
            for planet, record in zip(
                task.truth_planets, raw["config"]["planets"], strict=True
            ):
                assert {key: getattr(planet, key) for key in record} == record
            provenance = task.metadata["provenance"]
            assert provenance["source_task_id"] == task_id
            assert provenance["source_paper"] == manifest["source_paper"]
            answer = reference_submission(task, raw).model_dump_json()
            assert environments[task_id].current_task.scoring_fn(answer) == 1.0


async def _ack(session):
    return await _execute(
        session,
        "PythonREPL",
        input_code="print(STARGAZER_SUBMISSION_GUIDE)\n_protocol_guide_ack = True\nprint(_protocol_guide_ack)",
    )


@pytest.mark.anyio
async def test_released_environment_uses_original_workflow_and_completion(tmp_path):
    environment = create_environments(work_dir=tmp_path)["seed15_diff5"]
    task = environment.current_task.scoring_inputs["benchmark_task"]
    assert set(environment.tools) == {"PythonREPL", "submit_action"}
    async with SQLiteCommitStore(tmp_path / "released.sqlite3", "released") as store:
        session = await _start_session(environment.for_task("released"), store)
        state = await store.materialize("main")
        prompt = environment.get_task_prompt(state)
        assert "Lomb-Scargle" in prompt
        assert "MANDATORY MODEL GATING" in prompt
        assert "Corral's configured agent iteration limit" in prompt
        assert "no separate limit on the number of submit_action calls" in prompt
        assert str(task.truth_planets[0].P_days) not in prompt
        assert "truth_planets" not in state.model_dump_json()
        raw = json.loads(
            (DEFAULT_DATA_ROOT / "synthetic/seed15_diff5.json").read_text()
        )
        payload = reference_submission(task, raw).canonical_payload()
        assert "protocol guide not acknowledged" in await _execute(
            session, "submit_action", **payload
        )
        await _ack(session)
        feedback = json.loads(await _execute(session, "submit_action", **payload))
        assert feedback["success"]
        assert feedback["done"]
        submitted = await store.materialize("main")
        assert submitted.submission is None  # Corral's extra closing step remains.
        await _execute(session, "submit_answer", answer="Completed.")
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("main"))
            .score
            == 1
        )


@pytest.mark.anyio
async def test_live_feedback_equals_original_submit_action(
    tmp_path, environment, protocol_reference, assert_protocol_equal
):
    async with SQLiteCommitStore(tmp_path / "feedback.sqlite3", "feedback") as store:
        session = await _start_session(environment, store)
        await _ack(session)
        for step in protocol_reference["submissions"]:
            actual = await _execute(session, "submit_action", **step["payload"])
            if step["output"].startswith("{"):
                assert_protocol_equal(json.loads(actual), json.loads(step["output"]))
            else:
                assert actual == step["output"]
        feedback = json.loads(actual)
        assert feedback["comparison"]["matched_pairs"][0]["components"]
        assert "unmatched_truth" in feedback["matching"]["assignment"]
        assert "count" not in feedback["components"]
        assert _submission_state(await store.materialize("main"))["steps"] == 2


@pytest.mark.anyio
async def test_submissions_are_limited_only_by_agent_iterations(
    tmp_path, environment, exact_submission, monkeypatch
):
    def response(**_kwargs):
        return SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id=uuid4().hex,
                    function=SimpleNamespace(
                        name="submit_action", arguments='{"planets": []}'
                    ),
                )
                for _ in range(2)
            ],
            usage=None,
        )

    model_call = AsyncMock(side_effect=response)
    monkeypatch.setattr("corral.agents.base_agent.llm_call", model_call)
    agent = ToolCallingAgent(
        model="test-model",
        system_prompt="system",
        user_prompt="Task: {{task_guide}}\n{{examples}}\n{{surrender_instructions}}",
    )
    database = tmp_path / "unlimited.sqlite3"
    async with SQLiteCommitStore(database, "unlimited") as store:
        session = await _start_session(environment, store)
        await _ack(session)
        outcome = await agent.run_session(session)

        assert outcome.status == "iteration_limit"
        assert model_call.await_count == session.iteration_limit
        submission = _submission_state(await store.materialize("main"))
        # Multiple submissions per iteration remain usable beyond all old caps.
        assert submission["steps"] == 2 * session.iteration_limit
        assert len(submission["history"]) == submission["steps"]
        assert all(not entry["done"] for entry in submission["history"])
        assert not submission["done"]

    async with SQLiteCommitStore(database, "unlimited") as store:
        session = _resume_session(environment, await store.materialize("main"), store)
        feedback = json.loads(
            await _execute(session, "submit_action", **exact_submission)
        )
        assert feedback["success"]
        assert feedback["done"]
        assert score_execution(await store.materialize("main")) == 1.0


@pytest.mark.anyio
async def test_forced_submission_and_history_survive_restore(tmp_path, environment):
    database = tmp_path / "protocol.sqlite3"
    async with SQLiteCommitStore(database, "protocol") as store:
        session = await _start_session(environment, store)
        await _ack(session)
        output = await _execute(
            session,
            "PythonREPL",
            input_code='print("Gate decision: Kepler=YES; Best_RMS_over_med_sigma=1.09")',
        )
        assert "1.09" in output
    async with SQLiteCommitStore(database, "protocol") as store:
        session = _resume_session(environment, await store.materialize("main"), store)
        assert "Policy gate active" in await _execute(
            session, "PythonREPL", input_code="print(42)"
        )
        rejected = await _execute(session, "submit_action", planets=[{"P_days": -1}])
        assert "rejected" in rejected
        state = _submission_state(await store.materialize("main"))
        assert state["steps"] == 0
        assert len(state["history"]) == 1
        assert not state["force_submit"]
        history = await _execute(
            session, "PythonREPL", input_code="import json\nprint(json.dumps(history))"
        )
        assert json.loads(history) == state["history"]
        await _execute(session, "submit_action", planets=[])
        history = json.loads(
            await _execute(
                session,
                "PythonREPL",
                input_code='import json\nprint(json.dumps(history[-1]["metrics"]["components"]))',
            )
        )
        assert "count" in history  # Upstream history is not the redacted response.


@pytest.mark.anyio
async def test_submission_history_and_success_are_isolated_across_forks(
    tmp_path, environment, exact_submission
):
    database = tmp_path / "fork.sqlite3"
    async with SQLiteCommitStore(database, "fork") as store:
        session = await _start_session(environment, store)
        await _ack(session)
        await _execute(session, "submit_action", planets=[])
        branch = await session.fork_branch(branch_id="alternative")
        passed = json.loads(
            await _execute(session, "submit_action", **exact_submission)
        )
        failed = json.loads(await _execute(branch, "submit_action", planets=[]))
        assert passed["success"]
        assert passed["done"]
        assert not failed["success"]
        assert not failed["done"]
        assert "Stargazer is done" in await _execute(
            session, "PythonREPL", input_code="print(42)"
        )
        assert "Stargazer is done" in await _execute(
            session, "submit_action", **exact_submission
        )
        assert (
            await _execute(branch, "PythonREPL", input_code="print(42)")
        ).strip() == "42"
        retried = json.loads(await _execute(branch, "submit_action", planets=[]))
        assert not retried["done"]
        assert _submission_state(await store.materialize("main"))["steps"] == 2
        assert _submission_state(await store.materialize("alternative"))["steps"] == 3
        await _execute(branch, "submit_answer", answer=json.dumps(exact_submission))
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("alternative"))
            .score
            == 0
        )
    async with SQLiteCommitStore(database, "fork") as store:
        restored = _resume_session(environment, await store.materialize("main"), store)
        await _execute(restored, "submit_answer", answer="Not a candidate JSON.")
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("main"))
            .score
            == 1
        )


@pytest.mark.anyio
async def test_final_answer_cannot_bypass_stargazer_submissions(
    tmp_path, environment, exact_submission
):
    async with SQLiteCommitStore(tmp_path / "bypass.sqlite3", "bypass") as store:
        session = await _start_session(environment, store)
        await _execute(session, "submit_answer", answer=json.dumps(exact_submission))
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("main"))
            .score
            == 0
        )


@pytest.mark.anyio
async def test_python_state_and_protocol_are_isolated_between_trials_and_forks(
    tmp_path, environment
):
    database = tmp_path / "analysis.sqlite3"
    async with (
        SQLiteCommitStore(database, "analysis") as store,
        SQLiteCommitStore(database, "other") as other_store,
    ):
        session = await _start_session(environment.for_task("analysis"), store)
        other = await _start_session(environment.for_task("other"), other_store)
        await _ack(session)
        await _execute(
            session,
            "PythonREPL",
            input_code="values = np.arange(3.0)\ndef shifted():\n    return values + 2",
        )
        branch = await session.fork_branch(branch_id="alternative")
        result = await _execute(
            branch, "PythonREPL", input_code="values[0] = 40\nprint(shifted().tolist())"
        )
        assert json.loads(result) == [42, 3, 4]
        assert "NameError" in await _execute(other, "PythonREPL", input_code="values")
        assert "protocol guide not acknowledged" in await _execute(
            other, "submit_action", planets=[]
        )
    async with SQLiteCommitStore(database, "analysis") as store:
        session = _resume_session(
            environment.for_task("analysis"), await store.materialize("main"), store
        )
        assert json.loads(
            await _execute(
                session, "PythonREPL", input_code="print(shifted().tolist())"
            )
        ) == [2, 3, 4]
        assert (
            json.loads(await _execute(session, "submit_action", planets=[]))["success"]
            is False
        )


@pytest.mark.anyio
async def test_submit_gate_waits_for_the_protocol_acknowledgement(tmp_path, environment):
    database = tmp_path / "gate_order.sqlite3"
    async with SQLiteCommitStore(database, "gate_order") as store:
        session = await _start_session(environment, store)
        # Step 2 of the prompt asks for exactly this line, and an agent can
        # reach it before the step 0 acknowledgement.
        await _execute(
            session,
            "PythonREPL",
            input_code='print("Gate decision: Kepler=YES; Best_RMS_over_med_sigma=1.09")',
        )
        assert _submission_state(await store.materialize("main"))["force_submit"]

        # submit_action stays blocked until the guide is acknowledged, so
        # blocking the REPL too would leave the trial with no way forward.
        assert "protocol guide not acknowledged" in await _execute(
            session, "submit_action", planets=[]
        )
        assert "42" in await _execute(session, "PythonREPL", input_code="print(42)")

        # Once acknowledged, the gate applies as designed.
        await _ack(session)
        assert "Policy gate active" in await _execute(
            session, "PythonREPL", input_code="print(42)"
        )
