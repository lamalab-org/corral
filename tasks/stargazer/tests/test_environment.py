from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
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
from stargazer.score import make_stargazer_scorer
from stargazer.tools import create_tools

from corral.agents.session import AgentSession
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
        tools=["PythonREPL", "validate_fit"],
        scoring_fn=make_stargazer_scorer(simple_task),
        allow_previous_attempt_context=False,
        execution_version="test-blind-v1",
        submission_format=SUBMISSION_FORMAT,
        scoring_inputs={
            "benchmark_task": simple_task,
            "public_context": simple_task.public_fit_context(),
            "development_mode": True,
        },
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
    return state.environment.values["hidden_arguments"]["analysis"]


def test_official_levels_have_fixed_reference_valid_synthetic_banks(tmp_path):
    levels = {
        level: create_environments(
            development_mode=True, level=level, work_dir=tmp_path / f"level_{level}"
        )
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
        create_environments(development_mode=True, level=level, work_dir=tmp_path)


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
        environments = create_environments(
            development_mode=True, level=level, work_dir=tmp_path / str(level)
        )
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
async def test_blind_tools_and_final_candidate_contract(
    tmp_path, environment, exact_submission
):
    assert set(environment.tools) == {"PythonREPL", "validate_fit"}
    async with SQLiteCommitStore(tmp_path / "blind.sqlite3", "blind") as store:
        session = await _start_session(environment, store)
        prompt = environment.get_task_prompt(await store.materialize("main"))
        assert "validate_fit" in prompt
        assert "submit_answer" in prompt
        assert "submit_action" not in prompt
        # A perfect validation cannot make an incorrect final answer pass.
        fit = json.loads(await _execute(session, "validate_fit", **exact_submission))
        assert fit["valid"]
        assert fit["residuals"]["rms"] == 0
        assert not {"success", "reward", "matching", "comparison", "done"} & fit.keys()
        state = await store.materialize("main")
        with pytest.raises(ValueError, match="submission"):
            TaskScorer(environment.current_task).evaluate(state)
        await _execute(session, "submit_answer", answer='{"planets": []}')
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("main"))
            .score
            == 0
        )
        for name, arguments in [
            ("PythonREPL", {"input_code": "print(42)"}),
            ("submit_answer", {"answer": json.dumps(exact_submission)}),
        ]:
            result = await session.execute(
                Action(id=uuid4().hex, name=name, arguments=arguments)
            )
            assert not result.success


@pytest.mark.anyio
async def test_final_answer_is_graded_without_validations(
    tmp_path, environment, exact_submission
):
    async with SQLiteCommitStore(tmp_path / "final.sqlite3", "final") as store:
        session = await _start_session(environment, store)
        await _execute(session, "submit_answer", answer=json.dumps(exact_submission))
        state = await store.materialize("main")
        result = TaskScorer(environment.current_task).evaluate(state)
        assert result.score == 1
        assert result.metadata["evaluation"]["ok_complete_matching"]
        assert '"match_score"' not in state.model_dump_json()
        restored = _resume_session(environment, state, store)
        rejected = await restored.execute(
            Action(name="submit_answer", arguments={"answer": "{}"})
        )
        assert not rejected.success


@pytest.mark.anyio
@pytest.mark.parametrize("answer", ["{}", "bad json", '{"planets": null}'])
async def test_malformed_final_is_terminal_and_scores_zero(
    tmp_path, environment, answer
):
    async with SQLiteCommitStore(tmp_path / "invalid.sqlite3", "invalid") as store:
        session = await _start_session(environment, store)
        await _execute(session, "submit_answer", answer=answer)
        state = await store.materialize("main")
        assert state.is_terminal
        assert TaskScorer(environment.current_task).evaluate(state).score == 0


@pytest.mark.anyio
async def test_history_mutations_and_observation_tampering_do_not_change_grading(
    tmp_path, environment, exact_submission
):
    async with SQLiteCommitStore(tmp_path / "history.sqlite3", "history") as store:
        session = await _start_session(environment, store)
        original = json.loads(
            await _execute(session, "validate_fit", **exact_submission)
        )
        await _execute(
            session,
            "PythonREPL",
            input_code="history[0]['residuals']['rms'] = 999\nrvs_ms[:] = 999\nprint(instruments[0])",
        )
        assert (
            _submission_state(await store.materialize("main"))["history"][0][
                "residuals"
            ]["rms"]
            == 0
        )
        assert (
            json.loads(await _execute(session, "validate_fit", **exact_submission))
            == original
        )
        # Printed low-RMS text no longer forces submission.
        await _execute(
            session,
            "PythonREPL",
            input_code='print("Kepler=YES; Best_RMS_over_med_sigma=0.01")',
        )
        assert (
            await _execute(session, "PythonREPL", input_code="print(42)")
        ).strip() == "42"
        await _execute(session, "submit_answer", answer=json.dumps(exact_submission))
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("main"))
            .score
            == 1
        )


@pytest.mark.anyio
async def test_resume_and_fork_keep_public_history_and_separate_candidates(
    tmp_path, environment, exact_submission
):
    db = tmp_path / "resume.sqlite3"
    async with SQLiteCommitStore(db, "resume") as store:
        session = await _start_session(environment, store)
        await _execute(session, "validate_fit", planets=[])
        await _execute(
            session,
            "PythonREPL",
            input_code="values = np.arange(3.0)\ndef shifted():\n    return values + 2",
        )
        branch = await session.fork_branch(branch_id="alternative")
        assert json.loads(
            await _execute(
                branch,
                "PythonREPL",
                input_code="values[0] = 40\nprint(shifted().tolist())",
            )
        ) == [42, 3, 4]
        await _execute(branch, "validate_fit", **exact_submission)
        assert len(_submission_state(await store.materialize("main"))["history"]) == 1
        await _execute(branch, "submit_answer", answer='{"planets": []}')
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("alternative"))
            .score
            == 0
        )
    async with SQLiteCommitStore(db, "resume") as store:
        session = _resume_session(environment, await store.materialize("main"), store)
        assert json.loads(
            await _execute(
                session, "PythonREPL", input_code="print(shifted().tolist())"
            )
        ) == [2, 3, 4]
        await _execute(session, "submit_answer", answer=json.dumps(exact_submission))
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("main"))
            .score
            == 1
        )


@pytest.mark.anyio
async def test_parallel_final_rejected(tmp_path, environment, exact_submission):
    async with SQLiteCommitStore(tmp_path / "parallel.sqlite3", "parallel") as store:
        session = await _start_session(environment, store)
        with pytest.raises(ValueError, match="parallel"):
            await session.execute_many(
                (
                    Action(name="validate_fit", arguments=exact_submission),
                    Action(name="submit_answer", arguments={"answer": "{}"}),
                )
            )
        assert (await store.materialize("main")).submission is None


def test_official_execution_requires_worker(tmp_path):
    from corral.runtime import permissions

    if permissions.enabled():
        pytest.skip("already restricted")
    environment = create_environments(work_dir=tmp_path)["seed15_diff5"]
    with pytest.raises(RuntimeError, match="restricted Docker"):
        environment.initial_event(execution_id="unsafe")


@pytest.mark.anyio
async def test_old_or_different_protocol_is_rejected(tmp_path, environment):
    async with SQLiteCommitStore(tmp_path / "old.sqlite3", "old") as store:
        await _start_session(environment, store)
        state = await store.materialize("main")
        metadata = dict(state.task.metadata)
        metadata.pop("execution_version")
        data = state.model_dump(mode="json")
        data["task"]["metadata"] = metadata
        old = type(state).model_validate(data)
        with pytest.raises(ValueError, match="Incompatible"):
            environment.validate_state_tool_catalog(old)


@pytest.mark.anyio
async def test_prompt_and_repl_reference_substitution(
    tmp_path, simple_task, exact_submission, monkeypatch
):
    from dataclasses import replace

    from stargazer.env import PROTOCOL_VERSION

    results = []

    def forbidden(*_a, **_kw):
        raise AssertionError("private grader accessed")

    monkeypatch.setattr("stargazer.score.evaluate_submission", forbidden)
    monkeypatch.setattr("stargazer.score._match_planets", forbidden)
    for i, private in enumerate(
        (
            simple_task,
            replace(
                simple_task,
                truth_planets=(),
                metadata={"reference": "SECRET", "hints": {"count": 999}},
            ),
        )
    ):
        task = TaskDefinition(
            name="blind",
            description="blind",
            tools=["PythonREPL", "validate_fit"],
            scoring_fn=make_stargazer_scorer(private),
            submission_format=SUBMISSION_FORMAT,
            scoring_inputs={
                "benchmark_task": private,
                "public_context": private.public_fit_context(),
                "development_mode": True,
            },
            prompt_fn=_task_prompt,
            setup_fn=_configure_trial,
            resolve_answer=False,
            allow_previous_attempt_context=False,
            execution_version=PROTOCOL_VERSION,
        )
        env = StargazerEnvironment(
            "blind", task, toolset=Toolset(pool=create_tools(), workspace_factory=None)
        )
        async with SQLiteCommitStore(tmp_path / f"{i}.sqlite3", f"trial{i}") as store:
            session = await _start_session(env, store)
            outputs = [session.prompt, session.tools]
            for name, args in [
                ("validate_fit", exact_submission),
                ("validate_fit", {"planets": [{"P_days": -1}]}),
                (
                    "PythonREPL",
                    {"input_code": "import json\nprint(json.dumps(history))"},
                ),
            ]:
                outputs.append(await _execute(session, name, **args))
            results.append(outputs)
    assert results[0] == results[1]


@pytest.mark.anyio
async def test_assisted_resume_arrays_and_final_authority(
    tmp_path, environment, exact_submission
):
    async with SQLiteCommitStore(tmp_path / "assisted.sqlite3", "assisted") as store:
        session = await _start_session(environment, store)
        code = f"planets = {exact_submission['planets']!r}\nd = stargazer_diagnostics(planets)\ncurve = stargazer_predict(planets)\nprint(d['residuals']['rms'])"
        assert (await _execute(session, "PythonREPL", input_code=code)).strip() == "0.0"
        restored = _resume_session(environment, await store.materialize("main"), store)
        assert (
            await _execute(
                restored,
                "PythonREPL",
                input_code="print(np.array_equal(curve, stargazer_predict(planets)))",
            )
        ).strip() == "True"
        await _execute(
            restored,
            "PythonREPL",
            input_code="d['model_ms'][:]=999\nd['residuals_ms'][:]=999\ncurve[:]=999\nrvs_ms[:]=999\nhistory.append({'success': True})",
        )
        fit = json.loads(await _execute(restored, "validate_fit", **exact_submission))
        assert fit["residuals"]["rms"] == 0
        assert "model_ms" not in fit
        await _execute(restored, "submit_answer", answer=json.dumps(exact_submission))
        assert (
            TaskScorer(environment.current_task)
            .evaluate(await store.materialize("main"))
            .score
            == 1
        )


@pytest.mark.anyio
async def test_arms_reject_cross_setting_resume(tmp_path):
    envs = [
        create_environments(
            work_dir=tmp_path / str(assisted),
            development_mode=True,
            analysis_assistance=assisted,
        )["seed15_diff5"]
        for assisted in (False, True)
    ]
    assert (
        envs[0].current_task.execution_version != envs[1].current_task.execution_version
    )
    for i, env in enumerate(envs):
        async with SQLiteCommitStore(
            tmp_path / f"arm-{i}.sqlite3", f"arm-{i}"
        ) as store:
            session = await _start_session(env, store)
            result = await _execute(
                session,
                "PythonREPL",
                input_code="print('stargazer_predict' in globals())",
            )
            assert result.strip() == str(bool(i))
            state = await store.materialize("main")
            env.validate_state_tool_catalog(state)
            with pytest.raises(ValueError, match="Incompatible"):
                envs[1 - i].validate_state_tool_catalog(state)


@pytest.mark.anyio
@pytest.mark.parametrize("task_id", ["seed12_diff5", "seed17_diff8"])
async def test_recorded_reconstruction_sequences_are_reference_independent(
    tmp_path, task_id, monkeypatch
):
    from dataclasses import replace
    from pathlib import Path

    from stargazer.env import PROTOCOL_VERSION

    rows = json.loads(
        (Path(__file__).parent / "fixtures/run2_candidates.json").read_text()
    )
    candidates = [row["candidate"] for row in rows if row["task_id"] == task_id]
    assert len(candidates) == (5 if task_id == "seed12_diff5" else 2)
    materialized = load_task(DEFAULT_DATA_ROOT / "synthetic" / f"{task_id}.json")

    def forbidden(*_a, **_kw):
        raise AssertionError("pre-final private evaluation")

    monkeypatch.setattr("stargazer.score.evaluate_submission", forbidden)
    monkeypatch.setattr("stargazer.score._match_planets", forbidden)
    outputs = []
    for index, task in enumerate(
        (materialized, replace(materialized, truth_planets=(), metadata={}))
    ):
        definition = TaskDefinition(
            name="replay",
            description="replay",
            tools=["PythonREPL", "validate_fit"],
            scoring_fn=make_stargazer_scorer(task),
            execution_version=PROTOCOL_VERSION,
            allow_previous_attempt_context=False,
            submission_format=SUBMISSION_FORMAT,
            scoring_inputs={
                "benchmark_task": task,
                "public_context": task.public_fit_context(),
                "development_mode": True,
            },
            prompt_fn=_task_prompt,
            setup_fn=_configure_trial,
            resolve_answer=False,
        )
        env = StargazerEnvironment(
            "replay",
            definition,
            toolset=Toolset(pool=create_tools(), workspace_factory=None),
        )
        async with SQLiteCommitStore(
            tmp_path / f"{index}.sqlite3", f"replay-{index}"
        ) as store:
            session = await _start_session(env, store)
            observed = [session.prompt, session.tools]
            for candidate in candidates:
                fit = json.loads(await _execute(session, "validate_fit", **candidate))
                assert (
                    not {
                        "reward",
                        "success",
                        "done",
                        "matching",
                        "comparison",
                        "count_difference",
                        "match_score",
                    }
                    & fit.keys()
                )
                observed.append(fit)
                observed.append(
                    await _execute(
                        session,
                        "PythonREPL",
                        input_code="d = stargazer_diagnostics(history[-1]['candidate']['planets'], history[-1]['candidate']['noise_jitter_ms'])\nprint(d['residuals'], d['bic'])",
                    )
                )
                state = await store.materialize("main")
                assert not state.is_terminal
                assert state.submission is None
                observed.append(_submission_state(state))
            observed.append(
                await _execute(
                    session,
                    "PythonREPL",
                    input_code="print(sorted(STARGAZER_PUBLIC_RESOURCES))\nprint(stargazer_predict.__func__.__closure__)",
                )
            )
            outputs.append(observed)
    assert outputs[0] == outputs[1]
