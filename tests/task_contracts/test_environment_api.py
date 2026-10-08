"""Exercise installed benchmark definitions through Corral's public runtime.

The root suite covers samplemath and AFM with an instrument double. Each task CI
job selects its installed benchmark via CORRAL_TEST_ENVIRONMENT, including every
available level and both task layouts. Domain services are exercised by the task's
own tool tests.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from corral.agents.schema import AgentOutcome
from corral.core import RESOURCE_STATE_NAMESPACE, Action, ExecutionState
from corral.core.task import topological_order
from corral.core.tool import Tool
from corral.observability import NoOpObserver
from corral.orchestration.evaluation import evaluate_task
from corral.orchestration.models import EvaluateTaskInput
from corral.orchestration.registry import RuntimeRegistry
from corral.persistence import SQLiteCommitStore
from corral.run import CorralRunner
from corral.runtime import TaskRuntime
from corral.runtime.environment_loader import load_environment_group

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = (
    [os.environ["CORRAL_TEST_ENVIRONMENT"]]
    if "CORRAL_TEST_ENVIRONMENT" in os.environ
    else ["samplemath", "afm"]
)
CONFIGURATIONS = [
    (benchmark, int(level.name.removeprefix("level_")), kind == "subtasks_json")
    for benchmark in BENCHMARKS
    for level in sorted((ROOT / "tasks" / benchmark / "environments").glob("level_*"))
    for kind in ("tasks_json", "subtasks_json")
    if any((level / kind).glob("*.json"))
]


@pytest.fixture
def afm_instrument(monkeypatch):
    """Keep the real AFM loader, prompt and reset hook; replace device dependencies."""
    applications = []

    def spm():
        application = SimpleNamespace(
            Scan=SimpleNamespace(),
            ZController=SimpleNamespace(),
            ScanHead=SimpleNamespace(),
            OperatingMode=SimpleNamespace(),
            GetGalleryHistoryDirectoryPath=None,
        )

        def set_directory(path):
            application.GetGalleryHistoryDirectoryPath = path

        application.SetGalleryHistoryDirectoryPath = set_directory
        applications.append(application)
        return SimpleNamespace(application=application)

    score = ModuleType("score")
    for name in (
        "score_friction",
        "score_roughness",
        "score_roughness_and_friction",
        "score_topography",
    ):
        setattr(score, name, lambda **_params: lambda _answer: 1.0)
    tools = ModuleType("tools")
    for name in (
        "Code_Executor",
        "Document_Retrieval",
        "Image_Analyzer",
        "Image_optimizer",
        "scan_grain_area",
        "visualize_grain_boxes",
    ):
        setattr(tools, name, Tool(name=name, description="Instrument test double."))
    monkeypatch.setitem(sys.modules, "nanosurf", SimpleNamespace(SPM=spm))
    monkeypatch.setitem(sys.modules, "score", score)
    monkeypatch.setitem(sys.modules, "tools", tools)
    monkeypatch.setitem(
        sys.modules,
        "pythoncom",
        SimpleNamespace(CoInitialize=lambda: None, CoUninitialize=lambda: None),
    )
    # Register cleanup before the loader imports AFM's bare `env` module.
    monkeypatch.setitem(sys.modules, "env", None)
    del sys.modules["env"]
    return applications


class SubmitAgent:
    async def run_session(self, session):
        assert session.prompt
        result = await session.execute(
            Action(name="submit_answer", arguments={"answer": "api-contract-answer"})
        )
        assert result.success, result
        return AgentOutcome(status="completed", answer="api-contract-answer")


@pytest.mark.parametrize(("benchmark", "level", "subtasks"), CONFIGURATIONS)
def test_task_definitions_run_and_restore(
    benchmark, level, subtasks, tmp_path, monkeypatch, request
):
    applications = (
        request.getfixturevalue("afm_instrument") if benchmark == "afm" else []
    )
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv("CORRAL_WORK_DIR", str(tmp_path / "work"))
    env_kwargs = {"level": level, "subtasks": subtasks}
    if benchmark == "stargazer":
        # This API check runs a fixed submit-only agent on the host. Stargazer's
        # restricted Docker workers are exercised by its own permission tests.
        env_kwargs["development_mode"] = True
    # Do not supply repository_root: this also checks the CLI's default lookup.
    environments = load_environment_group(benchmark, env_kwargs=env_kwargs)
    tasks = {name: env.current_task for name, env in environments.items()}
    order = topological_order(tasks)

    async def run():
        outputs = {}
        database = tmp_path / "commits.sqlite3"
        async with SQLiteCommitStore(database) as store:
            for task_id in order:
                execution_id = f"contract:{task_id}"
                environment = environments[task_id].for_task(execution_id)
                expected_tools = set(environment.current_task.tools) - set(
                    environment.current_task.excluded_tools
                )
                assert expected_tools <= environment.tools.keys(), task_id
                try:
                    state = await TaskRuntime(store, NoOpObserver()).run(
                        SubmitAgent(),
                        environment,
                        execution_id=execution_id,
                        started_at=datetime.now(timezone.utc),
                        max_iterations=2,
                        dependency_outputs={
                            name: outputs[name]
                            for name in tasks[task_id].dependencies()
                        },
                    )
                    assert isinstance(state, ExecutionState)
                    assert state.runtime.status == "submitted", task_id
                    assert state.runtime.metadata["execution_completed"] is True
                    assert state.submission == "api-contract-answer"
                    assert state.task.metadata["prompt"]
                    if benchmark == "spectra_elucidation":
                        hidden = state.environment.values.get("hidden_arguments", {})
                        assert hidden["h_smiles"] == tasks[task_id].scoring_inputs
                    elif benchmark == "wetlab":
                        resources = state.environment.values.get(
                            RESOURCE_STATE_NAMESPACE, {}
                        )
                        assert resources["wetlab"]
                    output = environment.get_task_output(state)
                    assert output is not None, task_id
                    outputs[task_id] = output
                finally:
                    environment.shutdown_jobs()

        # Reconstruct projections from a reopened store, including dependencies.
        async with SQLiteCommitStore(database) as restored:
            for task_id in order:
                state = await restored.for_execution(f"contract:{task_id}").materialize(
                    "main"
                )
                assert environments[task_id].get_task_output(state) == outputs[task_id]
                assert dict(state.dependency_outputs) == {
                    name: outputs[name] for name in tasks[task_id].dependencies()
                }

    try:
        asyncio.run(run())
        if benchmark == "afm":
            assert len(applications) == len(order)
            for task_id, application in zip(order, applications, strict=True):
                params = tasks[task_id].initial_input["params"]
                assert application.ZController.PGain == params["pgain"]
                if "image_width" in params:
                    assert application.Scan.ImageWidth == params["image_width"] * 1e-9
                assert Path(application.GetGalleryHistoryDirectoryPath).is_dir()
    finally:
        for environment in environments.values():
            environment.shutdown_jobs()


def test_afm_serializes_different_tasks_and_trials(
    tmp_path, monkeypatch, afm_instrument
):
    application = sys.modules["nanosurf"].SPM().application
    monkeypatch.setattr(
        sys.modules["nanosurf"], "SPM", lambda: SimpleNamespace(application=application)
    )
    environments = load_environment_group(
        "afm", env_kwargs={"work_dir": str(tmp_path / "work")}
    )
    # These tasks reset to different modes and save directories.
    environments = dict(list(environments.items())[:2])
    active = peak = 0
    observations = []

    class ProbeAgent(SubmitAgent):
        async def run_session(self, session):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.1)
                env = session.environment
                observations.append(
                    (
                        application.OperatingMode.OperatingMode
                        == env.initial_params["mode"],
                        application.GetGalleryHistoryDirectoryPath
                        == env.workspace_path,
                    )
                )
                return await super().run_session(session)
            finally:
                active -= 1

    async def run():
        async with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
            registry = RuntimeRegistry(
                agents={"agent": ProbeAgent()}, environments=environments
            )
            try:
                runner = CorralRunner(
                    registry,
                    environments=environments,
                    state_store=store,
                    observer=NoOpObserver(),
                )
                result = await runner._execute_benchmark(
                    runner.build_input(
                        "afm-concurrency",
                        trials_per_task=2,
                        max_parallel=8,
                        max_parallel_per_task=2,
                        evaluate=False,
                    )
                )
                assert all(
                    trial.error is None and trial.output_ready
                    for trial in result.trials
                )
            finally:
                registry.close()

    asyncio.run(run())
    assert peak == 1
    assert observations == [(True, True)] * 4


def test_afm_guard_releases_after_error_and_cancelled_waiter(tmp_path, afm_instrument):
    environments = load_environment_group(
        "afm", env_kwargs={"work_dir": str(tmp_path / "work")}
    )
    first, second = list(environments.values())[:2]

    async def wait_for_instrument():
        async with second.task_execution_guard():
            raise AssertionError("A second task acquired the instrument")

    async def run():
        with pytest.raises(RuntimeError, match="failed task"):
            async with first.task_execution_guard():
                waiter = asyncio.create_task(wait_for_instrument())
                await asyncio.sleep(0)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                raise RuntimeError("failed task")
        async with asyncio.timeout(1):
            async with second.task_execution_guard():
                pass

    # The guard also remains usable across event loops on subsequent runs.
    asyncio.run(run())
    asyncio.run(run())


def test_afm_evaluation_uses_persisted_scan_with_full_path_submission(
    tmp_path, monkeypatch, afm_instrument
):
    checked = []

    def score_scan(answer):
        checked.append(answer)
        return float(Path(answer).read_bytes() == b"saved acquisition")

    monkeypatch.setattr(
        sys.modules["score"], "score_topography", lambda **_params: score_scan
    )
    environments = load_environment_group(
        "afm", env_kwargs={"work_dir": str(tmp_path / "work")}
    )
    task_id = next(iter(environments))
    original = None

    class ScanAgent:
        async def run_session(self, session):
            nonlocal original
            original = Path(session.environment.workspace_path) / "scans/scan.nid"
            original.parent.mkdir()
            original.write_bytes(b"saved acquisition")
            result = await session.execute(
                Action(name="submit_answer", arguments={"answer": str(original)})
            )
            assert result.success
            return AgentOutcome(status="completed", answer=str(original))

    async def run():
        async with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
            registry = RuntimeRegistry(agents={}, environments=environments)
            environment = registry.environment(task_id, "afm-snapshot")
            try:
                state = await TaskRuntime(store, NoOpObserver()).run(
                    ScanAgent(),
                    environment,
                    execution_id="afm-snapshot",
                    started_at=datetime.now(timezone.utc),
                    max_iterations=2,
                )
                request = EvaluateTaskInput(
                    execution_id=state.execution_id,
                    environment_id=task_id,
                    commit_hash=state.through_commit_hash,
                )
                assert original is not None
                original.write_bytes(b"changed live scan")
                for _ in range(2):
                    result = await evaluate_task(
                        request,
                        state_store=store,
                        registry=registry,
                        observer=NoOpObserver(),
                    )
                    assert result.score == 1
                    if original.parent.exists():
                        shutil.rmtree(original.parent)
                restored = await store.for_execution(state.execution_id).materialize(
                    "main"
                )
                assert restored.submission == str(original)
                assert len(checked) == 2
                assert all(path != str(original) for path in checked)
            finally:
                registry.close()

    asyncio.run(run())


def test_afm_cancellation_waits_for_instrument_reset(
    tmp_path, monkeypatch, afm_instrument
):
    environments = load_environment_group(
        "afm", env_kwargs={"work_dir": str(tmp_path / "work")}
    )
    templates = list(environments.values())[:2]
    first, second = [
        env.for_task(f"cancel-reset-{i}") for i, env in enumerate(templates)
    ]
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    second_started = threading.Event()
    configure_first, configure_second = first.configure, second.configure

    def slow_configure(state):
        started.set()
        try:
            if not release.wait(5):
                raise TimeoutError("Test did not release the instrument reset")
            return configure_first(state)
        finally:
            finished.set()

    def next_configure(state):
        second_started.set()
        assert finished.is_set()
        return configure_second(state)

    monkeypatch.setattr(first, "configure", slow_configure)
    monkeypatch.setattr(second, "configure", next_configure)

    async def run():
        async with SQLiteCommitStore(tmp_path / "commits.sqlite3") as store:
            runtime = TaskRuntime(store, NoOpObserver())

            async def execute(env, execution_id):
                return await runtime.run(
                    SubmitAgent(),
                    env,
                    execution_id=execution_id,
                    started_at=datetime.now(timezone.utc),
                    max_iterations=2,
                )

            running = asyncio.create_task(execute(first, "cancel-reset-0"))
            waiting = None
            try:
                assert await asyncio.to_thread(started.wait, 2)
                running.cancel()
                waiting = asyncio.create_task(execute(second, "cancel-reset-1"))
                await asyncio.sleep(0.1)
                assert not running.done()
                assert not second_started.is_set()
            finally:
                release.set()
                await asyncio.gather(running, return_exceptions=True)
                if waiting is not None:
                    result = await waiting
                    assert result.runtime.status == "submitted"
                first.shutdown_jobs()
                second.shutdown_jobs()
            assert running.cancelled()
            assert second_started.is_set()

    asyncio.run(run())
