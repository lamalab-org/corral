"""Exercise installed benchmark definitions through Corral's public runtime.

The root suite covers samplemath and AFM with an instrument double. Each task CI
job selects its installed benchmark via CORRAL_TEST_ENVIRONMENT, including every
available level and both task layouts. Domain services are exercised by the task's
own tool tests.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from corral.agents.schema import AgentOutcome
from corral.core import Action, ExecutionState
from corral.core.task import topological_order
from corral.core.tool import Tool
from corral.observability import NoOpObserver
from corral.persistence import SQLiteCommitStore
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


@pytest.fixture()
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
        "check_file_exists",
        "check_image_quality",
        "check_mathematical_eq",
        "check_numerical",
        "check_params_function",
        "check_roughness_function",
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
    # Do not supply repository_root: this also checks the CLI's default lookup.
    environments = load_environment_group(
        benchmark, env_kwargs={"level": level, "subtasks": subtasks}
    )
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
                    hidden = state.environment.values.get("hidden_arguments", {})  # noqa: PD011
                    if benchmark == "ml":
                        assert hidden["work_dir"] == environment.workspace_path
                    elif benchmark == "spectra_elucidation":
                        assert hidden["h_smiles"] == tasks[task_id].scoring_inputs
                    elif benchmark == "wetlab":
                        assert hidden["wetlab"]
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
