"""The budget ledger persists through Corral's own tool executor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_opt.env import LEDGER_RESOURCE, create_environments

from corral.core.resources import RESOURCE_CATALOG_METADATA_KEY
from corral.core.state import EnvironmentState, ExecutionState, TaskState
from corral.core.tool_catalog import TOOL_CATALOG_METADATA_KEY, TOOL_POLICY_METADATA_KEY
from corral.runtime.tool_execution import ToolExecutor


def _state(environment, values: dict[str, Any]) -> ExecutionState:
    return ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="ledger-test",
        branch_id="main",
        task=TaskState(
            environment={
                TOOL_CATALOG_METADATA_KEY: environment.tool_catalog_snapshot().model_dump(
                    mode="json"
                ),
                TOOL_POLICY_METADATA_KEY: environment.tool_policy_snapshot().model_dump(
                    mode="json"
                ),
                RESOURCE_CATALOG_METADATA_KEY: {},
            }
        ),
        environment=EnvironmentState(values=values),
    )


def _call(environment, state, name: str, arguments: dict[str, Any], action_id: str):
    tool = environment.tools[name]
    result = ToolExecutor(environment).execute(state, tool, arguments, action_id=action_id)
    return result, _state(environment, dict(result.environment))


def test_initial_event_seeds_the_ledger(tmp_path):
    environment = create_environments(level=1, work_dir=str(tmp_path))["mmlu_pro_a"]
    started = environment.initial_event(execution_id="ledger-test")
    ledger = started.environment["resources"][LEDGER_RESOURCE]
    assert ledger["reveals"] == 0
    assert ledger["revealed_ids"] == []
    assert LEDGER_RESOURCE not in started.environment["hidden_arguments"]


def test_each_run_keeps_its_graded_logs_out_of_the_workspace(tmp_path, monkeypatch):
    import asyncio
    import tarfile

    from inference_opt import runner
    from inference_opt.gateway import MockBackend

    monkeypatch.setattr(runner, "backend_for", lambda *_args: MockBackend())
    definition = create_environments(level=1, work_dir=str(tmp_path))["mmlu_pro_a"]
    environment = definition.for_task("ledger-test")
    seed = environment.initial_event(execution_id="ledger-test").environment
    state = _state(environment, dict(seed))
    policy = Path(environment.workspace_path) / "policy" / "policy.py"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(
        "def solve(question, ctx):\n    return ctx.student.generate(question.text)\n"
    )

    _, state = _call(environment, state, "dry_run_policy", {}, "dry")

    record = state.environment.values["resources"][LEDGER_RESOURCE]["runs"][0]
    store = environment.workspace_manager.artifact_store
    archive = tmp_path / "dry-1.tar.gz"
    archive.write_bytes(asyncio.run(store.get_bytes(record["private_artifact"])))
    with tarfile.open(archive) as tar:
        assert any("/log/" in name for name in tar.getnames())
    assert not list(Path(environment.workspace_path).rglob("log/*.json"))


def test_charges_persist_between_tool_calls(tmp_path):
    definition = create_environments(level=1, work_dir=str(tmp_path))["mmlu_pro_a"]
    environment = definition.for_task("ledger-test")
    seed = environment.initial_event(execution_id="ledger-test").environment
    state = _state(environment, dict(seed))

    _, state = _call(environment, state, "reveal_train_questions", {}, "reveal")
    committed = state.environment.values["resources"][LEDGER_RESOURCE]
    assert committed["reveals"] == 1
    assert len(committed["revealed_ids"]) == 5

    result, _ = _call(environment, state, "get_budget", {}, "budget")
    budget = json.loads(result.content.splitlines()[1])
    assert budget["used"]["reveals"] == 1
    assert budget["questions_revealed"] == 5
