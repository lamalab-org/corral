"""The budget ledger persists through Corral's own tool executor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from inference_opt import datasets
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
    environment = create_environments(level=1, work_dir=str(tmp_path))["math_algebra_b"]
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
    definition = create_environments(level=1, work_dir=str(tmp_path))["math_algebra_b"]
    environment = definition.for_task("ledger-test")
    seed = environment.initial_event(execution_id="ledger-test").environment
    state = _state(environment, dict(seed))
    policy = Path(environment.workspace_path) / "policy" / "policy.py"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(
        "def run(questions, ctx):\n"
        "    texts = ctx.student.generate([q.text for q in questions])\n"
        "    ctx.answer({q.id: t for q, t in zip(questions, texts)})\n"
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
    definition = create_environments(level=1, work_dir=str(tmp_path))["math_algebra_b"]
    environment = definition.for_task("ledger-test")
    seed = environment.initial_event(execution_id="ledger-test").environment
    state = _state(environment, dict(seed))

    _, state = _call(environment, state, "reveal_examples", {}, "reveal")
    committed = state.environment.values["resources"][LEDGER_RESOURCE]
    assert committed["reveals"] == 1
    assert len(committed["revealed_ids"]) == 5
    examples = {item.item_id for item in datasets.load_items("math_algebra", "examples")}
    assert set(committed["revealed_ids"]) <= examples

    result, _ = _call(environment, state, "get_budget", {}, "budget")
    budget = json.loads(result.content.splitlines()[1])
    assert budget["used"]["reveals"] == 1
    assert budget["questions_revealed"] == 5


def test_parallel_branch_tools_share_the_default_trial_budget(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from inference_opt.api import BudgetExhausted

    from corral.runtime.permissions import NODE_WORKSPACE_DIR

    definition = create_environments(level=1, work_dir=str(tmp_path))["math_algebra_b"]
    main = definition.for_task("parallel-trial")
    trial = Path(main.workspace_path)
    branches = [
        definition.for_task(
            f"branch-{i}", workspace_path=str(trial / NODE_WORKSPACE_DIR / f"node-{i}")
        )
        for i in range(8)
    ]

    def reveal(environment):
        seed = environment.initial_event(execution_id="branch").environment
        try:
            _call(
                environment,
                _state(environment, dict(seed)),
                "reveal_examples",
                {},
                "reveal",
            )
            return True
        except BudgetExhausted:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reveal, branches)) == 4
    seed = main.initial_event(execution_id="parallel-trial").environment
    result, _ = _call(main, _state(main, dict(seed)), "get_budget", {}, "budget")
    budget = json.loads(result.content.splitlines()[1])
    assert budget["shared_across_branches"]
    assert budget["used"]["reveals"] == 4
    assert budget["remaining"]["reveals"] == 0
    assert budget["questions_revealed"] == 20


@pytest.mark.parametrize("relocate", [False, True])
def test_checkpoint_restores_branch_usage_with_zero_main_usage(tmp_path, relocate):
    from inference_opt.api import BudgetExhausted
    from inference_opt.tools import _shared_ledger

    from corral.runtime.permissions import NODE_WORKSPACE_DIR

    definition = create_environments(level=1, work_dir=str(tmp_path))["math_algebra_b"]
    main = definition.for_task("original")
    branch = definition.for_task(
        "branch", workspace_path=str(Path(main.workspace_path) / NODE_WORKSPACE_DIR / "node")
    )
    seed = main.initial_event(execution_id="original").environment
    state = _state(main, dict(seed))
    for index in range(4):
        _, state = _call(branch, state, "reveal_examples", {}, f"reveal-{index}")
    checkpoint = main.capture_environment(seed)
    assert checkpoint["resources"][LEDGER_RESOURCE]["reveals"] == 0
    assert checkpoint["resources"][LEDGER_RESOURCE]["shared_ledger"]["used"]["reveals"] == 4

    _shared_ledger(main.workspace_path).path.unlink()
    recovered = definition.for_task("relocated") if relocate else main
    result, restored = _call(recovered, _state(recovered, checkpoint), "get_budget", {}, "budget")
    budget = json.loads(result.content.splitlines()[1])
    assert budget["remaining"]["reveals"] == 0
    assert budget["questions_revealed"] == 20
    with pytest.raises(BudgetExhausted):
        _call(recovered, restored, "reveal_examples", {}, "overdraw")
