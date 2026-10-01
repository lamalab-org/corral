"""A policy run in the real Docker jail: no answers, no network, metered student.

Run inside the trial image as root with CORRAL_PERMISSION_TESTS=1; elsewhere
these tests are skipped, because only Docker enforces the boundary.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from inference_opt import datasets
from inference_opt.gateway import MockBackend
from inference_opt.runner import private_root, run_policy

pytestmark = pytest.mark.skipif(
    os.environ.get("CORRAL_PERMISSION_TESTS") != "1",
    reason="requires the real Docker trial permission boundary",
)

BENCHMARK = "mmlu_pro"

PROBING_POLICY = """
import json, os, socket, sys

from corral.runtime import service_channel


class Policy:
    def run(self, questions, ctx):
        probe = {"uid": os.geteuid(), "env": sorted(os.environ)}
        for path in (LABELS, "/opt/corral/tasks/inference_opt/inference_opt/data"):
            try:
                open(path).read(1) if os.path.isfile(path) else os.listdir(path)
                probe.setdefault("readable", []).append(path)
            except OSError:
                probe.setdefault("blocked", []).append(path)
        try:
            socket.create_connection(("127.0.0.1", 8084), timeout=2).close()
            probe["network"] = "open"
        except OSError:
            probe["network"] = "blocked"
        try:
            service_channel.call({"op": "submit", "question_id": "not-in-run", "answer": "x"})
            probe["foreign_submit"] = "accepted"
        except service_channel.ServiceError as exc:
            probe["foreign_submit"] = str(exc)
        probe["examples"] = [e.answer for e in ctx.examples]
        probe["reply"] = ctx.student.generate("hello", question_id=questions[0].id)
        with open("/workspace/probe.json", "w") as stream:
            json.dump(probe, stream)
        for q in questions:
            ctx.submit(q.id, "ANSWER: A")
"""

HONEST_POLICY = """
class Policy:
    def solve(self, question, ctx):
        ctx.student.generate(question.text)
        return "ANSWER: " + ANSWERS[question.id]
"""


@pytest.fixture(autouse=True)
def restricted_runtime(tmp_path):
    from corral.runtime import permissions

    assert os.geteuid() == 0
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(mode=0o700)
    permissions.configure(checkpoints)
    yield
    permissions._enabled = False


@pytest.fixture
def workspace():
    root = Path(tempfile.mkdtemp(prefix="inference-opt-isolation-", dir="/workspace"))
    (root / "policy").mkdir()
    return root


def _run(workspace: Path, budget: int = 10, examples=None):
    items = datasets.load_items(BENCHMARK, "train")[:3]
    config = {
        "benchmark": BENCHMARK,
        "models": ["student_a"],
        "model_specs": {"student_a": "mockllm/model"},
        "policy_api": "primitive",
    }
    return items, run_policy(
        config=config,
        work_dir=workspace,
        run_id="exp-1",
        policy_dir=workspace / "policy",
        split="train",
        models=["student_a"],
        budget_per_model=budget,
        examples=examples or [],
        item_ids=[item.item_id for item in items],
        backends={"student_a": MockBackend("student says hi")},
    )["student_a"]


def test_a_jailed_policy_sees_no_answers_and_no_network(workspace):
    source = PROBING_POLICY.replace("LABELS", repr(str(datasets.labels_path())))
    (workspace / "policy" / "policy.py").write_text(source)
    examples = [{**datasets.public_record(datasets.load_items(BENCHMARK, "train")[5]),
                 "target": "Z"}]
    items, run = _run(workspace, examples=examples)

    assert run.error == "" and run.load_error == ""
    probe = json.loads((workspace / "probe.json").read_text())
    assert probe["uid"] != 0
    assert "readable" not in probe, probe
    assert probe["network"] == "blocked"
    assert not any(key.endswith(("_KEY", "_TOKEN")) for key in probe["env"])
    assert "CORRAL_INFERENCE_LABELS_PATH" not in probe["env"]
    assert "not a question in this run" in probe["foreign_submit"]
    # Paid-for examples are the only answers that cross into the jail.
    assert probe["examples"] == ["Z"]
    assert probe["reply"] == "student says hi"
    assert run.calls_used == 1
    assert set(run.answers) == {item.item_id for item in items}


def test_grading_happens_outside_and_leaves_no_gold_in_the_workspace(workspace):
    items = datasets.load_items(BENCHMARK, "train")[:3]
    targets = datasets.load_targets(BENCHMARK, "train")
    answers = {items[0].item_id: targets[items[0].item_id], items[1].item_id: "Z",
               items[2].item_id: targets[items[2].item_id]}
    (workspace / "policy" / "policy.py").write_text(
        f"ANSWERS = {answers!r}\n" + HONEST_POLICY
    )
    _, run = _run(workspace)

    assert run.correct == {items[0].item_id: True, items[1].item_id: False,
                           items[2].item_id: True}
    assert run.calls_used == 3
    copy = workspace / "runs" / "exp-1" / "student_a"
    rows = [json.loads(line) for line in (copy / "predictions.jsonl").read_text().splitlines()]
    assert [row["correct"] for row in rows] == [True, False, True]
    written = "".join(path.read_text() for path in copy.rglob("*") if path.is_file())
    assert '"target"' not in written
    # The graded Inspect log, with the targets, sits beside the workspace.
    logs = list((private_root(workspace) / "exp-1" / "student_a" / "log").glob("*.json"))
    assert logs and not logs[0].is_relative_to(workspace)


def test_the_gateway_stops_a_policy_at_its_budget(workspace):
    (workspace / "policy" / "policy.py").write_text(
        "class Policy:\n"
        "    def run(self, questions, ctx):\n"
        "        while True:\n"
        "            ctx.student.generate('again')\n"
    )
    _, run = _run(workspace, budget=4)
    assert run.calls_used == 4
    assert run.budget_exhausted
    assert run.error.startswith("budget_exhausted")


def test_tools_run_the_policy_jailed_through_corrals_executor(monkeypatch):
    """dry_run_policy as Corral calls it: real task folder, jailed run, committed ledger."""
    from corral.core.resources import RESOURCE_CATALOG_METADATA_KEY
    from corral.core.state import EnvironmentState, ExecutionState, TaskState
    from corral.core.tool_catalog import TOOL_CATALOG_METADATA_KEY, TOOL_POLICY_METADATA_KEY
    from corral.runtime.tool_execution import ToolExecutor

    from inference_opt.env import create_environments

    monkeypatch.setattr(
        "inference_opt.runner.backend_for", lambda *args: MockBackend("ANSWER: A")
    )
    base = tempfile.mkdtemp(prefix="inference-opt-executor-", dir="/workspace")
    environment = create_environments(level=1, work_dir=base)["mmlu_pro_a"].for_task(
        "executor-test"
    )
    workspace = Path(environment.workspace_path)
    (workspace / "policy").mkdir(parents=True, exist_ok=True)
    (workspace / "policy" / "policy.py").write_text(
        "import os\n"
        "class Policy:\n"
        "    def solve(self, question, ctx):\n"
        "        ctx.log(f'{os.geteuid()} {os.getcwd()}')\n"
        "        return ctx.student.generate(question.text)\n"
    )
    seed = environment.initial_event(execution_id="executor-test").environment
    state = ExecutionState(
        through_commit_hash="a" * 64,
        execution_id="executor-test",
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
        environment=EnvironmentState(values=dict(seed)),
    )
    result = ToolExecutor(environment).execute(
        state, environment.tools["dry_run_policy"], {}, action_id="dry"
    )
    headline, payload = result.content.splitlines()[:2]
    assert headline.startswith("Dry run OK"), result.content
    traces = json.loads(payload)["traces"]
    uid, cwd = traces[0]["log"][0].split()
    assert uid != "0" and cwd == "/workspace"
    ledger = result.environment["resources"]["inference_state"]
    assert ledger["debug_runs"] == 1
    assert ledger["student_calls"] == 2
    assert (workspace / "runs" / "dry-1" / "student_a" / "summary.json").is_file()
