"""A policy run end to end, locally: host subprocess, gateway, controller grading.

No GPU and no network: the student is a test backend. The jail itself is
covered by ``test_isolation.py``, which needs Docker.
"""

from __future__ import annotations

import json
import threading

import pytest
from inference_opt import datasets
from inference_opt.client import StudentCompletion
from inference_opt.gateway import MockBackend, StudentGateway
from inference_opt.runner import private_root, run_policy

BENCHMARK = "mmlu_pro"


class EchoBackend:
    """Replies with the last message and remembers every request."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, messages, **params) -> StudentCompletion:
        with self._lock:
            self.requests.append({"messages": messages, **params})
        return StudentCompletion(
            text=messages[-1]["content"], finish_reason="stop", output_tokens=3
        )


def _items(benchmark: str = BENCHMARK, n: int = 3):
    return datasets.load_items(benchmark, "train")[:n]


def _run(
    tmp_path,
    source: str,
    *,
    benchmark: str = BENCHMARK,
    n: int = 3,
    budget: int = 20,
    policy_api: str = "primitive",
    backends=None,
    models=("m",),
    examples=None,
    time_limit_s=None,
):
    workspace = tmp_path / "workspace"
    (workspace / "policy").mkdir(parents=True, exist_ok=True)
    (workspace / "policy" / "policy.py").write_text(source, encoding="utf-8")
    config = {
        "benchmark": benchmark,
        "models": list(models),
        "model_specs": dict.fromkeys(models, "mockllm/model"),
        "policy_api": policy_api,
    }
    runs = run_policy(
        config=config,
        work_dir=workspace,
        run_id="exp-1",
        policy_dir=workspace / "policy",
        split="train",
        models=list(models),
        budget_per_model=budget,
        examples=examples or [],
        item_ids=[item.item_id for item in _items(benchmark, n)],
        backends=backends or {model: MockBackend() for model in models},
        time_limit_s=time_limit_s,
    )
    return workspace, runs


def _gold_policy(benchmark: str, wrong: set[str] = frozenset()) -> str:
    targets = datasets.load_targets(benchmark, "train")
    answers = {
        item.item_id: ("Z" if item.item_id in wrong else targets[item.item_id])
        for item in _items(benchmark)
    }
    marker = "[ANSWER]{}[/ANSWER]" if benchmark == "chembench" else "ANSWER: {}"
    return (
        f"ANSWERS = {answers!r}\n"
        "class Policy:\n"
        "    def run(self, questions, ctx):\n"
        "        for q in questions:\n"
        "            ctx.student.generate(q.text, question_id=q.id)\n"
        f"            ctx.submit(q.id, {marker!r}.format(ANSWERS[q.id]))\n"
    )


class TestGrading:
    def test_answers_are_graded_against_the_labels(self, tmp_path):
        wrong = {_items()[1].item_id}
        _, runs = _run(tmp_path, _gold_policy(BENCHMARK, wrong))
        run = runs["m"]
        assert run.error == "" and run.grading_error == ""
        assert run.correct == {
            item.item_id: item.item_id not in wrong for item in _items()
        }
        assert run.calls_used == 3

    def test_chembench_is_graded_by_its_own_scorer(self, tmp_path):
        _, runs = _run(tmp_path, _gold_policy("chembench"), benchmark="chembench")
        run = runs["m"]
        assert run.grading_error == ""
        assert all(run.correct.values()), run.correct

    def test_an_unsubmitted_question_counts_as_wrong(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n    def run(self, questions, ctx):\n        pass\n",
        )
        run = runs["m"]
        assert run.answers == {}
        assert run.n_correct == 0
        assert set(run.correct) == {item.item_id for item in _items()}


class TestWorkspaceCopy:
    def test_the_copy_holds_results_but_never_a_target(self, tmp_path):
        workspace, runs = _run(tmp_path, _gold_policy(BENCHMARK))
        folder = workspace / "runs" / "exp-1" / "m"
        assert {path.name for path in folder.iterdir()} == {
            "predictions.jsonl",
            "student_calls.jsonl",
            "summary.json",
        }
        rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
        assert [row["correct"] for row in rows] == [True, True, True]
        written = "".join(path.read_text() for path in folder.iterdir())
        assert '"target"' not in written
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["n_correct"] == 3 and summary["calls_used"] == 3

    def test_the_graded_log_is_kept_beside_the_workspace(self, tmp_path):
        workspace, _ = _run(tmp_path, _gold_policy(BENCHMARK))
        logs = list((private_root(workspace) / "exp-1" / "m" / "log").glob("*.json"))
        assert logs
        assert all(not log.resolve().is_relative_to(workspace.resolve()) for log in logs)

    def test_a_rerun_replaces_the_previous_copy(self, tmp_path):
        workspace, _ = _run(tmp_path, _gold_policy(BENCHMARK))
        stale = workspace / "runs" / "exp-1" / "m" / "stale.txt"
        stale.write_text("left over from a crashed action")
        _run(tmp_path, _gold_policy(BENCHMARK))
        assert not stale.exists()


class TestPolicyShapes:
    def test_solve_is_a_shortcut_for_one_question_at_a_time(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def solve(self, question, ctx):\n"
            "        ctx.log('solving')\n"
            "        return 'ANSWER: ' + ctx.student.generate('x')[:1]\n",
        )
        run = runs["m"]
        assert len(run.answers) == 3
        # Calls inside solve() are attributed to their question without asking.
        assert {qid: usage["calls"] for qid, usage in run.per_question.items()} == {
            item.item_id: 1 for item in _items()
        }
        assert all(run.logs[item.item_id] == ["solving"] for item in _items())

    def test_a_failing_solve_costs_one_question(self, tmp_path):
        bad = _items()[0].item_id
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def solve(self, question, ctx):\n"
            f"        if question.id == {bad!r}:\n"
            "            raise ValueError('boom')\n"
            "        return 'ANSWER: A'\n",
        )
        run = runs["m"]
        assert set(run.answers) == {item.item_id for item in _items()[1:]}
        assert run.errors[bad].startswith("ValueError: boom")
        assert run.n_crashed == 1

    def test_a_crash_in_run_keeps_what_was_submitted(self, tmp_path):
        first = _items()[0].item_id
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.submit(questions[0].id, 'ANSWER: A')\n"
            "        raise RuntimeError('halfway')\n",
        )
        run = runs["m"]
        assert list(run.answers) == [first]
        assert "halfway" in run.error

    def test_returning_a_mapping_submits_it(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        return {q.id: 'ANSWER: B' for q in questions}\n",
        )
        assert set(runs["m"].answers.values()) == {"ANSWER: B"}

    def test_returning_anything_else_is_an_error(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "def run(questions, ctx):\n    return ['ANSWER: B' for q in questions]\n",
        )
        assert "run() returned a list" in runs["m"].error

    def test_a_run_that_submits_nothing_is_reported(self, tmp_path):
        _, runs = _run(tmp_path, "def run(questions, ctx):\n    pass\n")
        assert "without submitting any answer" in runs["m"].error

    def test_revealed_examples_reach_the_policy(self, tmp_path):
        example = {**datasets.public_record(_items(n=10)[9]), "target": "C"}
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.log([e.answer for e in ctx.examples])\n",
            examples=[example],
        )
        assert runs["m"].logs["_run"] == ["['C']"]

    def test_a_policy_that_does_not_load_is_reported(self, tmp_path):
        _, runs = _run(tmp_path, "raise ImportError('missing helper')\n")
        run = runs["m"]
        assert "missing helper" in run.load_error
        assert run.correct == {}


class TestBudget:
    def test_the_gateway_stops_the_run_at_its_budget(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.submit(questions[0].id, 'ANSWER: A')\n"
            "        while True:\n"
            "            ctx.student.generate('again')\n",
            budget=5,
        )
        run = runs["m"]
        assert run.calls_used == 5
        assert run.budget_exhausted
        assert run.error.startswith("budget_exhausted")
        assert len(run.answers) == 1

    def test_ctx_budget_counts_down(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        before = ctx.budget\n"
            "        ctx.student.generate('x')\n"
            "        ctx.log(f'{before} {ctx.budget} {ctx.student.calls_used}')\n",
            budget=7,
        )
        assert runs["m"].logs["_run"] == ["7 6 1"]

    def test_each_model_gets_its_own_budget(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        while True:\n"
            "            ctx.student.generate('again')\n",
            budget=4,
            models=("student_a", "student_b"),
        )
        assert {model: run.calls_used for model, run in runs.items()} == {
            "student_a": 4,
            "student_b": 4,
        }


class TestStudentClient:
    def test_primitive_tasks_offer_only_generate(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.log(hasattr(ctx.student, 'sample'))\n",
        )
        assert runs["m"].logs["_run"] == ["False"]

    def test_sampling_charges_every_draw_and_batch_keeps_order(self, tmp_path):
        echo = EchoBackend()
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        drawn = ctx.student.sample('p', n=3)\n"
            "        batch = ctx.student.batch(['one', 'two', 'three'])\n"
            "        ctx.log(f'{len(drawn)} {batch}')\n",
            policy_api="enhanced",
            backends={"m": echo},
        )
        run = runs["m"]
        assert run.calls_used == 6
        assert run.logs["_run"] == ["3 ['one', 'two', 'three']"]

    def test_max_tokens_is_capped_by_the_task(self, tmp_path):
        echo = EchoBackend()
        _run(
            tmp_path,
            "MANIFEST = {'max_tokens_per_call': 1000}\n"
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.student.generate('default')\n"
            "        ctx.student.generate('huge', max_tokens=10**9)\n",
            backends={"m": echo},
        )
        assert [request["max_tokens"] for request in echo.requests] == [1000, 32768]

    def test_a_question_outside_the_run_is_refused(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.submit('somebody-elses-question', 'ANSWER: A')\n",
        )
        assert "not a question in this run" in runs["m"].error

    def test_a_dead_student_is_an_infrastructure_error(self, tmp_path):
        def unreachable(messages, **_):
            raise ConnectionError("connection refused")

        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.student.generate('x')\n",
            backends={"m": unreachable},
        )
        assert "connection refused" in runs["m"].infrastructure_error


def test_the_run_stops_at_its_time_limit(tmp_path):
    _, runs = _run(
        tmp_path,
        "import time\n"
        "class Policy:\n"
        "    def run(self, questions, ctx):\n"
        "        ctx.submit(questions[0].id, 'ANSWER: A')\n"
        "        time.sleep(60)\n",
        time_limit_s=3,
    )
    run = runs["m"]
    assert run.error.startswith("time limit")
    assert len(run.answers) == 1


class TestGateway:
    def _gateway(self, **overrides):
        settings = {
            "backend": MockBackend(),
            "budget": 3,
            "max_tokens_cap": 100,
            "question_ids": ["q1"],
            "policy_api": "primitive",
        }
        settings.update(overrides)
        return StudentGateway(**settings)

    def test_unknown_requests_and_roles_are_refused(self):
        gateway = self._gateway()
        with pytest.raises(ValueError, match="unknown request"):
            gateway.handle({"op": "read_labels"})
        with pytest.raises(ValueError, match="unsupported message role"):
            gateway.handle(
                {"op": "generate", "conversations": [[{"role": "tool", "content": "x"}]]}
            )

    def test_primitive_tasks_get_one_completion_per_call(self):
        gateway = self._gateway()
        message = [{"role": "user", "content": "x"}]
        with pytest.raises(ValueError, match="one completion per call"):
            gateway.handle({"op": "generate", "conversations": [message, message]})

    def test_a_request_bigger_than_the_budget_is_refused_whole(self):
        from inference_opt.api import BudgetExhausted

        gateway = self._gateway(policy_api="enhanced")
        message = [{"role": "user", "content": "x"}]
        with pytest.raises(BudgetExhausted):
            gateway.handle({"op": "generate", "conversations": [message] * 4})
        assert gateway.used == 0
