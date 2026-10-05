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
from inference_opt.runner import MAX_TOKENS_CAP, private_root, run_policy

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
    backends=None,
    models=("m",),
    examples=None,
    time_limit_s=None,
    save_private=None,
):
    workspace = tmp_path / "workspace"
    (workspace / "policy").mkdir(parents=True, exist_ok=True)
    (workspace / "policy" / "policy.py").write_text(source, encoding="utf-8")
    config = {
        "benchmark": benchmark,
        "models": list(models),
        "model_specs": dict.fromkeys(models, "mockllm/model"),
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
        save_private=save_private,
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
        f"            ctx.answer(q.id, {marker!r}.format(ANSWERS[q.id]))\n"
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

    def test_the_private_folder_is_archived_with_its_targets(self, tmp_path):
        import tarfile

        saved = []

        def save(path):
            saved.append(path.read_bytes())
            return "blob:sha256:" + "0" * 64

        _, runs = _run(tmp_path, _gold_policy(BENCHMARK), save_private=save)

        assert runs["m"].private_artifact == "blob:sha256:" + "0" * 64
        archive = tmp_path / "archive.tar.gz"
        archive.write_bytes(saved[0])
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            log = next(name for name in names if name.endswith(".json") and "/log/" in name)
            graded = tar.extractfile(log).read().decode()
        target = datasets.load_targets(BENCHMARK, "train")[_items()[0].item_id]
        assert {"exp-1/m/predictions.jsonl", "exp-1/m/student_calls.jsonl"} <= set(names)
        assert f'"target": "{target}"' in graded

    def test_a_rerun_replaces_the_previous_copy(self, tmp_path):
        workspace, _ = _run(tmp_path, _gold_policy(BENCHMARK))
        stale = workspace / "runs" / "exp-1" / "m" / "stale.txt"
        stale.write_text("left over from a crashed action")
        _run(tmp_path, _gold_policy(BENCHMARK))
        assert not stale.exists()


class TestPolicyShapes:
    def test_a_batch_keeps_its_order_and_attributes_each_entry(self, tmp_path):
        echo = EchoBackend()
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        texts = ctx.student.generate(\n"
            "            [{'prompt': q.id, 'question_id': q.id} for q in questions]\n"
            "        )\n"
            "        ctx.answer({q.id: 'ANSWER: ' + t for q, t in zip(questions, texts)})\n",
            backends={"m": echo},
        )
        run = runs["m"]
        ids = [item.item_id for item in _items()]
        assert [run.answers[qid] for qid in ids] == [f"ANSWER: {qid}" for qid in ids]
        assert {qid: usage["calls"] for qid, usage in run.per_question.items()} == dict.fromkeys(
            ids, 1
        )

    def test_a_failed_entry_comes_back_empty_and_the_rest_still_answer(self, tmp_path):
        bad = _items()[0].item_id

        def flaky(messages, **_):
            if messages[-1]["content"] == bad:
                raise ConnectionError("dropped")
            return StudentCompletion(text="ok", finish_reason="stop", output_tokens=1)

        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        texts = ctx.student.generate(\n"
            "            [{'prompt': q.id, 'question_id': q.id} for q in questions]\n"
            "        )\n"
            "        ctx.log(texts)\n"
            "        ctx.answer({q.id: 'ANSWER: A' for q, t in zip(questions, texts) if t})\n",
            backends={"m": flaky},
        )
        run = runs["m"]
        assert run.logs["_run"] == ["['', 'ok', 'ok']"]
        assert set(run.answers) == {item.item_id for item in _items()[1:]}
        assert "dropped" in run.errors[bad]
        assert run.error == ""
        assert run.infrastructure_error == ""

    def test_a_failed_single_call_raises(self, tmp_path):
        def unreachable(_messages, **_):
            raise ConnectionError("dropped")

        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        try:\n"
            "            ctx.student.generate('x')\n"
            "        except RuntimeError as exc:\n"
            "            ctx.log(str(exc))\n",
            backends={"m": unreachable},
        )
        assert "student request failed" in runs["m"].logs["_run"][0]

    def test_a_crash_in_run_keeps_what_was_answered(self, tmp_path):
        first = _items()[0].item_id
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.answer(questions[0].id, 'ANSWER: A')\n"
            "        raise RuntimeError('halfway')\n",
        )
        run = runs["m"]
        assert list(run.answers) == [first]
        assert "halfway" in run.error

    def test_answer_takes_one_question_or_a_dict(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.answer({q.id: 'ANSWER: B' for q in questions})\n"
            "        ctx.answer(questions[0].id, 'ANSWER: C')\n",
        )
        answers = runs["m"].answers
        assert answers[_items()[0].item_id] == "ANSWER: C"
        assert [answers[item.item_id] for item in _items()[1:]] == ["ANSWER: B"] * 2

    @pytest.mark.parametrize("returned", ["{q.id: 'ANSWER: B' for q in questions}", "[]"])
    def test_returning_anything_is_an_error(self, tmp_path, returned):
        _, runs = _run(tmp_path, f"def run(questions, ctx):\n    return {returned}\n")
        assert "record answers with ctx.answer" in runs["m"].error
        assert runs["m"].answers == {}

    def test_a_run_that_answers_nothing_is_reported(self, tmp_path):
        _, runs = _run(tmp_path, "def run(questions, ctx):\n    pass\n")
        assert "without recording any answer" in runs["m"].error

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

    def test_each_student_sees_its_own_baseline_reply(self, tmp_path):
        example = {
            **datasets.public_record(datasets.load_items(BENCHMARK, "examples")[0]),
            "target": "C",
            "baseline": {
                "m1": {"correct": True, "completion": "ANSWER: C"},
                "m2": {"correct": False, "completion": ""},
            },
        }
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        e = ctx.examples[0]\n"
            "        ctx.log(f'{e.baseline_correct} {e.baseline_completion!r}')\n",
            models=("m1", "m2"),
            examples=[example],
        )
        assert runs["m1"].logs["_run"] == ["True 'ANSWER: C'"]
        assert runs["m2"].logs["_run"] == ["False ''"]

    def test_an_unmeasured_baseline_reply_is_none_not_empty(self, tmp_path):
        example = {
            **datasets.public_record(datasets.load_items(BENCHMARK, "examples")[0]),
            "target": "C",
            "baseline": {"m": None},
        }
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        e = ctx.examples[0]\n"
            "        ctx.log(f'{e.baseline_correct} {e.baseline_completion}')\n",
            examples=[example],
        )
        assert runs["m"].logs["_run"] == ["None None"]

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
            "        ctx.answer(questions[0].id, 'ANSWER: A')\n"
            "        while True:\n"
            "            ctx.student.generate('again')\n",
            budget=5,
        )
        run = runs["m"]
        assert run.calls_used == 5
        assert run.budget_exhausted
        assert run.error.startswith("budget_exhausted")
        assert len(run.answers) == 1

    def test_a_batch_bigger_than_the_budget_sends_what_fits(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.log(ctx.student.generate(['a', 'b', 'c']))\n"
            "        ctx.log(ctx.budget)\n"
            "        ctx.student.generate('d')\n",
            budget=2,
            backends={"m": EchoBackend()},
        )
        run = runs["m"]
        assert run.logs["_run"][0] == "budget exhausted: sent 2 of 3 prompts"
        assert run.logs["_run"][1:] == ["['a', 'b', '']", "0"]
        assert run.calls_used == 2
        assert run.budget_exhausted
        assert run.error.startswith("budget_exhausted")

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
    def test_entries_set_their_own_settings_over_the_shared_ones(self, tmp_path):
        echo = EchoBackend()
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        texts = ctx.student.generate(\n"
            "            ['plain', {'prompt': 'hot', 'temperature': 0.9, 'seed': 7}],\n"
            "            temperature=0.2,\n"
            "        )\n"
            "        ctx.log(texts)\n",
            backends={"m": echo},
        )
        run = runs["m"]
        assert run.calls_used == 2
        assert run.logs["_run"] == ["['plain', 'hot']"]
        settings = {
            r["messages"][-1]["content"]: (r["temperature"], r["seed"]) for r in echo.requests
        }
        assert settings == {"plain": (0.2, None), "hot": (0.9, 7)}

    def test_a_message_list_is_one_prompt_not_a_batch(self, tmp_path):
        echo = EchoBackend()
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        text = ctx.student.generate(\n"
            "            [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'}]\n"
            "        )\n"
            "        ctx.log(repr(text))\n",
            backends={"m": echo},
        )
        assert runs["m"].logs["_run"] == ["'u'"]
        assert [m["role"] for m in echo.requests[0]["messages"]] == ["system", "user"]

    def test_an_unknown_entry_setting_is_refused(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.student.generate([{'prompt': 'x', 'temprature': 1.0}])\n",
        )
        assert "may set only" in runs["m"].error
        assert runs["m"].calls_used == 0

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
        assert [request["max_tokens"] for request in echo.requests] == [
            1000,
            MAX_TOKENS_CAP,
        ]

    def test_calls_cut_off_at_the_limit_are_counted(self, tmp_path):
        def still_reasoning(messages, **params):
            del messages
            return StudentCompletion(
                "", finish_reason="length", output_tokens=params["max_tokens"]
            )

        workspace, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.student.generate(\n"
            "            [{'prompt': q.text, 'question_id': q.id} for q in questions],\n"
            "            max_tokens=20000,\n"
            "        )\n"
            "        ctx.answer({q.id: 'ANSWER: A' for q in questions})\n",
            backends={"m": still_reasoning},
        )
        run = runs["m"]
        assert (run.truncated_calls, run.capped_requests) == (3, 3)
        summary = json.loads((workspace / "runs/exp-1/m/summary.json").read_text())
        assert summary["truncated_calls"] == 3
        assert summary["capped_requests"] == 3
        predictions = (workspace / "runs/exp-1/m/predictions.jsonl").read_text()
        assert all(
            json.loads(line)["truncated_calls"] == 1 for line in predictions.splitlines()
        )

    def test_a_question_outside_the_run_is_refused(self, tmp_path):
        _, runs = _run(
            tmp_path,
            "class Policy:\n"
            "    def run(self, questions, ctx):\n"
            "        ctx.answer('somebody-elses-question', 'ANSWER: A')\n",
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
        "        ctx.answer(questions[0].id, 'ANSWER: A')\n"
        "        time.sleep(60)\n",
        time_limit_s=3,
    )
    run = runs["m"]
    assert run.error.startswith("time limit")
    assert len(run.answers) == 1


class TestTokenNote:
    def test_run_results_say_how_many_calls_hit_the_limit(self):
        from inference_opt.runner import ModelRun
        from inference_opt.tools import _token_note

        run = ModelRun(
            model="m", split="train", question_ids=[], calls_used=10,
            truncated_calls=4, capped_requests=2,
        )
        note = _token_note([run])
        assert f"4 of 10 student calls hit the {MAX_TOKENS_CAP}-token limit" in note
        assert "2 request(s) asked for more than" in note
        assert _token_note([ModelRun(model="m", split="train", question_ids=[])]) == ""


class TestGateway:
    def _gateway(self, **overrides):
        settings = {
            "backend": MockBackend(),
            "budget": 3,
            "max_tokens_cap": 100,
            "question_ids": ["q1"],
        }
        settings.update(overrides)
        return StudentGateway(**settings)

    @staticmethod
    def _request(**fields):
        return {"messages": [{"role": "user", "content": "x"}], **fields}

    def test_unknown_requests_fields_and_roles_are_refused(self):
        gateway = self._gateway()
        with pytest.raises(ValueError, match="unknown request"):
            gateway.handle({"op": "read_labels"})
        with pytest.raises(ValueError, match="unknown request field"):
            gateway.handle({"op": "generate", "requests": [self._request(answer="A")]})
        with pytest.raises(ValueError, match="unsupported message role"):
            gateway.handle(
                {
                    "op": "generate",
                    "requests": [{"messages": [{"role": "tool", "content": "x"}]}],
                }
            )
        assert gateway.used == 0

    def test_a_request_bigger_than_the_budget_sends_what_fits(self):
        gateway = self._gateway()
        reply = gateway.handle({"op": "generate", "requests": [self._request()] * 4})
        assert reply["errors"] == ["", "", "", "budget exhausted"]
        assert reply["completions"][3] == ""
        assert reply["remaining"] == 0
        assert gateway.used == 3
        assert gateway.exhausted

    def test_nothing_left_is_refused(self):
        from inference_opt.api import BudgetExhausted

        gateway = self._gateway(budget=0)
        with pytest.raises(BudgetExhausted):
            gateway.handle({"op": "generate", "requests": [self._request()]})

    def test_concurrency_is_capped_for_the_whole_run(self):
        import time

        lock = threading.Lock()
        state = {"now": 0, "most": 0}

        def slow(_messages, **_):
            with lock:
                state["now"] += 1
                state["most"] = max(state["most"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return StudentCompletion(text="ok", finish_reason="stop", output_tokens=1)

        gateway = self._gateway(backend=slow, budget=40, concurrency=3)
        body = {"op": "generate", "requests": [self._request()] * 10}
        threads = [threading.Thread(target=gateway.handle, args=(body,)) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert gateway.used == 30
        assert state["most"] == 3
