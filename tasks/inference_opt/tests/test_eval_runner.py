"""The policy evaluator, end to end, with no GPU and no network (``mockllm``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inference_opt.datasets import write_jsonl
from inference_opt.eval_runner import run_in_process
from inference_opt.eval_runner.spec import RunSpec

QUESTIONS = [
    {
        "item_id": "gsm8k:q1",
        "benchmark": "gsm8k",
        "question": "2+2?",
        "answer_format": "numeric",
        "target": "4",
        "index": 0,
    },
    {
        "item_id": "gsm8k:q2",
        "benchmark": "gsm8k",
        "question": "3+3?",
        "answer_format": "numeric",
        "target": "6",
        "index": 1,
    },
    {
        "item_id": "mmlu_pro:q3",
        "benchmark": "mmlu_pro",
        "question": "Capital of France?",
        "answer_format": "mcq_single",
        "options": ["Paris", "Berlin"],
        "target": "A",
        "index": 2,
    },
]


@pytest.fixture
def questions_file(tmp_path):
    path = tmp_path / "questions.jsonl"
    write_jsonl(path, QUESTIONS)
    return path


def make_spec(tmp_path, questions_file, source, **overrides):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "policy.py").write_text(source, encoding="utf-8")
    defaults = {
        "run_id": "t1",
        "policy_dir": str(policy_dir),
        "questions_path": str(questions_file),
        "out_dir": str(tmp_path / "out"),
        "model_spec": "mockllm/model",
        "total_calls": 40,
        "max_calls_per_question": 4,
        "benchmark": "gsm8k",
        "split": "train",
    }
    defaults.update(overrides)
    return RunSpec(**defaults)


class TestEndToEnd:
    def test_a_working_policy_answers_every_question(self, tmp_path, questions_file):
        spec = make_spec(
            tmp_path,
            questions_file,
            "class Policy:\n"
            "    def solve(self, q, ctx):\n"
            "        return ctx.student.generate(q.text)\n",
        )
        summary = run_in_process(spec)
        assert summary.ok, summary.error
        assert summary.n_questions == 3
        assert summary.n_answered == 3
        assert summary.calls_used == 3

    def test_predictions_and_logs_are_written(self, tmp_path, questions_file):
        spec = make_spec(
            tmp_path,
            questions_file,
            "class Policy:\n    def solve(self, q, ctx): return 'ANSWER: A'\n",
        )
        run_in_process(spec)
        with Path(spec.predictions_path).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        assert {row["item_id"] for row in rows} == {q["item_id"] for q in QUESTIONS}
        assert list(__import__("pathlib").Path(spec.log_dir).glob("*.json"))

    def test_sampling_charges_every_draw(self, tmp_path, questions_file):
        """`num_choices=n` is one request but n generations, so it costs n."""
        spec = make_spec(
            tmp_path,
            questions_file,
            "class Policy:\n"
            "    def solve(self, q, ctx):\n"
            "        return ctx.student.sample(q.text, n=3)[0]\n",
            policy_api="enhanced",
        )
        summary = run_in_process(spec)
        assert summary.calls_used == 9

    def test_shared_memory_forces_sequential_execution(self, tmp_path, questions_file):
        spec = make_spec(
            tmp_path,
            questions_file,
            "MANIFEST = {'memory': 'shared'}\n"
            "class Policy:\n"
            "    def solve(self, q, ctx):\n"
            "        ctx.memory.append('seen', q.id)\n"
            "        return str(len(ctx.memory.get('seen', [])))\n",
            policy_api="enhanced",
        )
        summary = run_in_process(spec)
        assert summary.execution == "sequential"

    def test_memory_is_read_only_without_the_manifest_flag(
        self, tmp_path, questions_file
    ):
        spec = make_spec(
            tmp_path,
            questions_file,
            "class Policy:\n"
            "    def solve(self, q, ctx):\n"
            "        ctx.memory.set('x', 1)\n"
            "        return 'ANSWER: A'\n",
            policy_api="enhanced",
        )
        summary = run_in_process(spec)
        assert summary.n_crashed == 3

    def test_setup_runs_once_and_sees_revealed_answers(self, tmp_path, questions_file):
        revealed = tmp_path / "revealed.jsonl"
        write_jsonl(
            revealed,
            [
                {
                    "item_id": "gsm8k:q0",
                    "question": "1+1?",
                    "answer_format": "numeric",
                    "target": "2",
                    "baseline_correct": False,
                },
            ],
        )
        spec = make_spec(
            tmp_path,
            questions_file,
            "MANIFEST = {'memory': 'shared', 'setup_calls': 2}\n"
            "class Policy:\n"
            "    def setup(self, ctx):\n"
            "        ctx.memory.set('n', len(ctx.train_examples))\n"
            "        ctx.memory.set('gold', ctx.train_examples[0].answer)\n"
            "    def solve(self, q, ctx):\n"
            "        return 'ANSWER: ' + ctx.memory.get('gold', 'X')\n",
            revealed_path=str(revealed),
            setup_calls=2,
            policy_api="enhanced",
        )
        summary = run_in_process(spec)
        assert summary.ok, summary.error
        rows = {
            row["item_id"]: row["answer"]
            for row in (
                json.loads(line)
                for line in Path(spec.predictions_path).read_text().splitlines()
                if line.strip()
            )
        }
        # The policy echoed the revealed gold answer "2", proving setup ran and
        # saw the labelled example. The evaluator preserves that raw completion;
        # Inspect's choice scorer grades it incorrect because "2" is not a letter.
        assert rows["gsm8k:q1"] == "ANSWER: 2"
        assert rows["gsm8k:q2"] == "ANSWER: 2"
        assert rows["mmlu_pro:q3"] == "ANSWER: 2"


class TestFailureHandling:
    def test_a_crashing_policy_costs_items_not_the_run(self, tmp_path, questions_file):
        spec = make_spec(
            tmp_path,
            questions_file,
            "class Policy:\n    def solve(self, q, ctx): raise ValueError('boom')\n",
        )
        summary = run_in_process(spec)
        assert summary.ok
        assert summary.n_crashed == 3
        assert summary.n_answered == 3

    def test_a_fallback_survives_budget_exhaustion(self, tmp_path, questions_file):
        """Remaining questions still run, so a defensive policy still scores."""
        spec = make_spec(
            tmp_path,
            questions_file,
            "class Policy:\n"
            "    def solve(self, q, ctx):\n"
            "        ctx.scratch['fallback'] = 'ANSWER: 4'\n"
            "        return ctx.student.generate(q.text)\n",
            total_calls=1,
            max_calls_per_question=1,
        )
        summary = run_in_process(spec)
        assert summary.budget_exhausted_at is not None
        rows = [
            json.loads(line)
            for line in Path(spec.predictions_path).read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 3
        assert any(row["answer"] == "ANSWER: 4" for row in rows)

    def test_a_policy_with_no_solve_is_refused(self, tmp_path, questions_file):
        spec = make_spec(tmp_path, questions_file, "answer = 1\n")
        summary = run_in_process(spec)
        assert not summary.ok
        assert "could not be loaded" in summary.error


class TestEnvironmentScrubbing:
    def test_provider_credentials_are_removed(self, monkeypatch):
        from inference_opt.eval_runner.__main__ import scrub_environment

        monkeypatch.setenv("OPENAI_API_KEY", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        monkeypatch.setenv("SOMETHING_TOKEN", "secret")
        monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        removed = scrub_environment()
        import os

        assert "OPENAI_API_KEY" in removed
        assert "ANTHROPIC_API_KEY" in removed
        assert "SOMETHING_TOKEN" in removed
        assert os.environ.get("VLLM_BASE_URL") == "http://127.0.0.1:8000/v1"

    def test_policy_cannot_read_provider_credentials(
        self, tmp_path, questions_file, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        monkeypatch.setenv("VLLM_API_KEY", "vllm-secret")
        spec = make_spec(
            tmp_path,
            questions_file,
            "import os\n"
            "class Policy:\n"
            "    def solve(self, q, ctx):\n"
            "        return os.environ.get('OPENAI_API_KEY') or os.environ.get('VLLM_API_KEY') or 'missing'\n",
        )
        run_in_process(spec)
        rows = [
            json.loads(line)
            for line in Path(spec.predictions_path).read_text().splitlines()
            if line.strip()
        ]
        assert {row["answer"] for row in rows} == {"missing"}
