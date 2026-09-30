"""The litellm-backed Inspect provider that calls students on vLLM servers."""

from __future__ import annotations

from types import SimpleNamespace

import anyio
import pytest
from inference_opt.datasets import write_jsonl
from inference_opt.eval_runner import run_in_process
from inference_opt.eval_runner.spec import RunSpec
from inference_opt.eval_runner.student_model import student_model
from inspect_ai.model import ChatMessageUser, GenerateConfig


def fake_response(content, reasoning="", finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=reasoning),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=40,
            total_tokens=51,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        ),
    )


@pytest.fixture
def calls(monkeypatch):
    seen = []

    async def acompletion(**kwargs):
        seen.append(kwargs)
        return fake_response("ANSWER: 4", reasoning="2+2 is 4")

    monkeypatch.setattr("litellm.acompletion", acompletion)
    return seen


def test_answer_and_reasoning_are_kept_apart(calls):
    model = student_model("vllm/Qwen/Qwen3-8B", "http://127.0.0.1:8085/v1", None, 60.0)
    config = GenerateConfig(temperature=0.7, max_tokens=512, seed=3)
    output = anyio.run(lambda: model.generate([ChatMessageUser(content="2+2?")], config=config))
    assert output.completion == "ANSWER: 4"
    parts = output.choices[0].message.content
    assert [p.reasoning for p in parts if p.type == "reasoning"] == ["2+2 is 4"]
    assert output.choices[0].message.text == "ANSWER: 4"
    assert output.usage.output_tokens == 40
    assert output.usage.reasoning_tokens == 30
    request = calls[0]
    assert request["model"] == "hosted_vllm/Qwen/Qwen3-8B"
    assert request["api_base"] == "http://127.0.0.1:8085/v1"
    assert request["n"] == 1
    assert request["timeout"] == 60.0
    assert (request["temperature"], request["max_tokens"], request["seed"]) == (0.7, 512, 3)


def test_a_cut_off_answer_reports_max_tokens(monkeypatch):
    async def acompletion(**kwargs):
        return fake_response(None, reasoning="still thinking", finish_reason="length")

    monkeypatch.setattr("litellm.acompletion", acompletion)
    model = student_model("vllm/Qwen/Qwen3-8B", "http://x/v1", None, None)
    output = anyio.run(lambda: model.generate([ChatMessageUser(content="q")]))
    assert output.completion == ""
    assert output.stop_reason == "max_tokens"


def test_a_policy_evaluation_runs_through_the_student_provider(tmp_path, calls):
    questions = tmp_path / "q.jsonl"
    write_jsonl(questions, [
        {"item_id": f"gsm8k:q{i}", "benchmark": "gsm8k", "question": "2+2?",
         "answer_format": "numeric", "target": "4", "index": i}
        for i in range(3)
    ])
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "policy.py").write_text(
        "class Policy:\n    def solve(self, q, ctx): return ctx.student.generate(q.text)\n"
    )
    spec = RunSpec(
        run_id="s", policy_dir=str(policy), questions_path=str(questions),
        out_dir=str(tmp_path / "out"), model_spec="vllm/Qwen/Qwen3-8B",
        base_url="http://127.0.0.1:8085/v1", total_calls=6, max_calls_per_question=2,
        benchmark="gsm8k", split="train", max_connections=4,
    )
    summary = run_in_process(spec)
    assert summary.ok, summary.error
    assert summary.n_answered == 3
    assert summary.calls_used == 3
    assert summary.output_tokens == 120
    assert len(calls) == 3
