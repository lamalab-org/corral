from dataclasses import dataclass

import pytest

from corral.agents.ai_scientist.workers import base
from corral.agents.ai_scientist.workers.base import (
    LiteLLMStructuredModel,
    LLMBudgetExceeded,
)
from corral.agents.ai_scientist.workers.synthesizer import FinalAnswer


@dataclass
class FakeResponse:
    content: str = '{"final_answer": "42"}'
    usage: dict[str, int] | None = None
    id: str = "response-1"


class Owner:
    def __init__(self):
        self.messages = []
        self.token_usage = None
        self.accumulated_usage = []

    def _accumulate_token_usage(self, usage):
        self.accumulated_usage.append(usage)


def gateway(*, max_calls=4, use_structured_output=True):
    return LiteLLMStructuredModel(
        owner=Owner(),
        default_model="test-model",
        evaluator_model="critic-model",
        system_prompt="system",
        temperature=0,
        api_endpoint=None,
        max_calls=max_calls,
        use_structured_output=use_structured_output,
    )


def test_structured_fallback_counts_both_physical_provider_requests(monkeypatch):
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            raise ValueError("response_format is not supported by this provider")
        return FakeResponse(usage={"total_tokens": 7})

    monkeypatch.setattr(base, "llm_call", fake_llm_call)
    model = gateway()

    result = model.generate("prompt", FinalAnswer, purpose="answer")

    assert result.final_answer == "42"
    assert model.call_count == 2
    assert model.token_count == 7
    assert len(calls) == 2
    assert model.use_structured_output is False


def test_fallback_cannot_exceed_the_physical_request_budget(monkeypatch):
    calls = []

    def reject_structured(**kwargs):
        calls.append(kwargs)
        raise ValueError("structured output is unsupported")

    monkeypatch.setattr(base, "llm_call", reject_structured)
    model = gateway(max_calls=1)

    with pytest.raises(LLMBudgetExceeded, match="json_fallback"):
        model.generate("prompt", FinalAnswer, purpose="answer")

    assert model.call_count == 1
    assert len(calls) == 1


def test_transient_failure_does_not_disable_structured_output(monkeypatch):
    def time_out(**kwargs):
        raise TimeoutError("temporary provider timeout")

    monkeypatch.setattr(base, "llm_call", time_out)
    model = gateway()

    with pytest.raises(TimeoutError, match="temporary"):
        model.generate("prompt", FinalAnswer)

    assert model.call_count == 1
    assert model.use_structured_output is True


def test_text_mode_counts_one_physical_request(monkeypatch):
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(base, "llm_call", fake_llm_call)
    model = gateway(use_structured_output=False)

    model.generate("prompt", FinalAnswer)

    assert model.call_count == 1
    assert len(calls) == 1
    assert "response_format" not in calls[0]


def test_multimodal_generation_attaches_local_images(monkeypatch, tmp_path):
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    image = tmp_path / "curve.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nplot")
    monkeypatch.setattr(base, "llm_call", fake_llm_call)
    model = gateway()

    result = model.generate_multimodal(
        "inspect the convergence curve",
        FinalAnswer,
        image_paths=[str(image)],
        purpose="visual_evaluation",
    )

    assert result.final_answer == "42"
    content = calls[0]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "inspect the convergence curve"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
