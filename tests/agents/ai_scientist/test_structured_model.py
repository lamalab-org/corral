from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier

import pytest

from corral.agents.ai_scientist.search.nodes import ExperimentDecision
from corral.agents.ai_scientist.workers.base import (
    LiteLLMStructuredModel,
    LLMBudgetExceeded,
)
from corral.agents.ai_scientist.workers.synthesizer import FinalAnswer
from corral.agents.schema import BudgetExhaustedError


@dataclass
class FakeResponse:
    content: str = '{"final_answer": "42"}'
    usage: dict[str, int] | None = None
    id: str = "response-1"


class Owner:
    def __init__(self):
        self.messages = []
        self.turn_usages = []
        self.token_usage = None
        self.accumulated_usage = []

    def _accumulate_token_usage(self, usage):
        self.accumulated_usage.append(usage)

    def _record_turn_usage(self, usage):
        self.turn_usages.append(usage)


def gateway(*, max_calls=4, use_structured_output=True, completion_runner=None):
    return LiteLLMStructuredModel(
        owner=Owner(),
        default_model="test-model",
        evaluator_model="critic-model",
        system_prompt="system",
        temperature=0,
        api_endpoint=None,
        max_calls=max_calls,
        use_structured_output=use_structured_output,
        completion_runner=completion_runner or (lambda **_kwargs: FakeResponse()),
    )


def test_structured_fallback_counts_both_physical_provider_requests():
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            raise ValueError("response_format is not supported by this provider")
        return FakeResponse(usage={"total_tokens": 7})

    model = gateway(completion_runner=fake_llm_call)

    result = model.generate("prompt", FinalAnswer, purpose="answer")

    assert result.final_answer == "42"
    assert model.call_count == 2
    assert model.token_count == 7
    assert len(calls) == 2
    assert model.use_structured_output is False


def test_fallback_cannot_exceed_the_physical_request_budget():
    calls = []

    def reject_structured(**kwargs):
        calls.append(kwargs)
        raise ValueError("structured output is unsupported")

    model = gateway(max_calls=1, completion_runner=reject_structured)

    with pytest.raises(LLMBudgetExceeded, match="json_fallback"):
        model.generate("prompt", FinalAnswer, purpose="answer")

    assert model.call_count == 1
    assert len(calls) == 1


def test_transient_failure_does_not_disable_structured_output():
    def time_out(**kwargs):
        raise TimeoutError("temporary provider timeout")

    model = gateway(completion_runner=time_out)

    with pytest.raises(TimeoutError, match="temporary"):
        model.generate("prompt", FinalAnswer)

    assert model.call_count == 1
    assert model.use_structured_output is True


def test_text_mode_counts_one_physical_request():
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    model = gateway(use_structured_output=False, completion_runner=fake_llm_call)

    model.generate("prompt", FinalAnswer)

    assert model.call_count == 1
    assert len(calls) == 1
    assert "response_format" not in calls[0]


def test_transcript_names_every_role_without_mutating_api_messages():
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    model = gateway(completion_runner=fake_llm_call)

    model.generate("prompt", FinalAnswer, purpose="evaluate_node_0007")

    assert [message["role"] for message in model.owner.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert all(
        message["name"] == "evaluate_node_0007" for message in model.owner.messages
    )
    assert all("name" not in message for message in calls[0]["messages"])
    assert model.owner.turn_usages == [{}]


def test_multimodal_generation_attaches_local_images(tmp_path):
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    image = tmp_path / "curve.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nplot")
    model = gateway(completion_runner=fake_llm_call)

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


@pytest.mark.parametrize("use_structured_output", [True, False])
@pytest.mark.parametrize(
    ("invalid", "error_text"),
    [
        ('{"decision":"act" "rationale":"measure"}', "Expecting ',' delimiter"),
        ('{"decision":"finish","rationale" "done"}', "Expecting ':' delimiter"),
        ('{"decision":"act","rationale":"measure"}', "An act decision requires"),
        (
            '{"decision":"finish","rationale":"done","arguments":"{broken}"}',
            "arguments",
        ),
        ("No JSON was produced.", "no JSON object"),
        ("[]", "must be one JSON object"),
        ("", "no JSON object"),
    ],
)
def test_invalid_decisions_are_corrected_on_the_next_model_call(
    invalid, error_text, use_structured_output
):
    calls = []
    corrected = '{"decision":"finish","rationale":"done","conclusion":"42"}'

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        return FakeResponse(
            content=invalid if len(calls) == 1 else corrected,
            usage={"total_tokens": 7},
        )

    model = gateway(
        max_calls=2,
        use_structured_output=use_structured_output,
        completion_runner=fake_llm_call,
    )
    result = model.generate("Use the measured evidence.", ExperimentDecision)

    assert result.conclusion == "42"
    assert model.call_count == 2
    assert model.remaining_calls == 0
    assert model.token_count == 14
    assert model.owner.turn_usages == [{"total_tokens": 7}] * 2
    assert model.owner.accumulated_usage == [{"total_tokens": 7}] * 2
    assert len(calls[0]["messages"]) == 2
    correction_messages = calls[1]["messages"]
    assert correction_messages[:2] == calls[0]["messages"]
    assert correction_messages[2] == {"role": "assistant", "content": invalid}
    assert correction_messages[3]["role"] == "user"
    assert error_text in correction_messages[3]["content"]
    assert '"title": "ExperimentDecision"' in correction_messages[3]["content"]
    assert all("name" not in message for message in correction_messages)
    assert ("response_format" in calls[1]) is use_structured_output
    assert model.owner.messages[2]["content"] == invalid
    assert model.owner.messages[-1]["content"] == corrected


def test_persistent_invalid_responses_stop_at_the_shared_call_budget():
    calls = []

    def invalid_response(**kwargs):
        calls.append(kwargs)
        return FakeResponse(content="invalid", usage={"total_tokens": 5})

    model = gateway(max_calls=3, completion_runner=invalid_response)

    with pytest.raises(LLMBudgetExceeded, match="LLM-call budget exhausted \\(3\\)"):
        model.generate("prompt", FinalAnswer)

    assert len(calls) == model.call_count == 3
    assert model.token_count == 15
    assert len(model.owner.turn_usages) == 3
    assert [len(call["messages"]) for call in calls] == [2, 4, 6]


def test_correction_after_unsupported_schema_counts_every_request():
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            raise ValueError("response_format is not supported")
        return FakeResponse(
            content="invalid" if len(calls) == 2 else '{"final_answer":"42"}'
        )

    model = gateway(max_calls=3, completion_runner=fake_llm_call)

    assert model.generate("prompt", FinalAnswer).final_answer == "42"
    assert model.call_count == 3
    assert "response_format" not in calls[2]
    assert calls[2]["messages"][2]["content"] == "invalid"


@pytest.mark.parametrize(
    "error",
    [
        BudgetExhaustedError("credits exhausted"),
        TimeoutError("provider timeout"),
        ValueError("invalid provider configuration"),
    ],
)
def test_provider_failures_during_correction_propagate(error):
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return FakeResponse(content="invalid")
        raise error

    model = gateway(completion_runner=fake_llm_call)

    with pytest.raises(type(error), match=str(error)):
        model.generate("prompt", FinalAnswer)

    assert len(calls) == model.call_count == 2
    assert model.use_structured_output is True


def test_multimodal_correction_preserves_images_and_selected_model(tmp_path):
    calls = []

    def fake_llm_call(**kwargs):
        calls.append(kwargs)
        return FakeResponse(content="invalid") if len(calls) == 1 else FakeResponse()

    image = tmp_path / "curve.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nplot")
    model = gateway(completion_runner=fake_llm_call)
    model.llm_kwargs = {"reasoning_effort": "medium"}

    result = model.generate_multimodal(
        "inspect the plot",
        FinalAnswer,
        image_paths=[str(image)],
        model="critic-model",
    )

    assert result.final_answer == "42"
    assert calls[1]["messages"][:2] == calls[0]["messages"]
    assert calls[1]["messages"][1]["content"][1]["type"] == "image_url"
    assert calls[1]["model"] == "critic-model"
    assert calls[1]["reasoning_effort"] == "medium"


def test_extractable_json_does_not_require_another_model_call():
    model = gateway(
        max_calls=1,
        completion_runner=lambda **_kwargs: FakeResponse(
            content='```json\n{"final_answer":"42"}\n```'
        ),
    )

    assert model.generate("prompt", FinalAnswer).final_answer == "42"
    assert model.call_count == 1


def test_parallel_workers_keep_correction_contexts_separate():
    first_responses = Barrier(2, timeout=10)

    def fake_llm_call(**kwargs):
        messages = kwargs["messages"]
        prompt = messages[1]["content"]
        if len(messages) == 2:
            first_responses.wait()
            content = f"invalid response for {prompt}"
        else:
            assert messages[2]["content"] == f"invalid response for {prompt}"
            content = f'{{"final_answer":"{prompt}"}}'
        return FakeResponse(content=content, usage={"total_tokens": 1})

    model = gateway(max_calls=4, completion_runner=fake_llm_call)

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(
                lambda prompt: model.generate(prompt, FinalAnswer),
                ["first experiment", "second experiment"],
            )
        )

    assert [result.final_answer for result in results] == [
        "first experiment",
        "second experiment",
    ]
    assert model.call_count == 4
    assert model.token_count == 4
    assert len(model.owner.turn_usages) == 4
