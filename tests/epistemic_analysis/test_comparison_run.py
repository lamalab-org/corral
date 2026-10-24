"""Checks for annotation transport and trace reconstruction."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from corral.epistemic_analysis.annotators import llm
from corral.epistemic_analysis.decisions import DecisionsAnnotator
from corral.epistemic_analysis.motifs import MOTIFS, STATUS_DESCRIPTIONS
from corral.epistemic_analysis.sources import langfuse


def test_json_quotes_can_contain_markdown_fences():
    raw = json.dumps(
        {"nodes": [{"text": "Example ```python\nprint('x')\n``` output."}]}
    )
    assert json.loads(llm._extract_json_text(raw)) == json.loads(raw)
    assert json.loads(
        llm._extract_json_text("```json\n" + raw + "\n```")
    ) == json.loads(raw)


@pytest.mark.parametrize("refused", [False, True])
def test_llm_sends_medium_and_omits_temperature(monkeypatch, refused):
    captured = []

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            pass

    async def completion(**kwargs):
        captured.append(kwargs)
        return Stream()

    fake = SimpleNamespace(
        acompletion=completion,
        token_counter=lambda **kwargs: 1,
        stream_chunk_builder=lambda *args, **kwargs: {
            "model": "claude-opus-5-5",
            "usage": {},
            "choices": [
                {
                    "finish_reason": "content_filter" if refused else "stop",
                    "message": {"content": "" if refused else '{"nodes": []}'},
                }
            ],
        },
    )
    monkeypatch.setattr(llm, "_load_litellm", lambda: fake)

    async def invoke():
        return await llm.llm_json_call_async(
            "anthropic/claude-opus-5-5",
            "s",
            "u",
            temperature=None,
            reasoning_effort="medium",
            max_output_tokens=32000,
        )

    if refused:
        with pytest.raises(llm.LLMRefusalError):
            asyncio.run(invoke())
    else:
        asyncio.run(invoke())
    assert len(captured) == 1
    assert captured[0]["reasoning_effort"] == "medium"
    assert captured[0]["max_tokens"] == 32000
    assert "temperature" not in captured[0]


def test_direct_classification_can_skip_explanation_calls():
    payload = {
        "model": "gpt-6-luna",
        "answers": [
            {
                "name": m.name,
                "type": "choice",
                "choice": "active",
                "confidence": 0.8,
                "probabilities": [
                    {"value": s, "probability": 0.8 if s == "active" else 0.05}
                    for s in STATUS_DESCRIPTIONS
                ],
            }
            for m in MOTIFS
        ],
        "usage": {},
    }
    create = AsyncMock(
        return_value=SimpleNamespace(model_dump=lambda **kwargs: payload)
    )
    client = SimpleNamespace(decisions=SimpleNamespace(create=create))
    report = asyncio.run(
        DecisionsAnnotator(client=client, include_explanations=False).annotate(
            [{"role": "user", "content": "A task."}]
        )
    )
    assert len(report.assessments) == 18
    assert report.raw_explanation_response is None
    assert report.usage["llm_calls"] == 1


def test_media_restoration_preserves_original_and_deduplicates_downloads():
    reference = "@@@langfuseMedia:type=text/plain|id=media1|source=field_size_limit@@@"
    calls = []
    api = SimpleNamespace(
        media=SimpleNamespace(
            get=lambda identity: SimpleNamespace(url="https://example.invalid/signed")
        )
    )

    def get(url):
        calls.append(url)
        return SimpleNamespace(
            text='[{"role":"user","content":"restored"}]', raise_for_status=lambda: None
        )

    original = [{"input": json.dumps(reference), "output": reference}]
    result = langfuse.resolve_observation_media(original, api, SimpleNamespace(get=get))
    assert result[0]["input"] == [{"role": "user", "content": "restored"}]
    assert result[0]["output"] == result[0]["input"]
    assert result[0]["_media_references"]["input"] == original[0]["input"]
    assert len(calls) == 1


def test_partly_executed_react_response_preserves_proposals():
    message = {
        "role": "assistant",
        "content": "<action>a</action><action_input>{}</action_input>\n<action>b</action><action_input>{}</action_input>",
        "tool_calls": [{"function": {"name": "a", "arguments": "{}"}}],
    }
    result = langfuse._render(message)["content"]
    assert result.count("<action>") == 1
    assert "<proposed_action>b</proposed_action>" in result
    message["tool_calls"][0]["function"]["name"] = "unrelated"
    with pytest.raises(ValueError, match="disagree"):
        langfuse._render(message)


def test_resumed_task_roots_require_same_execution():
    roots = [
        {
            "id": str(i),
            "type": "AGENT",
            "input": "task",
            "metadata": {"execution_id": "same"},
            "startTime": str(i),
            "endTime": str(i + 1),
        }
        for i in range(2)
    ]
    roots.append({"id": "synthetic", "type": "SPAN", "endTime": None})
    assert langfuse._task_root(roots)["id"] == "1"
    roots[0]["metadata"]["execution_id"] = "different"
    with pytest.raises(ValueError, match="ambiguous"):
        langfuse._task_root(roots)
