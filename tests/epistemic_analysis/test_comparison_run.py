"""Checks for annotation transport and trace reconstruction."""

import json
from types import SimpleNamespace

import pytest

from corral.epistemic_analysis.sources import langfuse


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
