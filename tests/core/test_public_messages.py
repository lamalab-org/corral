"""Keep public text and tool protocol intact while excluding empty/private turns."""

from copy import deepcopy

import pytest

from corral.agents.base_agent import provider_messages
from corral.core.messages import EMPTY_RESPONSE_REMINDER
from corral.epistemic_analysis.decisions import (
    SUBMISSION_REVIEW_MESSAGE_NAME,
    scientific_messages,
)


@pytest.mark.parametrize("project", [provider_messages, scientific_messages])
def test_public_history_preserves_tool_only_and_multimodal_turns(project):
    messages = [
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": " \n"},
        {"role": "assistant", "content": [], "reasoning_content": "private"},
        {
            "role": "assistant",
            "content": "legacy private summary",
            "metadata": {"content_source": "reasoning_summary"},
        },
        {"role": "assistant", "name": "thinking", "content": "private"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "measure-1", "function": {"name": "measure", "arguments": "{}"}}
            ],
            "thinking_blocks": [
                {"type": "thinking", "thinking": "", "signature": "private"}
            ],
        },
        {"role": "tool", "content": "", "tool_call_id": "measure-1"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "redacted_thinking", "data": "private"},
                {"type": "reasoning", "summary": "private"},
                {"type": "text", "text": " "},
                {"type": "text", "text": "Public explanation"},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,test"},
                }
            ],
        },
    ]
    original = deepcopy(messages)
    public = project(messages)
    assert messages == original
    assert len(public) == 4
    assert public[0]["tool_calls"] == messages[5]["tool_calls"]
    assert "thinking_blocks" not in public[0]
    assert public[1] == messages[6]
    assert public[2]["content"] == [{"type": "text", "text": "Public explanation"}]
    assert public[3] == messages[8]


def test_retry_reminder_is_actor_context_but_not_scientific_evidence():
    reminder = {"role": "user", "name": EMPTY_RESPONSE_REMINDER, "content": "Continue."}
    assert provider_messages([reminder]) == [reminder]
    assert scientific_messages([reminder]) == []


def test_submission_feedback_is_context_but_proposed_answer_is_scientific_activity():
    proposal = {
        "role": "assistant",
        "name": "submission_proposal",
        "content": "Proposed answer: X",
    }
    notice = {
        "role": "user",
        "name": SUBMISSION_REVIEW_MESSAGE_NAME,
        "content": "Submission held; validate the proposed answer.",
    }
    assert provider_messages([proposal, notice]) == [proposal, notice]
    assert scientific_messages([proposal, notice]) == [proposal]
