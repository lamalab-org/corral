"""Distinguish ReAct's actual multi-action messages from dispatch bookkeeping."""

import copy
import json

import pytest

from corral.epistemic_analysis.annotation_transcript import (
    SAVED_FIELDS,
    annotation_transcript,
)


def batch_document():
    actions = [
        ("read_file", {"path": "data.csv"}),
        ("read_file", {"path": "holdout.csv"}),
    ]
    prose = "Inspect the supplied files.\n" + "\n".join(
        f"<action>{name}</action><action_input>{json.dumps(args)}</action_input>"
        for name, args in actions
    )
    messages, sources = [], []
    for i, (name, args) in enumerate(actions):
        call = {
            "id": f"call{i}",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        fields = {"agent_run_id": "run", "tool_calls": [call]}
        messages.append(
            {
                "role": "assistant",
                "content": (prose if i == 0 else "null")
                + SAVED_FIELDS
                + json.dumps(fields),
            }
        )
        sources.append({"index": len(sources), "agent_run_id": "run"})
        messages.append({"role": "tool", "content": f"Result {i}: αβ"})
        sources.append(
            {"index": len(sources), "agent_run_id": "run", "tool_call_id": f"call{i}"}
        )
    return {
        "messages": messages,
        "message_sources": sources,
        "completion_status": "failed",
    }


def prepare(document, agent="react"):
    return annotation_transcript(
        document,
        agent,
        "The supplied holdout.csv is public.",
        [{"path": "task.json", "sha256": "abc"}],
    )


def test_react_batch_keeps_real_prose_calls_results_and_source_mapping():
    document = batch_document()
    original = copy.deepcopy(document)
    result = prepare(document)
    assert document == original
    assert [m["role"] for m in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert (
        result["messages"][1]["content"].split(SAVED_FIELDS)[0]
        == document["messages"][0]["content"].split(SAVED_FIELDS)[0]
    )
    fields = json.loads(result["messages"][1]["content"].split(SAVED_FIELDS)[1])
    assert [c["id"] for c in fields["tool_calls"]] == ["call0", "call1"]
    assert result["messages"][2:] == [document["messages"][1], document["messages"][3]]
    assert result["message_sources"][1]["merged_dispatch_indices"] == [2]
    assert result["annotation_preprocessing"]["original_to_annotation_indices"] == {
        "0": 1,
        "1": 2,
        "2": 1,
        "3": 3,
    }
    assert result["completion_status"] == "failed"
    with pytest.raises(ValueError, match="original transcript"):
        prepare(result)


def test_complete_initial_context_preserves_dispatch_mapping_and_results():
    document = batch_document()
    prefix = [
        {"role": "system", "content": "Scientific assistant"},
        {"role": "user", "content": "Task instructions and ReAct format"},
        {"role": "user", "content": "Available tool schemas: read_file, submit_answer"},
    ]
    sources = [
        {"origin": "reconstructed_react_initial_instructions", "index": None}
    ] * 3
    result = annotation_transcript(
        document,
        "react",
        "Task instructions",
        [],
        initial_messages=prefix,
        initial_sources=sources,
    )
    assert result["messages"][:3] == prefix
    assert result["messages"][4:] == [document["messages"][1], document["messages"][3]]
    assert result["annotation_preprocessing"]["original_to_annotation_indices"] == {
        "0": 3,
        "1": 4,
        "2": 3,
        "3": 5,
    }
    assert result["annotation_preprocessing"]["task_prompt_index"] == 1
    assert result["message_sources"][3]["merged_dispatch_indices"] == [2]
    with pytest.raises(ValueError, match="one source"):
        annotation_transcript(
            document, "react", "Task instructions", [], initial_messages=prefix
        )
    with pytest.raises(ValueError, match="rendered task"):
        annotation_transcript(
            document,
            "react",
            "A different task",
            [],
            initial_messages=prefix,
            initial_sources=sources,
        )


@pytest.mark.parametrize(
    "change",
    [
        "real_prose",
        "different_arguments",
        "actor",
        "duplicate_id",
        "error",
        "user_feedback",
        "missing_result",
        "malformed_actions",
    ],
)
def test_ambiguous_or_independent_messages_are_preserved(change):
    document = batch_document()
    message = document["messages"][2]
    fields = json.loads(message["content"].split(SAVED_FIELDS)[1])
    if change == "real_prose":
        message["content"] = message["content"].replace("null", "I now reconsider.", 1)
    elif change == "different_arguments":
        fields["tool_calls"][0]["function"]["arguments"] = '{"path":"secret.csv"}'
    elif change == "actor":
        document["message_sources"][2]["agent_run_id"] = fields["agent_run_id"] = (
            "other"
        )
    elif change == "duplicate_id":
        fields["tool_calls"][0]["id"] = "call0"
    elif change == "error":
        fields["error"] = "Provider failed"
    elif change == "user_feedback":
        document["messages"].insert(
            2, {"role": "user", "content": "Stop and reconsider."}
        )
        document["message_sources"].insert(2, {"index": 99})
    elif change == "missing_result":
        del document["messages"][1]
        del document["message_sources"][1]
    elif change == "malformed_actions":
        document["messages"][0]["content"] = document["messages"][0]["content"].replace(
            "<action_input>{", "<action_input>oops{", 1
        )
    if change in {"different_arguments", "actor", "duplicate_id", "error"}:
        message["content"] = "null" + SAVED_FIELDS + json.dumps(fields)
    result = prepare(document)
    assert result["messages"][1:] == document["messages"]
    assert not result["annotation_preprocessing"]["folded_react_dispatches"]


def test_other_agents_and_unexecuted_proposals_are_unchanged():
    document = batch_document()
    assert prepare(document, "codex")["messages"][1:] == document["messages"]
    document["messages"] = document["messages"][:2]
    document["message_sources"] = document["message_sources"][:2]
    assert prepare(document)["messages"][1:] == document["messages"]


def test_unfolded_fallback_retains_every_original_dispatch_and_result():
    document = batch_document()
    prepared = annotation_transcript(
        document, "react", "Task instructions", [], fold_react_dispatches=False
    )
    assert prepared["messages"][1:] == document["messages"]
    assert not prepared["annotation_preprocessing"]["folded_react_dispatches"]
    assert prepared["annotation_preprocessing"]["original_to_annotation_indices"] == {
        str(i): i + 1 for i in range(len(document["messages"]))
    }


def test_repeated_identical_explicit_calls_remain_two_executed_calls():
    document = batch_document()
    for message in document["messages"]:
        message["content"] = message["content"].replace("holdout.csv", "data.csv")
    result = prepare(document)
    fields = json.loads(result["messages"][1]["content"].split(SAVED_FIELDS)[1])
    assert [c["id"] for c in fields["tool_calls"]] == ["call0", "call1"]
    assert len([m for m in result["messages"] if m["role"] == "tool"]) == 2


def test_submission_batch_preserves_dispatch_chronology_and_prior_results():
    from corral.epistemic_analysis.context import submission_end
    from corral.epistemic_analysis.sources.inputs import CheckInput

    document = batch_document()
    # The model proposes read_file followed by submit_answer in one message;
    # the runtime records each actual dispatch after the preceding tool result.
    document["messages"][0]["content"] = document["messages"][0]["content"].replace(
        '<action>read_file</action><action_input>{"path": "holdout.csv"}',
        '<action>submit_answer</action><action_input>{"path": "holdout.csv"}',
    )
    fields = json.loads(document["messages"][2]["content"].split(SAVED_FIELDS)[1])
    fields["tool_calls"][0]["function"]["name"] = "submit_answer"
    document["messages"][2]["content"] = "null" + SAVED_FIELDS + json.dumps(fields)
    prepared = prepare(document)
    assert prepared["messages"][1:] == document["messages"]
    assert not prepared["annotation_preprocessing"]["folded_react_dispatches"]
    normalized = CheckInput.from_document({"messages": document["messages"]}).snapshot()
    prepared.update(source=normalized["source"])
    trace = CheckInput.from_document(prepared).trace
    end = submission_end(trace)
    assert end == 4
    assert trace.messages[end - 2]["content"] == "Result 0: αβ"
    assert trace.messages[end]["content"] == "Result 1: αβ"
