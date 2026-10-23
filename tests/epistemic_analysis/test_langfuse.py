"""Session retrieval and loss-aware conversion, without network/model calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from corral.epistemic_analysis.sources import langfuse as source


def observation(
    identity,
    kind,
    start,
    end,
    *,
    inputs=None,
    output=None,
    parent="step",
    trace="trace",
):
    return {
        "id": identity,
        "type": kind,
        "traceId": trace,
        "sessionId": "session",
        "projectId": "project",
        "parentObservationId": parent,
        "startTime": f"2026-09-01T00:00:{start:02d}Z",
        "endTime": f"2026-09-01T00:00:{end:02d}Z" if end is not None else None,
        "input": json.dumps(inputs),
        "output": json.dumps(output),
    }


def action(identity, name="measure"):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": '{"x": 1}'},
    }


@pytest.fixture
def rows():
    prompt = [
        {"role": "system", "content": "Solve the task."},
        {"role": "user", "content": "Find X."},
    ]
    first = {
        "role": "assistant",
        "content": "I think X.",
        "tool_calls": [action("one")],
    }
    tool = {"role": "tool", "tool_call_id": "one", "content": "X measured; [REDACTED]"}
    final = {"role": "assistant", "content": "X fits."}
    return [
        observation(
            "root",
            "AGENT",
            0,
            9,
            inputs=prompt,
            output={"status": "submitted", "answer": "X"},
            parent=None,
        ),
        observation("g1", "GENERATION", 1, 2, inputs=prompt, output=first),
        observation("t1", "TOOL", 2, 3, output=tool),
        observation(
            "g2", "GENERATION", 3, 5, inputs=[*prompt, first, tool], output=final
        ),
    ]


def test_reconstructs_history_once_and_keeps_source_mapping(rows):
    document = source.reconstruct_trace(list(reversed(rows)), "session")
    messages = document["messages"]
    assert len(messages) == 6
    assert messages[2]["content"].count("<action>") == 1
    assert messages[3] == {
        "role": "user",
        "content": "Observation: X measured; [REDACTED]",
    }
    assert messages[-1]["content"] == "<final_answer>X</final_answer>"
    assert document["source"]["message_sources"][2] == {
        "observation_id": "g2",
        "field": "input",
        "index": 2,
    }
    assert document["source"]["message_sources"][-1]["field"] == "output.answer"


def test_recovers_missing_result_only_at_unique_matching_action(rows):
    inputs = json.loads(rows[-1]["input"])
    inputs.pop()  # The terminal result is still recorded in its TOOL observation.
    rows[-1]["input"] = json.dumps(inputs)
    # The previous model generation has no tool output, so history coverage is
    # intact; the separate observation supplies the missing evidence.
    document = source.reconstruct_trace(rows, "session")
    assert document["messages"][3]["content"].startswith("Observation: X measured")
    assert document["source"]["message_sources"][3]["recovered_tool_result"]


def test_continuation_calls_can_be_grouped_in_export_but_split_in_history(rows):
    first = json.loads(rows[1]["output"])
    first["tool_calls"].append(action("two", "inspect"))
    rows[1]["output"] = json.dumps(first)
    inputs = json.loads(rows[-1]["input"])
    inputs.append(
        {"role": "assistant", "content": None, "tool_calls": [action("two", "inspect")]}
    )
    inputs.append({"role": "tool", "tool_call_id": "two", "content": "Checked."})
    rows[-1]["input"] = json.dumps(inputs)
    rows.append(observation("t2", "TOOL", 3, 4, output=inputs[-1]))
    messages = source.reconstruct_trace(rows, "session")["messages"]
    assert sum(m["content"].count("<action>") for m in messages) == 2
    assert sum(m["content"].startswith("Observation:") for m in messages) == 2


def test_repeated_messages_are_preserved(rows):
    history = json.loads(rows[-1]["input"])
    history += [{"role": "assistant", "content": "Try again."}] * 2
    rows[-1]["input"] = json.dumps(history)
    document = source.reconstruct_trace(rows, "session")
    assert [m["content"] for m in document["messages"]].count("Try again.") == 2


def test_tool_result_arrays_remain_json(rows):
    result = [1, 2, {"type": "measurement", "value": 3}]
    tool = json.loads(rows[2]["output"])
    tool["content"] = result
    rows[2]["output"] = json.dumps(tool)
    history = json.loads(rows[-1]["input"])
    history[-1] = tool
    rows[-1]["input"] = json.dumps(history)
    document = source.reconstruct_trace(rows, "session")
    assert (
        json.loads(document["messages"][3]["content"].removeprefix("Observation: "))
        == result
    )


def test_appends_terminal_tool_result_without_another_generation(rows):
    rows = rows[:-1]
    rows[0]["output"] = json.dumps({"error": "Execution failed"})
    messages = source.reconstruct_trace(rows, "session")["messages"]
    assert len(messages) == 4
    assert messages[-1]["content"].startswith("Observation:")
    assert not any("<final_answer>" in m["content"] for m in messages)


def test_existing_react_tags_and_final_answer_are_not_duplicated(rows):
    first = json.loads(rows[1]["output"])
    first["content"] = (
        'I think X. <action>measure</action><action_input>{"x": 1}</action_input>'
    )
    rows[1]["output"] = json.dumps(first)
    history = json.loads(rows[-1]["input"])
    history[2] = first
    rows[-1]["input"] = json.dumps(history)
    rows[-1]["output"] = json.dumps(
        {"role": "assistant", "content": "<final_answer>X</final_answer>"}
    )
    messages = source.reconstruct_trace(rows, "session")["messages"]
    assert sum(m["content"].count("<action>") for m in messages) == 1
    assert sum(m["content"].count("<final_answer>") for m in messages) == 1


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("open_root", "completed task root"),
        ("open_generation", "still open"),
        ("branch", "branched or truncated"),
        ("missing_tool", "earlier tool result"),
    ],
)
def test_rejects_incomplete_or_diverging_data(rows, change, match):
    if change == "open_root":
        rows[0]["endTime"] = None
    elif change == "open_generation":
        rows[1]["endTime"] = None
    elif change == "branch":
        rows[1]["output"] = json.dumps(
            {"role": "assistant", "content": "A different branch"}
        )
    else:
        rows.insert(
            2,
            observation(
                "lost",
                "TOOL",
                1,
                2,
                output={"role": "tool", "tool_call_id": "missing", "content": "Lost"},
            ),
        )
    with pytest.raises(ValueError, match=match):
        source.reconstruct_trace(rows, "session")


def test_render_preserves_recorded_reasoning_and_rejects_images():
    message = {
        "role": "assistant",
        "reasoning_content": "I think X.",
        "content": [{"type": "text", "text": "Measure X."}],
        "tool_calls": [action("one")],
    }
    assert source._render(message)["content"].startswith(
        "I think X.\nMeasure X.\n<action>"
    )
    message["content"] = [{"type": "image_url", "image_url": "example"}]
    with pytest.raises(ValueError, match="Non-text"):
        source._render(message)


class Pages:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.requests = []
        self.observations = self

    def get_many(self, **kwargs):
        self.requests.append(kwargs)
        page = next(self.pages)
        return SimpleNamespace(model_dump=lambda **_: page)


def test_fetch_paginates_deduplicates_and_bounds_session_filter(rows):
    api = Pages(
        [
            {"data": rows[:2], "meta": {"cursor": "next"}},
            {"data": rows[1:], "meta": {"cursor": None}},
        ]
    )
    result = source.fetch_observations(
        api, "session", "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z"
    )
    assert len(result) == 4
    assert api.requests[1]["cursor"] == "next"
    filters = json.loads(api.requests[0]["filter"])
    assert [item["column"] for item in filters] == [
        "sessionId",
        "startTime",
        "startTime",
    ]
    assert "io" in api.requests[0]["fields"]
    assert "parse_io_as_json" not in api.requests[0]


def test_fetch_rejects_repeated_cursor_and_wrong_session(rows):
    page = {"data": rows, "meta": {"cursor": "same"}}
    with pytest.raises(RuntimeError, match="pagination"):
        source.fetch_observations(Pages([page, page]), "session")
    with pytest.raises(ValueError, match="another session"):
        source.fetch_observations(Pages([page]), "wrong")


def test_actual_sdk_request_contract(rows):
    httpx = pytest.importorskip("httpx")
    langfuse = pytest.importorskip("langfuse.api.client")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"data": rows, "meta": {"cursor": None}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        api = langfuse.LangfuseAPI(
            base_url="https://langfuse.example",
            username="public",
            password="secret",
            httpx_client=http,
        )
        result = source.fetch_observations(api, "session")
    assert requests[0].url.path == "/api/public/v2/observations"
    assert len(result) == 4
    assert result[0]["sessionId"] == "session"
    assert (
        source.reconstruct_trace(result, "session")["messages"][-1]["content"]
        == "<final_answer>X</final_answer>"
    )
