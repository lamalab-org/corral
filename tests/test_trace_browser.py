"""Report browser checks independent of agent SDKs and benchmark execution."""

import csv
import gzip
import importlib.util
import io
import json
import lzma
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "corral_trace_browser",
    Path(__file__).resolve().parents[1] / "reports/serve_traces.py",
)
browser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(browser)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 16, 63, 65536])
def test_stream_handles_escapes_and_structures_across_chunks(chunk_size):
    value = {
        "brackets in strings": '}\\"[🧪é\\\\',
        "nested": [{"text": '\\"'}, None, False, True, -12.5, [], {}],
        "trailing": "\\",
    }
    raw = json.dumps({"skip": value, "keep": value}, ensure_ascii=False)
    stream = browser.JSONStream(io.StringIO(raw), chunk_size)
    found = {}
    for key in stream.object_keys():
        if key == "skip":
            stream.value(capture=False)
        else:
            found[key] = stream.value()
    assert found == {"keep": value}


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [
        (".json.gz", gzip.compress),
        (".json.xz", lzma.compress),
        (".json", lambda value: value),
    ],
)
def test_reads_only_requested_trial_and_preserves_saved_fields(
    tmp_path, suffix, compress
):
    trial = {
        "trial_id": "run:task:1",
        "messages": [{"role": "assistant", "content": 'x\\"🧪'}],
        "custom": {"value": 42},
    }
    report = {
        "metadata": {"nested": ["ignore"]},
        "task_results": {
            "before": {"trials": [{"messages": []}]},
            "task": {"metric": 0.5, "trials": [{"trial_id": "run:task:0"}, trial]},
            "after": {},
        },
    }
    path = tmp_path / f"report{suffix}"
    path.write_bytes(compress(json.dumps(report).encode()))
    assert browser.read_trial(path, "task", 1) == trial
    with pytest.raises(LookupError):
        browser.read_trial(path, "missing", 0)
    with pytest.raises(LookupError):
        browser.read_trial(path, "task", 8)


def write_index(root, report_file="level_1/react__model/report.json.gz", agent="react"):
    env = root / "environment"
    env.mkdir()
    row = {
        "environment": "environment",
        "level": "1",
        "agent": agent,
        "model": "model",
        "run_id": "run",
        "task_id": "task",
        "trial": "0",
        "execution_id": "run:task:0",
        "score": "",
        "success": "",
        "error": "",
        "duration_s": "1.25",
        "report_file": report_file,
    }
    with (env / "trials.csv").open("w", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    return env / report_file


def test_index_preserves_unscored_outcomes_and_reports_lfs(tmp_path):
    path = write_index(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(browser.LFS_POINTER + b"\noid sha256:test\nsize 10\n")
    traces, _ = browser.load_index(tmp_path)
    assert traces[0]["success"] is None
    assert traces[0]["score"] is None
    assert traces[0]["duration"] == 1.25
    assert traces[0]["availability"] == "lfs"
    with pytest.raises(FileNotFoundError, match="git lfs pull"):
        browser.read_trial(path, "task", 0)


def test_index_rejects_paths_outside_reports(tmp_path):
    write_index(tmp_path, "../../outside.json")
    with pytest.raises(ValueError, match="escapes reports directory"):
        browser.load_index(tmp_path)


def test_cache_invalidates_changed_reports_and_detects_stale_index(tmp_path):
    path = write_index(tmp_path)
    path.parent.mkdir(parents=True)

    def save(trial_id, content):
        path.write_bytes(
            gzip.compress(
                json.dumps(
                    {
                        "task_results": {
                            "task": {
                                "trials": [
                                    {
                                        "trial_id": trial_id,
                                        "messages": [
                                            {"role": "assistant", "content": content}
                                        ],
                                    }
                                ]
                            }
                        }
                    }
                ).encode()
            )
        )

    save("run:task:0", "First")
    store = browser.TraceStore(tmp_path)
    trace_id = store.traces[0]["id"]
    assert (
        json.loads(gzip.decompress(store.trial(trace_id)))["messages"][0]["content"]
        == "First"
    )
    save("run:task:0", "Changed content")
    assert (
        json.loads(gzip.decompress(store.trial(trace_id)))["messages"][0]["content"]
        == "Changed content"
    )
    save("another:task:0", "Wrong trial")
    with pytest.raises(ValueError, match="index and report disagree"):
        store.trial(trace_id)


def test_stream_rejects_incomplete_json():
    with pytest.raises(ValueError, match="Incomplete"):
        browser.JSONStream(io.StringIO('{"value":"unfinished\\'), 3).value()


def legacy_openhands_messages(terminal=(), session="first"):
    """Runtime calls were saved before the adapter flushed its buffered prose."""
    messages = []
    for name in ("measure", "submit_answer"):
        call_id = f"{session}-{name}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {"name": name, "arguments": '{"value":42}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": "42",
                },
            ]
        )
    messages.extend(
        [
            {"role": "system", "content": "You are OpenHands."},
            {"role": "user", "content": "Solve the task."},
            {"role": "assistant", "name": "thinking", "content": "Inspect inputs."},
            {"role": "assistant", "content": "A buffered progress message."},
            {"role": "assistant", "name": "thinking", "content": "Submit the answer."},
        ]
    )
    return messages + list(terminal)


@pytest.mark.parametrize(
    "terminal",
    [
        [],
        [{"role": "assistant", "content": "Done."}],
        [{"role": "assistant", "name": "openhands-error", "content": "Failed."}],
        [
            {"role": "assistant", "content": "Done."},
            {"role": "assistant", "name": "openhands-error", "content": "Failed."},
        ],
    ],
)
def test_legacy_openhands_presentation_preserves_calls_and_marks_untimed_prose(
    terminal,
):
    messages = legacy_openhands_messages(terminal)
    saved = json.loads(json.dumps(messages))

    presentation = browser.message_presentation(messages, "openhands")

    assert presentation == {
        "order": [4, 5, 6, 7, 8, 0, 1, 2, 3, *range(9, len(messages))],
        "untimed": [6, 7, 8],
    }
    assert sorted(presentation["order"]) == list(range(len(messages)))
    assert messages == saved
    displayed = [messages[index] for index in presentation["order"]]
    assert [message["role"] for message in displayed[:2]] == ["system", "user"]
    # Neither call IDs nor the order of the authoritative runtime stream changes.
    assert displayed[5:9] == saved[:4]
    for call, result in zip(displayed[5:9:2], displayed[6:9:2], strict=True):
        assert call["tool_calls"][0]["id"] == result["tool_call_id"]
    assert displayed[9:] == terminal


@pytest.mark.parametrize("agent", ["react", "codex", "claude-code", "", None])
def test_presentation_leaves_other_agents_unchanged(agent):
    messages = legacy_openhands_messages()
    assert browser.message_presentation(messages, agent) is None


def test_presentation_leaves_chronological_openhands_unchanged():
    legacy = legacy_openhands_messages([{"role": "assistant", "content": "Done."}])
    messages = [legacy[index] for index in [4, 5, 6, 0, 1, 7, 8, 2, 3, 9]]
    assert browser.message_presentation(messages, "openhands") is None


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "An incomplete trace."}],
        legacy_openhands_messages()[:4],
        legacy_openhands_messages()[:5],
    ],
)
def test_presentation_does_not_invent_missing_openhands_prologues(messages):
    assert browser.message_presentation(messages, "openhands") is None


@pytest.mark.parametrize("late_result", [False, True])
@pytest.mark.parametrize("distinct_runs", [False, True])
def test_legacy_openhands_presentation_keeps_repeated_sessions_separate(
    late_result, distinct_runs
):
    first = legacy_openhands_messages([{"role": "assistant", "content": "Done."}])
    second = legacy_openhands_messages(
        [{"role": "assistant", "content": "Done again."}], session="second"
    )
    boundary = (
        [{"role": "tool", "tool_call_id": "late", "content": "Delayed result."}]
        if late_result
        else []
    )
    if distinct_runs:
        for message in first + boundary:
            message["agent_run_id"] = "first"
        for message in second:
            message["agent_run_id"] = "second"
    messages = first + boundary + second
    start = len(first) + len(boundary)
    local_order = [4, 5, 6, 7, 8, 0, 1, 2, 3, 9]

    presentation = browser.message_presentation(messages, "openhands")

    assert presentation == {
        "order": local_order
        + list(range(len(first), start))
        + [start + index for index in local_order],
        "untimed": [6, 7, 8, start + 6, start + 7, start + 8],
    }
    assert sorted(presentation["order"]) == list(range(len(messages)))


def test_presentation_does_not_move_another_runs_prompt_ahead_of_runtime_calls():
    messages = legacy_openhands_messages()
    for index, message in enumerate(messages):
        message["agent_run_id"] = "first" if index < 4 else "second"
    assert browser.message_presentation(messages, "openhands") is None


@pytest.mark.parametrize(
    ("agent", "chronological", "normalized"),
    [("openhands", False, True), ("openhands", True, False), ("react", False, False)],
)
def test_trace_response_adds_presentation_without_rewriting_saved_messages(
    tmp_path, agent, chronological, normalized
):
    messages = legacy_openhands_messages([{"role": "assistant", "content": "Done."}])
    if chronological:
        messages = [messages[index] for index in [4, 5, 6, 0, 1, 7, 8, 2, 3, 9]]
    trial = {
        "trial_id": "run:task:0",
        "messages": messages,
        "custom": {"preserved": True},
    }
    path = write_index(tmp_path, agent=agent)
    path.parent.mkdir(parents=True)
    path.write_bytes(
        gzip.compress(
            json.dumps({"task_results": {"task": {"trials": [trial]}}}).encode()
        )
    )
    original_report = path.read_bytes()
    store = browser.TraceStore(tmp_path)
    trace_id = store.traces[0]["id"]

    response = json.loads(gzip.decompress(store.trial(trace_id)))

    if normalized:
        assert response.pop("_trace_browser") == {
            "order": [4, 5, 6, 7, 8, 0, 1, 2, 3, 9],
            "untimed": [6, 7, 8],
        }
    else:
        assert "_trace_browser" not in response
    assert response == trial
    assert path.read_bytes() == original_report
