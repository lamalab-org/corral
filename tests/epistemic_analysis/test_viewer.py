"""The comparison viewer serves saved annotations without changing their scope."""

import csv
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from corral.epistemic_analysis.viewer.server import RunStore, handler_for


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def run(tmp_path):
    fields = ["langfuse_trace_id", "environment", "agent", "model", "task_id"]
    with (tmp_path / "sample.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            dict(
                zip(fields, ["abc123", "lab", "react", "model", "task-1"], strict=False)
            )
        )
    messages = [
        {"role": "user", "content": "Measure the sample."},
        {"role": "assistant", "content": "Let me start."},
        {"role": "assistant", "content": "<action>submit_answer</action>"},
        {"role": "assistant", "content": "Post-submission text."},
    ]
    write(
        tmp_path / "traces/abc123.json", {"messages": messages, "message_sources": []}
    )
    write(
        tmp_path / "results/decisions_direct/abc123.json",
        {
            "status": "complete",
            "annotation_view": {"messages": messages[:3], "mappings": []},
            "windows": [
                {
                    "start": 0,
                    "end": 2,
                    "report": {
                        "assessments": [{"name": "pattern", "status": "absent"}]
                    },
                },
                {
                    "start": 1,
                    "end": 3,
                    "report": {
                        "assessments": [{"name": "pattern", "status": "active"}]
                    },
                },
            ],
        },
    )
    write(
        tmp_path / "results/decisions_graph/abc123.json",
        {
            "status": "complete",
            "graph": {
                "messages": messages,
                "nodes": [],
                "edges": [],
                "annotation_view": {"messages": messages[:3], "mappings": []},
                "qc": {"uncovered_messages": [1]},
                "annotation_requests": [
                    {
                        "request": {
                            "input": json.dumps(
                                {
                                    "candidates": {
                                        "m1_s0": {
                                            "quote": "Let me start.",
                                            "kind": "prose",
                                        },
                                    }
                                }
                            )
                        },
                        "response": {
                            "answers": [
                                {
                                    "name": "m1_s0_T",
                                    "choice": "present",
                                    "confidence": 0.04,
                                    "probabilities": [
                                        {"value": "present", "probability": 0.52}
                                    ],
                                },
                                {
                                    "name": "m2_s0_F",
                                    "choice": "present",
                                    "confidence": 0.99,
                                },
                            ]
                        },
                    }
                ],
            },
        },
    )
    return tmp_path


def test_window_results_and_annotation_boundaries_are_preserved(run):
    store = RunStore(run)
    trace = store.trace("abc123")
    assert len(trace["messages"]) == 4
    assert len(trace["annotation_messages"]) == 3
    direct = trace["methods"]["decisions_direct"]
    assert [(w["start"], w["end"]) for w in direct["windows"]] == [(0, 2), (1, 3)]
    assert [w["assessments"][0]["status"] for w in direct["windows"]] == [
        "absent",
        "active",
    ]
    assert trace["methods"]["llm_graph"] == {"status": "pending"}
    assert "annotation_requests" not in trace["methods"]["decisions_graph"]


def test_saved_low_confidence_answers_are_separate_from_empty_graph(run):
    store = RunStore(run)
    answers = store.decisions("abc123", 1)
    assert len(answers["candidates"]) == 1
    assert answers["candidates"][0]["answers"] == [
        {
            "type": "T",
            "choice": "present",
            "confidence": 0.04,
            "probabilities": [{"value": "present", "probability": 0.52}],
        }
    ]
    assert store.trace("abc123")["methods"]["decisions_graph"]["nodes"] == []


def test_whole_trace_checks_survive_viewer_projection(run):
    path = run / "results/decisions_graph/abc123.json"
    result = json.loads(path.read_text())
    checks = [
        {
            "name": "false_reporting",
            "status": "present",
            "support": [{"msg_idx": 3, "quote": "Post-submission text."}],
        }
    ]
    result["graph"].update(trace_checks=checks, trace_checks_status="complete")
    write(path, result)
    data = RunStore(run).trace("abc123")["methods"]["decisions_graph"]
    assert data["trace_checks"] == checks
    assert data["trace_checks_status"] == "complete"
    assert data["view_length"] == 3


def test_newly_completed_results_are_visible_without_restarting(run):
    store = RunStore(run)
    before = store.trace("abc123")
    assert store.catalog()["traces"][0]["statuses"]["llm_graph"] == "pending"
    write(
        run / "results/llm_graph/abc123.json",
        {"status": "refused", "error": "Provider refused"},
    )
    store.catalog_time = 0
    assert store.catalog()["traces"][0]["statuses"]["llm_graph"] == "refused"
    after = store.trace("abc123")
    assert after["revision"] != before["revision"]
    assert after["methods"]["llm_graph"]["error"] == "Provider refused"


def test_only_selected_traces_and_known_downloads_are_accessible(run):
    store = RunStore(run)
    for identity in ("other", "../.env", "abc123/../../.env"):
        with pytest.raises(KeyError):
            store.trace(identity)
        with pytest.raises(KeyError):
            store.download_path(identity, "trace")
    with pytest.raises(KeyError):
        store.download_path("abc123", "../../.env")


def test_http_serves_app_json_and_rejects_arbitrary_paths(run):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(RunStore(run)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            assert b"One message. Three perspectives." in response.read()
            assert (
                "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            )
        with urlopen(base + "/api/trace/abc123") as response:
            assert json.load(response)["id"] == "abc123"
        with urlopen(base + "/api/download/abc123/trace") as response:
            assert "attachment" in response.headers["Content-Disposition"]
            assert len(json.load(response)["messages"]) == 4
        for path in ("/.env", "/api/download/abc123/.env", "/api/trace/%2e%2e%2f.env"):
            with pytest.raises(HTTPError) as exc:
                urlopen(base + path)
            assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
