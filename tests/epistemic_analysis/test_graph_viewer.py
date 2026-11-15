"""Native graph viewing keeps completed, trimmed and unfinished work distinct."""

import hashlib
import json
import re
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

pytest.importorskip(
    "corral.epistemic_analysis.viewer",
    reason="Local annotation viewer is not installed",
)

from corral.epistemic_analysis.viewer.graph_run import (
    PATTERN_VERSION,
    GraphRunStore,
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def graph_run(tmp_path):
    source = tmp_path / "input.json"
    source_hash = write(
        source,
        {
            "trace": {
                "messages": [
                    {"role": "assistant", "content": "Original middle and tail."}
                ]
            }
        },
    )
    job = {
        "id": "one",
        "gpu": 0,
        "input": str(source),
        "input_file_sha256": source_hash,
        "input_fingerprint": "input",
        "origin": {"environment": "spectra", "agent": "react", "model": "agent-model"},
    }
    write(
        tmp_path / "queue-request.json",
        {"profile_fingerprint": "profile", "selection": [job, {**job, "id": "two"}]},
    )
    graph = {
        "messages": [{"role": "assistant", "content": "Original middle and tail."}],
        "message_sources": [{"parent_msg_idx": 8, "context_only": False}],
        "nodes": [
            {
                "node_id": "N1",
                "type": "H",
                "time": 0,
                "support": [{"msg_idx": 0, "start": 20, "end": 24, "quote": "tail"}],
            }
        ],
        "edges": [],
        "provenance": {
            "model": "KnowLine",
            "window": 32,
            "overlap": 16,
            "min_confidence": 0.6,
        },
        "qc": {"warnings": ["Reduced source coverage"]},
        "annotation_view": {
            "messages": [{"role": "assistant", "content": "Original [omitted] tail."}],
            "mappings": [{"msg_idx": 0, "omitted_chars": 11}],
        },
    }
    signature = write(tmp_path / "graphs/one.json", graph)
    write(
        tmp_path / "completed/one.json",
        {
            "id": "one",
            "state": "requires_review",
            "input_fingerprint": "input",
            "profile_fingerprint": "profile",
            "graph": "graphs/one.json",
            "graph_sha256": signature,
            "node_count": 1,
            "edge_count": 0,
        },
    )
    return tmp_path


def test_completed_review_graph_preserves_source_and_scope(graph_run):
    store = GraphRunStore(graph_run)
    catalog = store.catalog()
    assert catalog["total"] == 1
    assert catalog["total_jobs"] == 2
    assert catalog["traces"][0]["statuses"] == {"decisions_graph": "requires_review"}
    trace = store.trace("one")
    assert trace["messages"][0]["content"] == "Original middle and tail."
    assert trace["annotation_messages"][0]["content"] == "Original [omitted] tail."
    assert trace["message_sources"][0]["parent_msg_idx"] == 8
    assert trace["methods"]["decisions_graph"]["threshold"] == 0.6
    assert trace["methods"]["decisions_graph"]["trace_checks_status"] == "not_requested"
    assert (
        store.download_path("one", "decisions_graph") == graph_run / "graphs/one.json"
    )


def test_derived_scope_hides_later_labels_and_saved_decisions(graph_run):
    path = graph_run / "graphs/one.json"
    graph = json.loads(path.read_text())
    graph["messages"].append({"role": "tool", "content": "answer accepted"})
    graph["annotation_view"]["messages"].append(graph["messages"][-1])
    graph["provenance"]["graph_scope"] = {
        "message_count": 1,
        "application": "filtered_saved_labels",
    }
    digest = write(path, graph)
    path = graph_run / "completed/one.json"
    record = json.loads(path.read_text())
    write(path, {**record, "graph_sha256": digest})
    store = GraphRunStore(graph_run)
    result = store.trace("one")
    assert len(result["messages"]) == 2
    assert len(result["annotation_messages"]) == 1
    assert (
        result["methods"]["decisions_graph"]["graph_scope"]["application"]
        == "filtered_saved_labels"
    )
    assert store.decisions("one", 1) == {"candidates": [], "outside_graph_scope": True}


def test_patterns_are_revision_pinned_visible_and_downloadable(graph_run):
    store = GraphRunStore(graph_run)
    before = store.trace("one")["revision"]
    record = json.loads((graph_run / "completed/one.json").read_text())
    pattern = {
        "name": "fixed_hypothesis_test_tuning",
        "category": "reasoning",
        "description": "Graph shape",
        "count": 1,
        "status": "matched",
    }
    report = {
        "version": PATTERN_VERSION,
        "graph_sha256": record["graph_sha256"],
        "patterns": [pattern],
        "repeated_tests": [],
        "reviews": [],
    }
    path = graph_run / "pattern-analysis/one.json"
    write(path, report)
    trace = store.trace("one")
    assert trace["revision"] != before
    assert trace["methods"]["decisions_graph"]["pattern_analysis"]["patterns"] == [
        pattern
    ]
    assert trace["parent_patterns"]["patterns"][0]["status"] == "matched"
    assert store.catalog()["pattern_counts"] == {"fixed_hypothesis_test_tuning": 1}
    assert store.download_path("one", "patterns") == path
    write(path, {**report, "graph_sha256": "stale"})
    assert store.trace("one")["methods"]["decisions_graph"]["pattern_analysis"] is None
    assert store.catalog()["pattern_counts"] == {}
    with pytest.raises(KeyError):
        store.download_path("one", "patterns")


def test_newly_finished_jobs_appear_and_unfinished_jobs_stay_unavailable(graph_run):
    store = GraphRunStore(graph_run)
    graph = json.loads((graph_run / "graphs/one.json").read_text())
    signature = write(graph_run / "graphs/two.json", graph)
    assert store.catalog()["total"] == 1
    with pytest.raises(KeyError, match="not finished"):
        store.trace("two")
    record = json.loads((graph_run / "completed/one.json").read_text())
    write(
        graph_run / "completed/two.json",
        {
            **record,
            "id": "two",
            "graph": "graphs/two.json",
            "graph_sha256": signature,
            "state": "complete",
        },
    )
    assert store.catalog()["total"] == 2
    assert store.trace("two")["methods"]["decisions_graph"]["status"] == "complete"


def test_unknown_or_modified_artifacts_are_rejected(graph_run):
    store = GraphRunStore(graph_run)
    for identity in ("../one", "missing"):
        with pytest.raises(KeyError):
            store.trace(identity)
    with pytest.raises(KeyError):
        store.download_path("one", "../input")
    store.trace("one")
    (graph_run / "graphs/one.json").write_text("{}")
    with pytest.raises(ValueError, match="completion record"):
        store.trace("one")


def test_scientist_counts_once_and_requires_every_unit_to_finish(graph_run):
    request = json.loads((graph_run / "queue-request.json").read_text())
    for job, kind in zip(request["selection"], ("main", "node"), strict=True):
        job["parent_id"] = "parent"
        job["scientist_unit"] = {
            "kind": kind,
            "node_id": "main" if kind == "main" else "node_0001",
        }
    write(graph_run / "queue-request.json", request)
    store = GraphRunStore(graph_run)
    catalog = store.catalog()
    assert catalog["total_jobs"] == 2
    assert catalog["parent_trace_count"] == 1
    assert catalog["updated_parent_count"] == 0
    assert catalog["partial_parent_count"] == 1
    (parent,) = catalog["parent_traces"]
    assert parent["id"] == "parent"
    assert parent["unit_count"] == 2
    assert parent["updated_unit_count"] == 1
    assert parent["statuses"]["decisions_graph"] == "partial"
    assert [(u["unit_label"], u["available"]) for u in parent["units"]] == [
        ("Main run", True),
        ("Node 0001", False),
    ]
    graph = json.loads((graph_run / "graphs/one.json").read_text())
    signature = write(graph_run / "graphs/two.json", graph)
    record = json.loads((graph_run / "completed/one.json").read_text())
    write(
        graph_run / "completed/two.json",
        {
            **record,
            "id": "two",
            "graph": "graphs/two.json",
            "graph_sha256": signature,
        },
    )
    catalog = store.catalog()
    assert catalog["updated_parent_count"] == 1
    assert catalog["partial_parent_count"] == 0
    assert (
        catalog["parent_traces"][0]["statuses"]["decisions_graph"] == "requires_review"
    )


def test_native_saved_decisions_use_selected_probability(graph_run):
    graph = json.loads((graph_run / "graphs/one.json").read_text())
    graph["annotation_requests"] = [
        {
            "exchange": "a" * 64,
            "state_ref": "b" * 64,
            "questions": ["m0_s0_H", "edge_N1_N2"],
        }
    ]
    signature = write(graph_run / "graphs/one.json", graph)
    record = json.loads((graph_run / "completed/one.json").read_text())
    write(graph_run / "completed/one.json", {**record, "graph_sha256": signature})
    write(
        graph_run / "artifacts/gpu-0/exchanges" / f"{'a' * 64}.json",
        {
            "response": {
                "answers": {
                    "m0_s0_H": {
                        "choice": "present",
                        "confidence": 0.1,
                        "probabilities": {"present": 0.7, "absent": 0.3},
                    }
                }
            }
        },
    )
    write(
        graph_run / "artifacts/gpu-0/states" / f"{'b' * 64}.json",
        {"candidates": {"m0_s0": {"msg_idx": 0, "quote": "Original"}}},
    )
    result = GraphRunStore(graph_run).decisions("one", 0)
    answer = result["candidates"][0]["answers"][0]
    assert answer["confidence"] == 0.7
    assert answer["native_confidence"] == 0.1


def test_graphs_share_review_port_with_separate_routes(graph_run):
    from corral.epistemic_analysis.viewer.benchmark import handler_for

    handler = handler_for(
        SimpleNamespace(catalog=lambda: {"app": "review"}),
        "preserved-review-token",
        graph_store=GraphRunStore(graph_run),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/catalog") as response:
            assert json.load(response) == {"app": "review"}
        with urlopen(base + "/") as response:
            html = response.read().decode()
            assert "preserved-review-token" in html
            assert 'href="/graphs/"' in html
        with urlopen(base + "/graphs?trace=one") as response:
            assert response.url.endswith("/graphs/?trace=one")
            html = response.read().decode()
            assert 'name="app-base" content="/graphs"' in html
            assets = re.findall(
                r'(?:src|href)="(/graphs/[^" ]+\?v=[a-f0-9]{12})"', html
            )
            assert len(assets) == 3
            assert "__VIEWER_VERSION__" not in html
            assert 'id="startup-message"' in html
        for asset in assets:
            with urlopen(base + asset) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
        with urlopen(base + "/graphs/api/catalog") as response:
            assert json.load(response)["mode"] == "graph_run"
        with urlopen(base + "/graphs/connection-check") as response:
            html = response.read().decode()
            assert "Corral graph server reached" in html
            assert "<script" not in html
            assert "<link" not in html
            assert response.headers["Cache-Control"] == "no-store"
        with urlopen(base + "/graphs/api/download/one/decisions_graph") as response:
            assert json.load(response)["nodes"][0]["node_id"] == "N1"
        with urlopen(base + "/graphs/app.js") as response:
            assert b"APP_BASE + path" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
