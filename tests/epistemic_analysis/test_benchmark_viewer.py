"""Ground-truth integrity, aggregation, and local review HTTP boundaries."""

import csv
import hashlib
import importlib.util
import io
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_viewer",
    Path(__file__).parents[2] / "src/corral/epistemic_analysis/viewer/benchmark.py",
)
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def install_code_assessments(root):
    manifest = viewer.read_json(root / "manifest.json")
    manifest.update(
        code_checks=["insufficient_validation"], code_policy_version="test-v1"
    )
    write(root / "manifest.json", manifest)
    check = {
        "name": "insufficient_validation",
        "status": "absent",
        "decision_source": "code",
        "model_called": False,
        "rationale": "Assigned by code: final candidate was validated.",
        "request_count": 0,
        "elapsed_s": 0,
    }
    traces = {}
    for row in manifest["selection"]:
        tid = row["id"]
        plan = viewer.read_json(root / "plans" / f"{tid}.json")
        traces[tid] = {
            "trace_sha256": viewer.digest(root / "traces" / f"{tid}.json"),
            "checks": [check],
            "units": [{"unit_id": u["id"], "checks": [check]} for u in plan["units"]],
        }
    write(
        root / "code-assessments.json", {"policy_version": "test-v1", "traces": traces}
    )


@pytest.fixture
def root(tmp_path):
    questions = {
        name: {
            "instructions": f"Frozen {name} instructions",
            "criteria": {
                "present": "Observed behavior",
                "absent": "No qualifying behavior",
                "uncertain": "Missing evidence",
                **(
                    {"not_applicable": "No validation tools"}
                    if name == "insufficient_validation"
                    else {}
                ),
            },
        }
        for name in ["false_reporting", "insufficient_validation"]
    }
    write(
        tmp_path / "manifest.json",
        {
            "protocol": {"questions": questions, "trace_check_version": "frozen-v1"},
            "models": {"fast": {}, "uncertain": {}},
            "selection": [
                {
                    "id": t,
                    "environment": "lab",
                    "task_id": t,
                    "reference_tokens": 100,
                    "score": "secret outcome",
                }
                for t in ("t1", "t2")
            ],
        },
    )
    for t in ("t1", "t2"):
        write(
            tmp_path / "traces" / f"{t}.json",
            {
                "messages": [
                    {"role": "user", "content": "Task [a] 🧪"},
                    {"role": "assistant", "content": "🧪 café\nResult [a]: complete."},
                ],
                "message_sources": [
                    {},
                    {"original_role": "assistant", "name": "experiment_node_0_step_0"},
                ],
                "completion_status": "complete",
                "reconstruction_status": "complete",
            },
        )
        units = [
            {
                "id": "u0",
                "kind": "controller",
                "indices": [0],
                "slices": [{"id": "slice-000"}],
            }
        ]
        if t == "t2":
            units.append(
                {
                    "id": "u1",
                    "kind": "node",
                    "node_id": "node_0",
                    "indices": [1],
                    "slices": [{"id": "slice-001"}],
                }
            )
        else:
            units[0]["indices"] = [0, 1]
        write(tmp_path / "plans" / f"{t}.json", {"units": units})
        for model in ("fast", "uncertain"):
            checks = [
                {
                    "name": name,
                    "status": "present" if model == "fast" else "uncertain",
                    "elapsed_s": 3,
                    "request_count": 2,
                    "trigger_slices": ["slice-000"],
                }
                for name in questions
            ]
            write(
                tmp_path / "runs" / model / t / "assessment.0.json",
                {
                    "checks": checks,
                    "units": [{"unit_id": u["id"], "checks": checks} for u in units],
                },
            )
    with (tmp_path / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["model", "trace_id", "check", "outcome", "elapsed_s", "repeat"],
        )
        writer.writeheader()
        for model in ("fast", "uncertain"):
            for t in ("t1", "t2"):
                for name in questions:
                    for elapsed in (1, 2):
                        writer.writerow(
                            {
                                "model": model,
                                "trace_id": t,
                                "check": name,
                                "outcome": "ok",
                                "elapsed_s": elapsed,
                                "repeat": 0,
                            }
                        )
    return tmp_path


@pytest.fixture
def store(root):
    return viewer.ReviewStore(root)


def payload(
    store, trace_id="t1", unit_id="u0", check="false_reporting", status="present"
):
    review = store.get_review(trace_id, unit_id, check)
    return {
        **review,
        "initial_status": status,
        "final_status": status,
        "reviewer": "Researcher",
        "rationale": "The original passage supports this judgment.",
        "evidence": [{"msg_idx": 1, "start": 2, "end": 6, "quote": "café"}],
    }


def approve(store, **kwargs):
    body = payload(store, **kwargs)
    initial = store.save_review({**body, "action": "initial"})
    return store.save_review(
        {**body, "version": initial["version"], "action": "approve"}
    )


def test_blind_catalog_and_trace_do_not_include_predictions_or_task_score(store):
    catalog = store.catalog()
    assert catalog["total"] == 6
    assert catalog["approved"] == catalog["final_decisions"] == 0
    assert "secret outcome" not in json.dumps(catalog)
    assert "models" not in catalog
    trace = store.trace("t1")
    assert "predictions" not in trace
    assert trace["messages"][1]["role"] == "assistant"
    assert store.get_review("t1", "u0", "false_reporting")["initial"] is None


def test_unicode_evidence_offsets_and_revision_history_survive_reopen(store, root):
    record = approve(store)
    assert record["version"] == 2
    assert store.gold()[0]["status"] == "present"
    reopened = viewer.ReviewStore(root)
    record = reopened.get_review("t1", "u0", "false_reporting")
    assert record["evidence"][0]["quote"] == "café"
    assert record["initial"]["evidence"] == record["evidence"]
    assert [h["kind"] for h in reopened.history("t1", "u0", "false_reporting")] == [
        "approve",
        "initial",
    ]


@pytest.mark.parametrize(
    "replacement",
    [
        {"quote": "fabricated"},
        {"start": 3},
        {"end": 99},
        {"msg_idx": -1},
        {"start": True},
    ],
)
def test_fabricated_or_invalid_evidence_is_rejected(store, replacement):
    body = payload(store)
    body["evidence"][0].update(replacement)
    with pytest.raises(ValueError):
        store.save_review({**body, "action": "initial"})
    assert not store.records()


def test_initial_is_required_and_immutable_and_present_needs_evidence(store):
    body = payload(store)
    with pytest.raises(ValueError, match="saved initial"):
        store.save_review({**body, "action": "approve"})
    initial = store.save_review({**body, "action": "initial"})
    body["version"] = initial["version"]
    with pytest.raises(ValueError, match="already recorded"):
        store.save_review({**body, "action": "initial"})
    with pytest.raises(ValueError, match="exact evidence"):
        store.save_review({**body, "evidence": [], "action": "approve"})
    with pytest.raises(ValueError, match="rationale"):
        store.save_review({**body, "rationale": "", "action": "approve"})


def test_conflicting_writes_do_not_overwrite_another_review(store):
    body = payload(store)
    store.save_review({**body, "action": "initial"})
    with pytest.raises(viewer.Conflict):
        store.save_review({**body, "rationale": "Stale edit", "action": "draft"})
    assert store.get_review("t1", "u0", "false_reporting")["rationale"] != "Stale edit"


def test_only_validation_check_allows_not_applicable(store):
    with pytest.raises(ValueError, match="not allowed"):
        store.save_review(
            {**payload(store, status="not_applicable"), "action": "initial"}
        )
    approve(store, check="insufficient_validation", status="not_applicable")
    assert store.gold()[0]["status"] == "not_applicable"


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("absent", "present", "present"),
        ("absent", "uncertain", "uncertain"),
        ("not_applicable", "absent", "absent"),
        ("not_applicable", "not_applicable", "not_applicable"),
    ],
)
def test_branched_gold_requires_all_units_and_uses_frozen_or_rule(
    store, first, second, expected
):
    approve(
        store,
        trace_id="t2",
        unit_id="u0",
        check="insufficient_validation",
        status=first,
    )
    assert store.gold() == []
    approve(
        store,
        trace_id="t2",
        unit_id="u1",
        check="insufficient_validation",
        status=second,
    )
    assert store.gold()[0]["status"] == expected


def test_reverting_approved_review_to_draft_removes_it_from_gold(store):
    record = approve(store)
    store.save_review({**record, "action": "draft"})
    assert store.gold() == []


def test_draft_retains_initial_selection_without_locking_or_scoring_it(store):
    saved = store.save_review({**payload(store, status="uncertain"), "action": "draft"})
    assert saved["initial"] is None
    assert saved["draft_initial_status"] == "uncertain"
    assert store.gold() == []


def test_prediction_reveal_requires_initial_and_is_audited(store):
    with pytest.raises(ValueError, match="initial judgment"):
        store.reveal("t1", "u0", "false_reporting")
    initial = store.save_review({**payload(store), "action": "initial"})
    revealed = store.reveal("t1", "u0", "false_reporting")
    assert revealed["review"]["version"] == initial["version"] + 1
    assert revealed["review"]["initial"] == initial["initial"]
    assert "model" not in revealed["predictions"][0]
    assert store.history("t1", "u0", "false_reporting")[0]["kind"] == "reveal"


def test_dashboard_uses_approved_labels_and_final_decision_latency(store):
    empty = store.dashboard()
    assert empty["gold_count"] == 0
    assert all(m["macro_f1"] is None for m in empty["models"])
    assert all(m["mean_s"] == 3 and m["timed_decisions"] == 4 for m in empty["models"])
    approve(store)
    assert store.get_review("t1", "u0", "false_reporting")["initial"]["comparison_seen"]
    dashboard = store.dashboard()
    fast, uncertain = dashboard["models"]
    assert fast["macro_f1"] == fast["accuracy"] == fast["coverage"] == 1
    assert uncertain["macro_f1"] == uncertain["accuracy"] == uncertain["coverage"] == 0
    assert uncertain["matrix"]["present"]["uncertain"] == 1
    assert fast["supported_checks"] == 1
    assert fast["checks"][1]["f1"] is None


def test_source_changes_cannot_reuse_existing_gold_database(store, root):
    approve(store)
    write(
        root / "traces/t1.json", {"messages": [{"role": "user", "content": "Changed"}]}
    )
    with pytest.raises(viewer.Conflict, match="Source trace changed"):
        store.raw_trace("t1")
    with pytest.raises(ValueError, match="different traces or rubric"):
        viewer.ReviewStore(root)


def test_exports_include_provenance_and_leave_unreviewed_csv_rows_blank(store, root):
    legacy = root / "human-review.csv"
    legacy.write_text("leave this untouched")
    approve(store)
    exported = store.export()
    assert exported["fingerprint"] == store.fingerprint
    assert exported["source_hashes"]["traces/t1.json"]
    assert len(exported["history"]) == 2
    rows = list(csv.DictReader(io.StringIO(store.export_csv())))
    assert len(rows) == 4
    assert rows[0]["human_status"] == "present"
    assert rows[0]["evidence_message_ids"] == "m1"
    assert all(not r["human_status"] for r in rows[1:])
    assert legacy.read_text() == "leave this untouched"


def test_search_is_literal_and_offsets_point_to_original_content(store):
    results = store.search("t2", "[A]", "u1")
    assert len(results["hits"]) == 1
    hit = results["hits"][0]
    original = store.raw_trace("t2")["messages"][hit["index"]]["content"]
    assert original[hit["start"] : hit["end"]] == "[a]"
    assert store.message("t1", 1, 2)["text"].startswith("café")
    with pytest.raises(KeyError):
        store.get_review("../../outside", "u0", "false_reporting")


def test_prompt_context_exposes_missing_system_and_preserves_source(store, root):
    before = (root / "traces/t1.json").read_bytes()
    context = store.prompt_context("t1")
    assert context["sections"][0]["status"] == "recorded"
    assert context["sections"][0]["content"] == "Task [a] 🧪"
    assert context["sections"][0]["message_index"] == 0
    assert "system prompt was not saved" in context["notes"][0]
    write(
        root / "prompt-context/t1.json",
        {
            "trace_sha256": viewer.digest(root / "traces/t1.json"),
            "sections": [
                {
                    "title": "System prompt",
                    "role": "system",
                    "status": "reconstructed",
                    "content": "Recovered instructions",
                    "source_evidence": [{"git_commit": "pinned"}],
                }
            ],
            "notes": ["Recovered from pinned source, not a recorded model request."],
        },
    )
    context = store.prompt_context("t1")
    assert context["sections"][0]["status"] == "reconstructed"
    assert context["notes"] == [
        "Recovered from pinned source, not a recorded model request."
    ]
    assert (root / "traces/t1.json").read_bytes() == before
    assert store.message("t1", 1, 2)["text"].startswith("café")
    with pytest.raises(KeyError):
        store.prompt_context("../../outside")


def test_prompt_context_rejects_sidecar_for_another_trace(store, root):
    write(root / "prompt-context/t1.json", {"trace_sha256": "wrong", "sections": []})
    with pytest.raises(viewer.Conflict, match="different source trace"):
        store.prompt_context("t1")


def test_prompt_context_labels_reconstructed_task_and_keeps_full_system(root):
    path = root / "traces/t1.json"
    trace = viewer.read_json(path)
    trace["messages"] = [
        {"role": "user", "content": "Recovered task"},
        {"role": "system", "content": "Full system " * 2000},
        {"role": "user", "content": "Recorded task"},
        {"role": "assistant", "content": "An action"},
        {"role": "user", "content": "Later observation"},
    ]
    trace["message_sources"] = [{"origin": "reconstructed_task_prompt"}, {}, {}, {}, {}]
    write(path, trace)
    context = viewer.ReviewStore(root).prompt_context("t1")
    assert [s["status"] for s in context["sections"]] == [
        "recorded",
        "reconstructed",
        "recorded",
    ]
    assert context["sections"][0]["content"] == "Full system " * 2000
    assert not context["notes"]


def test_http_requires_same_origin_and_token_for_writes(store):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), viewer.handler_for(store, "test-token")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            assert b'meta name="review-token" content="test-token"' in response.read()
        body = json.dumps({**payload(store), "action": "initial"}).encode()
        for headers in (
            {},
            {"X-Review-Token": "test-token", "Origin": "https://unrelated.example"},
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/api/review", data=body, headers=headers))
            assert error.value.code == 403
        with urlopen(
            Request(
                base + "/api/review",
                data=body,
                headers={"X-Review-Token": "test-token"},
            )
        ) as response:
            assert json.load(response)["stage"] == "initial"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/predictions")
        assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(
                Request(base + "/api/catalog", headers={"Host": "unrelated.example"})
            )
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture
def assistant_file(root):
    """A separate, evidence-linked assistant reference with a branched positive."""
    manifest = json.loads((root / "manifest.json").read_text())
    reference = {
        "reference_kind": "assistant reference, not human approval",
        "rubric_version": manifest["protocol"]["trace_check_version"],
        "rubric_sha256": hashlib.sha256(
            json.dumps(
                manifest["protocol"], sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest(),
        "traces": [],
    }
    for trace_id in ("t1", "t2"):
        plan = json.loads((root / "plans" / f"{trace_id}.json").read_text())
        units = []
        for unit in plan["units"]:
            positive = trace_id == "t2" and unit["id"] == "u1"
            units.append(
                {
                    "unit_id": unit["id"],
                    "checks": [
                        {
                            "name": "false_reporting",
                            "status": "present" if positive else "absent",
                            "rationale": "Inspect the exact original passage.",
                            "confidence": "high",
                            "support": (
                                [
                                    {
                                        "msg_idx": 1,
                                        "start": 2,
                                        "end": 6,
                                        "quote": "café",
                                    }
                                ]
                                if positive
                                else []
                            ),
                        },
                        {
                            "name": "insufficient_validation",
                            "status": "uncertain",
                            "rationale": "The full validator inventory was not saved.",
                            "confidence": "medium",
                            "support": [],
                        },
                    ],
                }
            )
        reference["traces"].append(
            {
                "trace_id": trace_id,
                "reviewer": "assistant/test-reviewer",
                "coverage_notes": "All original messages reviewed.",
                "units": units,
            }
        )
    path = root / "assistant-review-20261106" / "assistant-labels.json"
    write(path, reference)
    return path


@pytest.fixture
def annotation_store(root, assistant_file):
    # An aggregate positive can come from another branch, not this unit.
    path = root / "runs/fast/t2/assessment.0.json"
    result = json.loads(path.read_text())
    result["units"][0]["checks"][0]["status"] = "absent"
    write(path, result)
    return viewer.ReviewStore(root)


def annotation(bundle, source_id="fast", check="false_reporting"):
    row = next(row for row in bundle["checks"] if row["name"] == check)
    return next(a for a in row["annotations"] if a["source_id"] == source_id)


def annotation_payload(
    store,
    trace_id="t1",
    unit_id="u0",
    check="false_reporting",
    source_id="fast",
    verdict="correct",
    **extra,
):
    current = annotation(store.annotations(trace_id, unit_id), source_id, check)
    return {
        "trace_id": trace_id,
        "unit_id": unit_id,
        "check": check,
        "source_id": source_id,
        "version": current["verdict"]["version"],
        "source_fingerprint": current["source_fingerprint"],
        "verdict": verdict,
        "corrected_status": "",
        "notes": "Checked against the source trace.",
        "reviewer": "Researcher",
        **extra,
    }


def test_named_comparison_opens_without_initial_or_creating_gold(annotation_store):
    store = annotation_store
    bundle = store.annotations("t1", "u0")
    assert {source["id"] for source in bundle["sources"]} == {
        "fast",
        "uncertain",
        "assistant",
    }
    assert {row["name"] for row in bundle["checks"]} == set(store.questions)
    assert annotation(bundle, "fast")["status"] == "present"
    assert annotation(bundle, "uncertain")["status"] == "uncertain"
    assert annotation(bundle, "assistant")["status"] == "absent"
    assert bundle["assistant"]["available"] is True
    assert store.get_review("t1", "u0", "false_reporting")["initial"] is None
    assert store.records() == []
    assert store.gold() == []


def test_unit_labels_are_distinct_from_trace_aggregation_and_evidence_is_exact(
    annotation_store,
):
    store = annotation_store
    first = store.annotations("t2", "u0")
    for source in ("fast", "assistant"):
        assert annotation(first, source)["status"] == "absent"
        assert annotation(first, source)["trace_status"] == "present"
    second = annotation(store.annotations("t2", "u1"), "assistant")
    assert second["status"] == second["trace_status"] == "present"
    assert second["confidence"] == "high"
    assert second["rationale"]
    evidence = second["evidence"][0]
    text = store.raw_trace("t2")["messages"][evidence["msg_idx"]]["content"]
    assert text[evidence["start"] : evidence["end"]] == evidence["quote"] == "café"


def test_annotation_exposure_is_recorded_in_later_initial_judgment(annotation_store):
    store = annotation_store
    store.annotations("t1", "u0")
    initial = store.save_review({**payload(store), "action": "initial"})
    assert initial["initial"]["comparison_seen"] is True
    assert initial["initial"]["status"] == "present"


@pytest.mark.parametrize("verdict", ["correct", "incorrect", "unsure"])
def test_each_verdict_persists_independently_of_human_ground_truth(
    annotation_store, root, verdict
):
    store = annotation_store
    baseline_csv = store.export_csv()
    body = annotation_payload(store, verdict=verdict)
    if verdict == "incorrect":
        body["corrected_status"] = "absent"
    saved = store.save_annotation_review(body)
    assert saved["verdict"] == verdict
    assert saved["version"] == 1
    reopened = viewer.ReviewStore(root)
    current = annotation(reopened.annotations("t1", "u0"))["verdict"]
    assert current["verdict"] == verdict
    assert current["notes"] == body["notes"]
    assert current["corrected_status"] == body["corrected_status"]
    assert current["stale"] is False
    assert reopened.gold() == []
    assert reopened.records() == []
    assert reopened.export_csv() == baseline_csv


def test_assistant_verdict_does_not_overwrite_an_existing_human_label(annotation_store):
    store = annotation_store
    human = approve(store)
    body = annotation_payload(store, source_id="assistant", verdict="incorrect")
    store.save_annotation_review({**body, "corrected_status": "present"})
    assert store.get_review("t1", "u0", "false_reporting") == human
    assert store.gold()[0]["status"] == "present"
    assert annotation(store.annotations("t1", "u0"), "assistant")["status"] == "absent"


def test_verdict_conflicts_and_reset_preserve_history(annotation_store):
    store = annotation_store
    body = annotation_payload(store, verdict="incorrect", corrected_status="absent")
    saved = store.save_annotation_review(body)
    with pytest.raises(viewer.Conflict):
        store.save_annotation_review(
            {**body, "verdict": "unsure", "corrected_status": ""}
        )
    current = annotation(store.annotations("t1", "u0"))["verdict"]
    assert current["verdict"] == "incorrect"
    reset = store.save_annotation_review(
        {
            **body,
            "version": saved["version"],
            "verdict": "unreviewed",
            "corrected_status": "",
        }
    )
    assert reset["version"] == 2
    assert reset["verdict"] == "unreviewed"
    assert len(store.annotation_history("t1", "u0", "false_reporting", "fast")) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"reviewer": ""},
        {"verdict": "present"},
        {"verdict": "incorrect", "corrected_status": "not_applicable"},
        {"verdict": "correct", "corrected_status": "absent"},
        {"version": True},
        {"source_id": "../../outside"},
    ],
)
def test_invalid_annotation_verdicts_are_rejected_without_saving(
    annotation_store, changes
):
    store = annotation_store
    body = annotation_payload(store)
    with pytest.raises((ValueError, KeyError)):
        store.save_annotation_review({**body, **changes})
    assert annotation(store.annotations("t1", "u0"))["verdict"]["version"] == 0


def test_correction_can_use_na_only_for_validation(annotation_store):
    store = annotation_store
    body = annotation_payload(
        store,
        check="insufficient_validation",
        verdict="incorrect",
        corrected_status="not_applicable",
    )
    saved = store.save_annotation_review(body)
    assert saved["corrected_status"] == "not_applicable"


def test_changed_model_annotation_marks_previous_verdict_stale(annotation_store, root):
    store = annotation_store
    body = annotation_payload(store)
    saved = store.save_annotation_review(body)
    path = root / "runs/fast/t1/assessment.0.json"
    prediction = json.loads(path.read_text())
    prediction["units"][0]["checks"][0]["status"] = "absent"
    prediction["checks"][0]["status"] = "absent"
    write(path, prediction)
    changed_bundle = store.annotations("t1", "u0")
    changed = annotation(changed_bundle)
    assert changed["status"] == "absent"
    assert changed["source_fingerprint"] != body["source_fingerprint"]
    assert changed["verdict"]["stale"] is True
    assert changed_bundle["reviews_summary"]["reviewed"] == 0
    assert changed["verdict"]["source_annotation"]["status"] == "present"
    with pytest.raises(viewer.Conflict):
        store.save_annotation_review({**body, "version": saved["version"]})
    corrected = store.save_annotation_review(
        {
            **body,
            "version": saved["version"],
            "source_fingerprint": changed["source_fingerprint"],
            "verdict": "incorrect",
            "corrected_status": "present",
        }
    )
    assert corrected["version"] == 2
    assert annotation(store.annotations("t1", "u0"))["verdict"]["stale"] is False


def test_assistant_rationale_changes_also_invalidate_prior_verdict(
    annotation_store, assistant_file
):
    store = annotation_store
    body = annotation_payload(store, source_id="assistant")
    store.save_annotation_review(body)
    reference = json.loads(assistant_file.read_text())
    reference["traces"][0]["units"][0]["checks"][0]["rationale"] = (
        "A revised explanation needs another human review."
    )
    write(assistant_file, reference)
    bundle = store.annotations("t1", "u0")
    revised = annotation(bundle, "assistant")
    assert revised["status"] == "absent"
    assert revised["source_fingerprint"] != body["source_fingerprint"]
    assert revised["verdict"]["stale"] is True
    assert bundle["reviews_summary"]["reviewed"] == 0
    assert revised["verdict"]["source_annotation"]["rationale"] == (
        "Inspect the exact original passage."
    )
    assert store.gold() == []


def test_missing_assistant_is_unavailable_not_an_absent_label(store):
    bundle = store.annotations("t1", "u0")
    assert bundle["assistant"]["available"] is False
    assert bundle["assistant"]["reason"]
    assert annotation(bundle, "fast")["status"] == "present"
    assistant = annotation(bundle, "assistant")
    assert assistant["status"] not in viewer.STATUSES
    with pytest.raises(ValueError):
        store.save_annotation_review(annotation_payload(store, source_id="assistant"))


@pytest.mark.parametrize(
    "problem", ["malformed", "rubric", "quote", "unit", "incomplete", "source_hash"]
)
def test_invalid_assistant_reference_is_not_silently_loaded(
    root, assistant_file, problem
):
    if problem == "malformed":
        assistant_file.write_text("{not JSON")
    else:
        reference = json.loads(assistant_file.read_text())
        if problem == "rubric":
            reference["rubric_sha256"] = "0" * 64
        elif problem == "quote":
            reference["traces"][1]["units"][1]["checks"][0]["support"][0]["quote"] = (
                "wrong"
            )
        elif problem == "unit":
            reference["traces"][1]["units"][0]["checks"][0]["support"] = [
                {"msg_idx": 1, "start": 2, "end": 6, "quote": "café"}
            ]
        elif problem == "incomplete":
            reference["traces"].pop()
        else:
            reference["integrity"] = {
                "frozen_source_file_sha256": {str(root / "traces/t1.json"): "0" * 64}
            }
        write(assistant_file, reference)
    store = viewer.ReviewStore(root)
    bundle = store.annotations("t1", "u0")
    assert bundle["assistant"]["available"] is False
    assert bundle["assistant"]["reason"]
    assert annotation(bundle, "fast")["status"] == "present"
    assert annotation(bundle, "assistant")["status"] not in viewer.STATUSES
    assert store.gold() == []


def test_annotation_exports_keep_sources_and_verdicts_separate_from_gold(
    annotation_store, root
):
    store = annotation_store
    legacy = root / "human-review.csv"
    legacy.write_text("source template remains untouched\n")
    store.save_annotation_review(annotation_payload(store))
    store.save_annotation_review(
        annotation_payload(
            store,
            source_id="assistant",
            verdict="incorrect",
            corrected_status="present",
        )
    )
    exported = store.export()
    assert exported["gold"] == []
    assert exported["reviews"] == []
    assert len(exported["annotation_reviews"]) == 2
    assert {r["source_id"] for r in exported["annotation_reviews"]} == {
        "fast",
        "assistant",
    }
    assert exported["annotation_sources"]
    rows = list(csv.DictReader(io.StringIO(store.export_annotation_reviews_csv())))
    reviewed = [row for row in rows if row["verdict"] != "unreviewed"]
    assert {row["source_id"] for row in reviewed} == {"fast", "assistant"}
    assert all(row["source_fingerprint"] for row in reviewed)
    assert all(
        not row["human_status"]
        for row in csv.DictReader(io.StringIO(store.export_csv()))
    )
    assert legacy.read_text() == "source template remains untouched\n"


def test_http_annotation_review_routes_require_token_and_preserve_read_only_gets(
    annotation_store,
):
    store = annotation_store
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), viewer.handler_for(store, "test-token")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    headers = {"X-Review-Token": "test-token", "Content-Type": "application/json"}
    try:
        body = json.dumps({"trace_id": "t1", "unit_id": "u0"}).encode()
        for request in [
            Request(base + "/api/annotations", data=body),
            Request(
                base + "/api/annotations",
                data=body,
                headers={**headers, "Origin": "https://unrelated.example"},
            ),
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(request)
            assert error.value.code == 403
        with urlopen(
            Request(base + "/api/annotations", data=body, headers=headers)
        ) as response:
            bundle = json.load(response)
        assert annotation(bundle, "assistant")["status"] == "absent"
        review = annotation_payload(
            store, verdict="incorrect", corrected_status="absent"
        )
        with pytest.raises(HTTPError) as error:
            urlopen(
                Request(
                    base + "/api/annotation-review", data=json.dumps(review).encode()
                )
            )
        assert error.value.code == 403
        with urlopen(
            Request(
                base + "/api/annotation-review",
                data=json.dumps(review).encode(),
                headers=headers,
            )
        ) as response:
            assert json.load(response)["verdict"] == "incorrect"
        with pytest.raises(HTTPError) as error:
            urlopen(
                base
                + "/api/annotation-review?trace_id=t1&unit_id=u0&check=false_reporting&source_id=fast&verdict=correct"
            )
        assert error.value.code == 404
        with urlopen(
            base
            + "/api/annotation-history?trace_id=t1&unit_id=u0&check=false_reporting&source_id=fast"
        ) as response:
            assert len(json.load(response)) == 1
        with urlopen(base + "/api/annotation-reviews.csv") as response:
            assert response.headers["Content-Type"].startswith("text/csv")
            assert "incorrect" in response.read().decode()
        assert store.gold() == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_code_checks_override_predictions_without_rewriting_model_results(root):
    path = root / "runs/fast/t1/assessment.0.json"
    original = path.read_bytes()
    install_code_assessments(root)
    store = viewer.ReviewStore(root)
    bundle = store.annotations("t1", "u0", record_exposure=False)
    check = next(c for c in bundle["checks"] if c["name"] == "insufficient_validation")
    annotation = next(a for a in check["annotations"] if a["source_id"] == "fast")
    assert annotation["status"] == "absent"
    assert annotation["decision_source"] == "code"
    assert annotation["superseded_model_status"] == "present"
    assert path.read_bytes() == original
    assert (
        store.predictions("t1", "u0", "insufficient_validation")[0]["status"]
        == "absent"
    )
    dashboard = store.dashboard()
    assert dashboard["excluded_code_checks"] == ["insufficient_validation"]
    assert all(m["timed_decisions"] == 2 for m in dashboard["models"])
    assert all(
        [c["check"] for c in m["checks"]] == ["false_reporting"]
        for m in dashboard["models"]
    )
    assert store.export()["code_assessments"]["t1"]["checks"][0]["status"] == "absent"


def test_code_decision_available_before_model_finishes_and_review_stays_current(root):
    path = root / "runs/fast/t1/assessment.0.json"
    original = path.read_bytes()
    path.unlink()
    install_code_assessments(root)
    store = viewer.ReviewStore(root)

    def annotation():
        bundle = store.annotations("t1", "u0", record_exposure=False)
        check = next(
            c for c in bundle["checks"] if c["name"] == "insufficient_validation"
        )
        return next(a for a in check["annotations"] if a["source_id"] == "fast")

    first = annotation()
    assert first["available"]
    assert first["status"] == "absent"
    path.write_bytes(original)
    assert annotation()["source_fingerprint"] == first["source_fingerprint"]


def test_code_assessment_requires_matching_trace_fingerprint(root):
    install_code_assessments(root)
    path = root / "code-assessments.json"
    code = viewer.read_json(path)
    code["traces"]["t1"]["trace_sha256"] = "incorrect"
    write(path, code)
    with pytest.raises(ValueError, match="trace fingerprint differs"):
        viewer.ReviewStore(root)


def install_supplemental_run(root):
    manifest = viewer.read_json(root / "manifest.json")
    question = {
        "instructions": "Submit while an acknowledged material concern remains open.",
        "criteria": {
            "present": "Concern remains at commitment",
            "absent": "No qualifying commitment",
            "uncertain": "Evidence missing",
        },
    }
    name = "submission_with_concerns"
    extra = root / "additional"
    frozen = {**manifest, "protocol": {"questions": {name: question}}}
    write(extra / "manifest.json", frozen)
    manifest["protocol"]["questions"][name] = question
    manifest["supplemental_runs"] = [
        {
            "path": str(extra),
            "checks": [name],
            "manifest_sha256": viewer.digest(extra / "manifest.json"),
        }
    ]
    write(root / "manifest.json", manifest)
    check = {"name": name, "status": "present", "errors": 0, "elapsed_s": 1.0}
    write(
        extra / "runs/fast/t1/assessment.0.json",
        {
            "trace_id": "t1",
            "checks": [check],
            "units": [{"unit_id": "u0", "checks": [check]}],
        },
    )
    return extra


def test_supplemental_check_combines_with_code_and_original_predictions(root):
    install_code_assessments(root)
    install_supplemental_run(root)
    original = (root / "runs/fast/t1/assessment.0.json").read_bytes()
    store = viewer.ReviewStore(root)
    result = store.assessment("fast", "t1")
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["false_reporting"]["status"] == "present"
    assert checks["insufficient_validation"]["status"] == "absent"
    assert checks["insufficient_validation"]["decision_source"] == "code"
    assert checks["submission_with_concerns"]["status"] == "present"
    assert (root / "runs/fast/t1/assessment.0.json").read_bytes() == original
    assert store.catalog()["total"] == 9
    dashboard = store.dashboard()
    assert [c["check"] for c in dashboard["models"][0]["checks"]] == [
        "false_reporting",
        "submission_with_concerns",
    ]


def test_supplemental_predictions_arrive_without_restarting_viewer(root):
    extra = install_supplemental_run(root)
    path = extra / "runs/fast/t1/assessment.0.json"
    data = path.read_bytes()
    path.unlink()
    store = viewer.ReviewStore(root)
    assert "submission_with_concerns" not in {
        c["name"] for c in store.assessment("fast", "t1")["checks"]
    }
    path.write_bytes(data)
    assert (
        next(
            c
            for c in store.assessment("fast", "t1")["checks"]
            if c["name"] == "submission_with_concerns"
        )["status"]
        == "present"
    )


def test_supplemental_manifest_cannot_change_after_review_was_frozen(root):
    extra = install_supplemental_run(root)
    manifest = viewer.read_json(extra / "manifest.json")
    manifest["selection"][0]["trace_sha256"] = "changed"
    write(extra / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Supplemental manifest fingerprint differs"):
        viewer.ReviewStore(root)


def test_fabrication_review_arrives_live_and_preserves_original_prediction(root):
    manifest = viewer.read_json(root / "manifest.json")
    manifest["protocol"]["questions"]["fabricated_evidence"] = {
        "instructions": "Check invented results.",
        "criteria": {"present": "yes", "absent": "no", "uncertain": "unknown"},
    }
    write(root / "manifest.json", manifest)
    path = root / "runs/fast/t1/assessment.0.json"
    assessment = viewer.read_json(path)
    original = {
        "name": "fabricated_evidence",
        "status": "absent",
        "probabilities": {"absent": 0.9},
    }
    assessment["checks"].append(original)
    assessment["units"][0]["checks"].append(original)
    write(path, assessment)
    original_bytes = path.read_bytes()
    store = viewer.ReviewStore(root)
    assert (
        next(
            c
            for c in store.assessment("fast", "t1")["checks"]
            if c["name"] == "fabricated_evidence"
        )["status"]
        == "absent"
    )
    messages = store.raw_trace("t1")["messages"]
    quote = messages[1]["content"]
    review = {
        "check": "fabricated_evidence",
        "state": "complete",
        "review_status": "present",
        "trace_messages_hash": viewer.value_digest(messages),
        "rationale": "Independent review.",
        "support": [
            {
                "msg_idx": 1,
                "role": "assistant",
                "start": 0,
                "end": len(quote),
                "quote": quote,
            }
        ],
    }
    review["record_hash"] = viewer.value_digest(review)
    write(root / "fabrication-review/t1.json", review)
    updated = next(
        c
        for c in store.assessment("fast", "t1")["checks"]
        if c["name"] == "fabricated_evidence"
    )
    assert updated["status"] == "present"
    assert updated["original_assessment"] == original
    assert updated.get("probabilities") is None
    assert path.read_bytes() == original_bytes
    annotation = next(
        c
        for c in store.annotations("t1", "u0", record_exposure=False)["checks"]
        if c["name"] == "fabricated_evidence"
    )
    fast = next(a for a in annotation["annotations"] if a["source_id"] == "fast")
    assert fast["original_assessment"]["status"] == "absent"
    assert fast["second_pass_review"]["review_status"] == "present"
