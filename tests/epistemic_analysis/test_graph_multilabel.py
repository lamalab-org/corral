"""Behavioral checks for overlapping nodes and inspectable Jev decisions."""

import asyncio
import json
from dataclasses import replace

import pytest

from corral.epistemic_analysis import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
    annotate_trace,
    ontology,
)
from corral.epistemic_analysis.annotators.graph import GraphExtractor
from corral.epistemic_analysis.cli import validate_options
from corral.epistemic_analysis.graph import validate_support_quotes
from corral.epistemic_analysis.overflow import long_message_view, tool_excerpt_view
from corral.epistemic_analysis.pipeline import can_reuse


class Decisions:
    """Exercise Decisions transport/validation without making HTTP calls."""

    def __init__(self, classify, same=False, confidence=0.9):
        self.classify = classify
        self.same = same
        self.confidence = confidence
        self.bodies = []

    async def evaluate(self, model, state, questions):
        if not questions:
            return {}
        self.bodies.append(
            {
                "input": json.dumps(state),
                "questions": [
                    {
                        "name": k,
                        "type": "choice",
                        "instructions": q.get("instructions", ""),
                        "choices": [
                            {"value": v, "description": d}
                            for v, d in q["criteria"].items()
                        ],
                    }
                    for k, q in questions.items()
                ],
            }
        )
        answers = {}
        for key in questions:
            if key.startswith("same_"):
                choice = "same" if self.same else "distinct"
            elif key.startswith("claim_"):
                choice = "single"
            elif key.startswith("edge_"):
                choice = "unrelated"
            else:
                span_key, node_type = key.rsplit("_", 1)
                choice = (
                    "present"
                    if node_type in self.classify(state["candidates"][span_key])
                    else "absent"
                )
            answers[key] = {
                "type": "choice",
                "choice": choice,
                "confidence": self.confidence,
            }
        return answers


def annotate(messages, decisions, **options):
    trace = NormalizedTrace(messages, SourceReference("test", "test.json", "hash"))
    client = decisions
    backend = GraphExtractor(client).backend()
    config = AnnotationConfig("jev", **options)
    return asyncio.run(annotate_trace(trace, backend, config)), trace, backend, config


def test_current_definitions_are_sent_for_every_node_decision():
    decisions = Decisions(lambda candidate: {"H", "J", "C"}, confidence=0.7)
    graph, _, _, _ = annotate(
        [
            {
                "role": "assistant",
                "content": "The result suggests X, which I will keep despite insufficient evidence.",
            }
        ],
        decisions,
    )
    assert {n["type"] for n in graph.nodes} == {"H", "J", "C"}
    assert graph.annotation_status == "complete"
    assert len({json.dumps(n["support"]) for n in graph.nodes}) == 1
    all_types = set()
    for body in decisions.bodies:
        state = json.loads(body["input"])
        for question in body["questions"]:
            key = question["name"]
            if key.startswith("edge_"):
                continue
            all_types.add(key.rsplit("_", 1)[1])
            assert (
                state["node_ontology"]
                == ontology.PASS_A_INSTRUCTIONS.split("Return JSON with keys:")[0]
            )
            assert state["annotation_rules"] == ontology.PASS_A_SYSTEM
            assert (
                next(
                    c["description"]
                    for c in question["choices"]
                    if c["value"] == "present"
                )
                == ontology.NODE_CRITERIA[key.rsplit("_", 1)[1]]
            )
    assert all_types == {"H", "T", "J", "C"}
    assert "commitment_threshold" not in graph.provenance


@pytest.mark.parametrize(("same", "expected"), [(False, 2), (True, 1)])
def test_same_type_distinct_instances_and_duplicate_support(same, expected):
    decisions = Decisions(lambda candidate: {"H"}, same=same)
    graph, _, _, _ = annotate(
        [
            {
                "role": "assistant",
                "content": "X might explain the data. Y might explain the data.",
            }
        ],
        decisions,
    )
    assert len(graph.nodes) == expected
    if same:
        assert len(graph.nodes[0]["support"]) == 2
    else:
        assert graph.nodes[0]["support"] != graph.nodes[1]["support"]
    assert all(
        validate_support_quotes(graph.messages, n["support"])[0] for n in graph.nodes
    )


def test_judgment_consolidation_preserves_all_reasons_before_edges():
    message = (
        "The measured peak agrees with this structure, and the carbon count also agrees. "
        "Proposed structure is A."
    )
    decisions = Decisions(
        lambda c: (
            {"E"}
            if c["kind"] == "observation"
            else {"H"}
            if c["quote"] == "Proposed structure is A."
            else {"J"}
        ),
        same=True,
    )
    graph, _, _, _ = annotate(
        [
            {"role": "tool", "content": "Measured spectrum."},
            {"role": "assistant", "content": message},
        ],
        decisions,
    )
    judgment = next(n for n in graph.nodes if n["type"] == "J")
    assert [n["type"] for n in graph.nodes].count("J") == 1
    assert [n["type"] for n in graph.nodes].count("H") == 1
    assert judgment["text"] == message.removesuffix(" Proposed structure is A.")
    assert len(judgment["support"]) == 2
    assert validate_support_quotes(graph.messages, judgment["support"])[0]
    assert "jev" not in judgment
    audit = judgment["judgment_consolidation"]
    assert len(audit["original_nodes"]) == 2
    assert all(n["jev"]["choice"] == "present" for n in audit["original_nodes"])
    assert audit["decisions"][0]["answer"]["choice"] == "same"
    saw_merge = saw_edges = False
    for body in decisions.bodies:
        state = json.loads(body["input"])
        keys = [q["name"] for q in body["questions"]]
        if any(k.startswith("same_") for k in keys):
            saw_merge = True
            assert state["messages"]["m1"]["content"] == message
        if any(k.startswith("edge_") for k in keys):
            saw_edges = True
            js = [n for n in state["nodes"].values() if n["type"] == "J"]
            assert len(js) == 1
            assert js[0]["text"] == judgment["text"]
    assert saw_merge
    assert saw_edges


@pytest.mark.parametrize("conflict", ["distinct", "uncertain"])
def test_judgment_group_cannot_bridge_distinct_or_uncertain_assessments(conflict):
    class PairChoices:
        async def evaluate(self, model, state, questions):
            answers = {}
            for key in questions:
                answer = {"choice": "same", "confidence": 0.9}
                if key == "same_N1_N3":
                    answer = (
                        {"choice": "distinct", "confidence": 0.9}
                        if conflict == "distinct"
                        else {"choice": "same", "confidence": 0.55}
                    )
                answers[key] = answer
            return answers

    nodes = [
        {
            "node_id": f"N{i}",
            "type": "J",
            "time": 0,
            "text": text,
            "support": [{"msg_idx": 0, "start": i * 10, "quote": text}],
            "jev": {"choice": "present", "confidence": 0.9},
        }
        for i, text in enumerate(
            ["First reason", "Second reason", "Separate assessment"], 1
        )
    ]
    result, warnings = asyncio.run(
        GraphExtractor(PairChoices(), min_confidence=0.6)._merge_duplicates(
            "test", {}, nodes
        )
    )
    assert len(result) == 2
    assert result[0]["merged_nodes"] == ["N2"]
    assert result[1]["node_id"] == "N3"
    assert warnings


def test_judgment_group_audit_retains_original_offsets_after_message_trimming():
    quotes = ["The measured peak agrees.", "The count also agrees."]
    text = quotes[0] + "\n" * 10000 + quotes[1]
    source = NormalizedTrace(
        [{"role": "assistant", "content": text}],
        SourceReference("grouped", "grouped.json", "hash"),
    )
    view = long_message_view(source, tool_excerpt_view(source))
    assert len(view.messages[0]["content"]) < len(text)
    decisions = Decisions(lambda c: {"J"} if c["quote"] in quotes else set(), same=True)
    client = decisions
    graph = asyncio.run(
        annotate_trace(
            source,
            GraphExtractor(client).backend(),
            AnnotationConfig("jev"),
            prepared_view=view,
        )
    )
    assert len(graph.nodes) == 1
    judgment = graph.nodes[0]
    originals = judgment["judgment_consolidation"]["original_nodes"]
    for i, q in enumerate(quotes):
        expected = {
            "msg_idx": 0,
            "quote": q,
            "start": text.index(q),
            "end": text.index(q) + len(q),
        }
        assert judgment["support"][i] == expected
        assert originals[i]["support"] == [expected]


def test_final_answer_contents_can_also_be_hypothesis_judgment_and_commitment():
    decisions = Decisions(lambda candidate: {"F", "H", "J", "C"}, same=True)
    graph, _, _, _ = annotate(
        [
            {
                "role": "assistant",
                "content": "<final_answer>The result suggests X. I retain X despite insufficient evidence.</final_answer>",
            }
        ],
        decisions,
    )
    assert {n["type"] for n in graph.nodes} == {"F", "H", "J", "C"}
    assert len(graph.nodes) == 4
    assert all(len(n["support"]) == 2 for n in graph.nodes)


@pytest.mark.parametrize("types", [{"F"}, {"F", "N"}, {"F", "C"}])
def test_submit_answer_is_a_valid_final_annotation(types):
    submission = (
        '<action>submit_answer</action><action_input>{"answer":"X"}</action_input>'
    )
    graph, _, _, _ = annotate(
        [
            {"role": "assistant", "content": submission},
            {"role": "tool", "content": "answer accepted"},
            {"role": "assistant", "content": "<final_answer>X</final_answer>"},
        ],
        Decisions(lambda candidate: types),
    )
    assert {n["type"] for n in graph.nodes} == types - {"N"}
    assert all(n["time"] == 0 and n["support"][0]["quote"] == "X" for n in graph.nodes)
    assert graph.annotation_status == "complete"
    assert graph.qc["review_issues"] == []


def test_submit_answer_requires_a_final_answer_node():
    graph, _, _, _ = annotate(
        [{"role": "assistant", "content": "<action>submit_answer</action>"}],
        Decisions(lambda candidate: {"N"}),
    )
    assert graph.annotation_status == "incomplete"
    assert "Final-answer message 0 has no F node." in graph.qc["review_issues"]


def test_test_plan_and_action_count_once_and_neutral_result_stays_evidence():
    def classify(candidate):
        if candidate["kind"] == "observation":
            return {"E"}
        if "copy" in candidate["quote"]:
            return {"N"}
        return {"T"}

    graph, _, _, _ = annotate(
        [
            {
                "role": "assistant",
                "content": "I will measure X. <action>measure</action>",
            },
            {"role": "tool", "content": "Observation: X is 4."},
            {"role": "assistant", "content": "<action>copy</action>"},
            {"role": "tool", "content": "Observation: copied."},
        ],
        Decisions(classify, same=True),
        window=4,
    )
    assert [n["type"] for n in graph.nodes] == ["T", "E", "N", "E"]
    assert graph.nodes[0]["support"][0]["quote"] == "<action>measure</action>"
    assert graph.annotation_status == "complete"


def test_defaults_bounded_windows_clipping_and_full_decision_artifacts(tmp_path):
    from corral.cli import build_parser

    args = build_parser().parse_args(["analyze", "--input", "trace.json"])
    config = validate_options(args)
    assert (config.window, config.overlap, config.tool_output_max_chars) == (254, 1, 0)
    decisions = Decisions(lambda c: {"E"} if c["kind"] == "observation" else {"H"})
    graph, trace, backend, config = annotate(
        [
            {"role": "assistant", "content": "I think X."},
            {"role": "tool", "content": "Observation: " + "a" * 10000 + "END"},
            {"role": "assistant", "content": "I think Y."},
            {"role": "assistant", "content": "I think Z."},
        ],
        decisions,
        window=2,
        tool_output_max_chars=4000,
    )
    assert all(
        len(json.loads(body["input"])["messages"]) <= 2 for body in decisions.bodies
    )
    assert graph.qc["context_coverage"]["retained_tool_chars"] == 4000
    assert graph.messages == trace.messages
    node_keys = [
        question["name"]
        for body in decisions.bodies
        for question in body["questions"]
        if not question["name"].startswith("edge_")
    ]
    assert len(node_keys) == len(set(node_keys))
    # Both transports now report eligible pairs missed by graph windows.
    assert graph.annotation_status == "incomplete"
    assert any("cross graph windows" in issue for issue in graph.qc["review_issues"])
    assert not can_reuse(graph, trace, backend, config)
    # A complete window remains reusable, and configuration changes invalidate it.
    graph, trace, backend, config = annotate(trace.messages, decisions, window=4)
    assert can_reuse(graph, trace, backend, config)
    old = replace(
        backend,
        provenance={**backend.provenance, "candidate_extractor": "exact_spans_v1"},
    )
    assert not can_reuse(graph, trace, old, config)
    typesafe = replace(
        backend,
        provenance={
            **{key: value for key, value in backend.provenance.items() if key != "api"},
            "endpoint": "https://api.typesafe.ai/v1/systemone",
        },
    )
    assert not can_reuse(graph, trace, typesafe, config)


def test_invalid_annotations_require_review():
    graph, trace, backend, config = annotate(
        [{"role": "assistant", "content": "<final_answer>X</final_answer>"}],
        Decisions(lambda c: {"H"}),
    )
    assert graph.annotation_status == "incomplete"
    assert graph.qc["review_issues"] or graph.qc["uncovered_messages"]
    assert not can_reuse(graph, trace, backend, config)
    assert not any(n["type"] == "N" for n in graph.nodes)


def test_impossible_neutral_label_on_prose_never_requires_review():
    graph, trace, backend, config = annotate(
        [{"role": "assistant", "content": "Ordinary prose."}],
        Decisions(lambda c: {"N"}),
    )
    assert graph.nodes == []
    assert graph.annotation_status == "complete"
    assert graph.qc["review_issues"] == []
    assert can_reuse(graph, trace, backend, config)


@pytest.mark.parametrize("confidence", [0.04, 0.9])
def test_unlabelled_prose_is_a_valid_abstention(confidence):
    # Low-confidence "present" and confident "absent" both leave no node.
    types = {"T"} if confidence < 0.5 else set()
    graph, trace, backend, config = annotate(
        [{"role": "assistant", "content": "Let me start with mass spec and 1H NMR."}],
        Decisions(lambda c: types, confidence=confidence),
    )
    assert graph.nodes == []
    assert graph.annotation_status == "complete"
    assert graph.qc["uncovered_messages"] == [0]
    assert graph.qc["message_coverage"]["unlabelled_messages"] == [0]
    assert can_reuse(graph, trace, backend, config)


def test_task_description_evidence_exception_is_preserved():
    graph, _, _, _ = annotate(
        [
            {"role": "user", "content": "The initial measured voltage is 4 V."},
            {"role": "tool", "content": "Observation: 5 V."},
        ],
        Decisions(lambda c: {"E"}),
    )
    assert [(n["time"], n["type"]) for n in graph.nodes] == [(0, "E"), (1, "E")]


def test_unexecuted_test_plan_is_not_removed_by_an_unrelated_neutral_action():
    graph, _, _, _ = annotate(
        [
            {
                "role": "assistant",
                "content": "I will measure X next. <action>copy</action>",
            },
        ],
        Decisions(lambda c: {"N"} if c["kind"] == "action" else {"T"}),
    )
    assert [n["type"] for n in graph.nodes] == ["T", "N"]


@pytest.mark.parametrize("confidence", [0.07, 0.44, 0.9])
def test_test_wins_over_neutral_even_below_the_confidence_threshold(confidence):
    graph, _, _, _ = annotate(
        [{"role": "assistant", "content": "<action>measure</action>"}],
        Decisions(lambda c: {"T", "N"}, confidence=confidence),
    )
    assert [n["type"] for n in graph.nodes] == ["T"]
    assert graph.nodes[0]["jev"]["confidence"] == confidence
    assert graph.nodes[0]["postprocessing"] == "test_over_neutral"
    assert graph.annotation_status == "complete"
    assert graph.qc["review_issues"] == []
    assert not any("omitted m0_s0 (T)" in w for w in graph.qc["warnings"])


@pytest.mark.parametrize("types", [{"N"}, set(), {"H"}, {"H", "N"}, {"J", "C", "N"}])
def test_neutral_is_only_an_action_fallback(types):
    graph, _, _, _ = annotate(
        [{"role": "assistant", "content": "<action>copy</action>"}],
        Decisions(lambda c: types),
    )
    assert {n["type"] for n in graph.nodes} == (types - {"N"} or {"N"})
    assert graph.annotation_status == "complete"
    if not types:
        assert graph.nodes[0]["postprocessing"] == "neutral_fallback"
        assert graph.nodes[0]["jev"]["choice"] == "absent"


@pytest.mark.parametrize("budget", [40, 50])
def test_runtime_notice_is_neutral_without_a_model_decision(budget):
    notice = f"agent exhausted its {budget} interaction budget"
    decisions = Decisions(lambda c: set())
    graph, _, _, _ = annotate([{"role": "assistant", "content": notice}], decisions)
    assert decisions.bodies == []
    assert [n["type"] for n in graph.nodes] == ["N"]
    assert graph.nodes[0]["postprocessing"] == "runtime_notice_neutral_fallback"
    assert graph.nodes[0]["support"] == [
        {"msg_idx": 0, "quote": notice, "start": 0, "end": len(notice)}
    ]
    assert graph.annotation_status == "complete"
    assert graph.qc["uncovered_messages"] == []
    assert graph.qc["review_issues"] == []
