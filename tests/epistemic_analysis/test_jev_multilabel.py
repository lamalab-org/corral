"""Behavioral checks for overlapping nodes and inspectable Jev decisions."""

import ast
import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from corral.epistemic_analysis import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
    annotate_trace,
    ontology,
    read_graph,
)
from corral.epistemic_analysis.annotators.jev import JevAnnotator, JevClient
from corral.epistemic_analysis.cli import validate_options
from corral.epistemic_analysis.graph import validate_support_quotes
from corral.epistemic_analysis.pipeline import can_reuse
from corral.epistemic_analysis.storage import save_annotation_artifacts


class Decisions:
    """Exercise Decisions transport/validation without making HTTP calls."""

    def __init__(self, classify, same=False, confidence=0.9):
        self.classify = classify
        self.same = same
        self.confidence = confidence
        self.bodies = []

    def post(self, body):
        self.bodies.append(body)
        state = json.loads(body["input"])
        answers = []
        for question in body["questions"]:
            key = question["name"]
            assert question["type"] == "choice"
            assert "other" not in {c["value"] for c in question["choices"]}
            if key.startswith("same_"):
                choice = "same" if self.same else "distinct"
            elif key.startswith("edge_"):
                choice = "unrelated"
            else:
                span_key, node_type = key.rsplit("_", 1)
                candidate = state["candidates"][span_key]
                choice = (
                    "present" if node_type in self.classify(candidate) else "absent"
                )
            alternatives = len(question["choices"]) - 1
            answers.append(
                {
                    "name": key,
                    "type": "choice",
                    "choice": choice,
                    "confidence": self.confidence,
                    "probabilities": [
                        {
                            "value": c["value"],
                            "probability": self.confidence
                            if c["value"] == choice
                            else (1 - self.confidence) / alternatives,
                        }
                        for c in question["choices"]
                    ],
                }
            )
        return {"model": "decisions-resolved", "answers": answers}


def annotate(messages, decisions, **options):
    trace = NormalizedTrace(messages, SourceReference("test", "test.json", "hash"))
    client = JevClient("not-saved", batch_size=8)
    client._post = decisions.post
    backend = replace(JevAnnotator(client).backend(), extract_trace_checks=None)
    config = AnnotationConfig("jev", **options)
    return asyncio.run(annotate_trace(trace, backend, config)), trace, backend, config


def test_current_definitions_are_sent_for_every_node_decision():
    original = Path(__file__).resolve().parents[2] / "reasoning_reports/analyze.py"
    assignments = {
        target.id: ast.literal_eval(statement.value)
        for statement in ast.parse(original.read_text()).body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        and target.id in {"PASS_A_SYSTEM", "PASS_A_INSTRUCTIONS"}
    }
    # Preserve the legacy structure while sharing current action and commitment
    # definitions across LLM and Jev annotation.
    assignments["PASS_A_INSTRUCTIONS"] = assignments["PASS_A_INSTRUCTIONS"].replace(
        "tag <final_answer>", "action `submit_answer`"
    )
    for kind in ("T", "N", "C"):
        assignments["PASS_A_INSTRUCTIONS"] = re.sub(
            rf"^{kind} = .+$",
            f"{kind} = {ontology.NODE_CRITERIA[kind]}",
            assignments["PASS_A_INSTRUCTIONS"],
            flags=re.MULTILINE,
        )
    assert assignments["PASS_A_SYSTEM"] == ontology.PASS_A_SYSTEM
    assert assignments["PASS_A_INSTRUCTIONS"] == ontology.PASS_A_INSTRUCTIONS
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
                == assignments["PASS_A_INSTRUCTIONS"].split("Return JSON with keys:")[0]
            )
            assert state["annotation_rules"] == assignments["PASS_A_SYSTEM"]
            assert (
                next(
                    c["description"]
                    for c in question["choices"]
                    if c["value"] == "present"
                )
                == ontology.NODE_CRITERIA[key.rsplit("_", 1)[1]]
            )
    assert all_types == set(ontology.NODE_TYPES)
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
    assert all(
        n["time"] == 0 and n["support"][0]["quote"] == submission for n in graph.nodes
    )
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
    )
    assert [n["type"] for n in graph.nodes] == ["T", "E", "N", "E"]
    assert graph.nodes[0]["support"][0]["quote"] == "<action>measure</action>"
    assert graph.annotation_status == "complete"


def test_defaults_bounded_windows_clipping_and_full_decision_artifacts(tmp_path):
    from corral.cli import build_parser

    args = build_parser().parse_args(
        ["analyze", "--input", "trace.json", "--annotator", "jev"]
    )
    config = validate_options(args)
    assert (config.window, config.overlap, config.tool_output_max_chars) == (2, 1, 4000)
    decisions = Decisions(lambda c: {"E"} if c["kind"] == "observation" else {"H"})
    graph, trace, backend, config = annotate(
        [
            {"role": "assistant", "content": "I think X."},
            {"role": "tool", "content": "Observation: " + "a" * 10000 + "END"},
            {"role": "assistant", "content": "I think Y."},
            {"role": "assistant", "content": "I think Z."},
        ],
        decisions,
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
    assert graph.annotation_requests
    assert any(
        a["choice"] == "absent"
        for r in graph.annotation_requests
        for a in r["response"]["answers"]
    )
    assert "not-saved" not in json.dumps(graph.annotation_requests)
    assert "Authorization" not in json.dumps(graph.annotation_requests)
    save_annotation_artifacts(graph, tmp_path)
    saved = json.loads(
        (tmp_path / graph.provenance["annotation_requests_artifact"]).read_text()
    )
    assert saved == graph.annotation_requests
    assert read_graph(graph.to_dict()).annotation_requests == saved
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


@pytest.mark.parametrize(
    ("text", "types"),
    [
        ("<final_answer>X</final_answer>", {"H"}),
        ("Ordinary prose.", {"N"}),
    ],
)
def test_invalid_annotations_require_review(text, types):
    graph, trace, backend, config = annotate(
        [{"role": "assistant", "content": text}], Decisions(lambda c: types)
    )
    assert graph.annotation_status == "incomplete"
    assert graph.qc["review_issues"] or graph.qc["uncovered_messages"]
    assert not can_reuse(graph, trace, backend, config)
    assert not any(n["type"] == "N" for n in graph.nodes)


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
