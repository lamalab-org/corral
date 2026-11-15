"""Network-free checks of Jev transport, extraction, and graph validation."""

from __future__ import annotations

import asyncio

import pytest

from corral.epistemic_analysis import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
    annotate_trace,
)
from corral.epistemic_analysis.annotators import graph as extraction
from corral.epistemic_analysis.annotators.local import DEFAULT_ANNOTATION_MODEL
from corral.epistemic_analysis.graph import (
    _filter_invalid_edges,
    validate_support_quotes,
)
from corral.epistemic_analysis.ontology import NODE_TYPES


def choice_answer(question, choice, confidence=0.9):
    others = len(question["criteria"]) - 1
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": {
            key: 0.9 if key == choice else 0.1 / others for key in question["criteria"]
        },
    }


MESSAGES = [
    {"role": "system", "content": "Solve the scientific task."},
    {"role": "user", "content": "Find the answer."},
    {
        "role": "assistant",
        "content": 'I think X is the answer. <action>measure</action><action_input>{"x": 1}</action_input>',
    },
    {"role": "user", "content": "Observation: X was measured."},
    {
        "role": "assistant",
        "content": "The measurement supports X. <final_answer>X</final_answer>",
    },
    {
        "role": "assistant",
        "content": '<action>write_file</action><action_input>{"text": "X"}</action_input>',
    },
    {"role": "user", "content": "Observation: File copied."},
]


class FakeQuestions:
    def __init__(self, *, support=True, confidence=0.9):
        self.support = support
        self.confidence = confidence
        self.requests = []

    async def evaluate(self, model, state, questions):
        self.requests.append((state, questions))
        answers = {}
        for key, question in questions.items():
            if key.startswith("same_"):
                selected = "distinct"
            elif key.startswith("claim_"):
                selected = "single"
            elif key.startswith("edge_"):
                _, src, dst = key.split("_")
                relation, message = {
                    ("H", "T"): ("tests", "m2"),
                    ("T", "E"): ("observes", "m3"),
                    ("E", "J"): ("informs", "m4"),
                }.get(
                    (state["nodes"][src]["type"], state["nodes"][dst]["type"]),
                    ("unrelated", "none"),
                )
                if state["nodes"][src]["time"] > state["nodes"][dst]["time"]:
                    relation, message = "unrelated", "none"
                selected = message if "none" in question["criteria"] else relation
                if not self.support and "none" in question["criteria"]:
                    selected = "none"
            else:
                candidate_key, node_type = key.rsplit("_", 1)
                candidate = state["candidates"][candidate_key]
                text = candidate["quote"]
                types = set()
                if text.startswith("I think"):
                    types = {"H"}
                elif text.startswith("The measurement"):
                    types = {"J"}
                elif text.startswith("<action>measure"):
                    types = {"T"}
                elif text.startswith("<action>write_file"):
                    types = {"N"}
                elif candidate["kind"] == "observation":
                    types = {"E"}
                elif candidate["kind"] == "final":
                    types = {"F", "C"}
                selected = "present" if node_type in types else "absent"
            answers[key] = choice_answer(question, selected, self.confidence)
        # Also exercise the exact HTTP response contract in the fake.
        return extraction.validate_answers({"answers": answers}, questions)


def run_annotation(client=None, **kwargs):
    trace = NormalizedTrace(MESSAGES, SourceReference("test", "test", "test"))
    return asyncio.run(
        annotate_trace(
            trace,
            extraction.GraphExtractor(client or FakeQuestions(), **kwargs).backend(),
            AnnotationConfig(DEFAULT_ANNOTATION_MODEL, window=4, overlap=2),
        )
    ).to_dict()


def test_spans_preserve_verbatim_quotes_and_actions():
    spans = extraction.candidate_spans(MESSAGES, max_chars=12)
    assert all(
        s.quote == MESSAGES[s.msg_idx]["content"][s.offset : s.offset + len(s.quote)]
        for s in spans
    )
    assert all(s.msg_idx >= 1 or s.kind == "task" for s in spans)
    assert any(s.msg_idx == 0 and s.kind == "task" for s in spans)
    actions = [s for s in spans if s.kind == "action"]
    assert len(actions) == 2
    assert '<action_input>{"x": 1}</action_input>' in actions[0].quote
    assert [s.quote for s in spans if s.kind == "final"] == ["X"]
    assert len([s for s in spans if s.kind == "observation"]) == 2


def test_annotation_graph_support_overlap_and_observations_after_neutral_calls():
    client = FakeQuestions()
    result = run_annotation(client)
    nodes = result["nodes"]
    by_type = {kind: [n for n in nodes if n["type"] == kind] for kind in NODE_TYPES}
    assert len(by_type["H"]) == len(by_type["T"]) == 1
    assert len(by_type["E"]) == 2
    assert len(by_type["J"]) == len(by_type["F"]) == len(by_type["C"]) == 1
    assert len(by_type["N"]) == 1
    assert by_type["H"][0]["hypothesis"]["canonical"] == "I think X is the answer."
    assert {e["relation"] for e in result["edges"]} == {"tests", "observes", "informs"}
    assert len(result["edges"]) == 3
    for item in nodes + result["edges"]:
        assert item["support"]
        assert validate_support_quotes(MESSAGES, item["support"])[0]
    assert _filter_invalid_edges(result["edges"], nodes)[1] == []
    keys = [
        key
        for _, questions in client.requests
        for key in questions
        if not key.startswith("edge_")
    ]
    assert len(keys) == len(set(keys))  # overlap never reclassifies the same span
    assert result["provenance"]["annotator"] == "graph"
    assert result["provenance"]["strict_support"] is True


def test_edges_require_separately_selected_support():
    result = run_annotation(FakeQuestions(support=False))
    assert result["edges"] == []
    assert any("no confident supporting message" in w for w in result["qc"]["warnings"])


def test_low_confidence_decisions_are_omitted():
    result = run_annotation(FakeQuestions(confidence=0.2), min_confidence=0.6)
    assert not any(n["type"] in {"H", "T", "J"} for n in result["nodes"])
    assert result["edges"] == []
    assert any("Low node confidence" in w for w in result["qc"]["warnings"])


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {
            "type": "choice",
            "choice": "invented",
            "confidence": 1,
            "probabilities": {"H": 1, "other": 0},
        },
        {
            "type": "choice",
            "choice": "H",
            "confidence": float("nan"),
            "probabilities": {"H": 1, "other": 0},
        },
        {
            "type": "choice",
            "choice": "H",
            "confidence": 1,
            "probabilities": {"H": 0.1, "other": 0.1},
        },
        {"type": "choice", "choice": "H", "confidence": 1, "probabilities": {"H": 1}},
    ],
)
def test_invalid_answers_fail_instead_of_producing_partial_graphs(answer):
    question = {"type": "choice", "criteria": {"H": "Hypothesis", "other": "None"}}
    with pytest.raises(extraction.AnnotationError):
        extraction.validate_answers({"answers": {"q": answer}}, {"q": question})


@pytest.mark.parametrize("probability", [float("nan"), -0.1, 1.1, True])
def test_invalid_commitment_probabilities_fail(probability):
    with pytest.raises(extraction.AnnotationError, match="Noul"):
        extraction.validate_answers(
            {"answers": {"q": {"type": "noul", "noul": probability}}},
            {"q": {"type": "noul"}},
        )
