"""Impossible labels never reach inference; valid labels and fallbacks survive."""

import asyncio

import pytest

from corral.epistemic_analysis.annotators.graph import GraphExtractor, Span
from corral.epistemic_analysis.ontology import NODE_TYPES


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        (Span(0, 0, "A hypothesis.", "prose"), {"H", "T", "J", "C"}),
        (Span(0, 0, "Measured 4 V.", "observation"), {"E"}),
        (Span(0, 0, "Initial voltage: 4 V.", "task"), {"E"}),
        (Span(0, 0, "<action>measure</action>", "action"), {"H", "T", "J", "C", "N"}),
        (Span(0, 0, "Answer A.", "final"), {"H", "T", "J", "C", "F"}),
        (
            Span(0, 0, "<action>submit_answer</action>", "action"),
            set(NODE_TYPES) - {"E"},
        ),
        (Span(0, 0, "{}", "action", "submit_answer"), set(NODE_TYPES) - {"E"}),
        (Span(0, 0, "A", "final", "submit_answer"), {"H", "T", "J", "C", "F"}),
    ],
)
def test_questions_and_answer_validation_use_the_same_eligibility(
    monkeypatch, span, expected
):
    from corral.epistemic_analysis.annotators import graph

    monkeypatch.setattr(graph, "candidate_spans", lambda *args: [span])

    class Present:
        async def evaluate(self, model, state, questions):
            if all(key.startswith("claim_") for key in questions):
                return {
                    key: {"type": "choice", "choice": "single", "confidence": 0.9}
                    for key in questions
                }
            assert list(questions) == [
                f"{span.key}_{t}" for t in NODE_TYPES if t in expected
            ]
            # Extra answers must still be rejected even if a client returns them.
            return {
                f"{span.key}_{t}": {
                    "type": "choice",
                    "choice": "present",
                    "confidence": 0.9,
                }
                for t in NODE_TYPES
            }

    nodes, _ = asyncio.run(
        GraphExtractor(Present()).extract_nodes(
            [
                {
                    "role": "tool" if span.kind == "observation" else "assistant",
                    "content": span.quote,
                }
            ],
            "test",
            10,
            0,
            100,
        )
    )
    assert {n["type"] for n in nodes} == expected - {"N"}


@pytest.mark.parametrize("submission", [False, True])
@pytest.mark.parametrize("confidence", [0.05, 0.9])
def test_action_with_no_accepted_label_keeps_neutral_answer(submission, confidence):
    action = "submit_answer" if submission else "copy"

    class Absent:
        async def evaluate(self, model, state, questions):
            assert "m0_s0_E" not in questions
            assert ("m0_s0_F" in questions) == submission
            assert "m0_s0_N" in questions
            return {
                key: {"type": "choice", "choice": "absent", "confidence": confidence}
                for key in questions
            }

    nodes, _ = asyncio.run(
        GraphExtractor(Absent()).extract_nodes(
            [{"role": "assistant", "content": f"<action>{action}</action>"}],
            "test",
            10,
            0,
            100,
        )
    )
    assert [n["type"] for n in nodes] == ["N"]
    assert nodes[0]["postprocessing"] == "neutral_fallback"
    assert nodes[0]["jev"]["choice"] == "absent"
    assert nodes[0]["jev"]["confidence"] == confidence
