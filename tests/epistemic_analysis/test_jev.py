"""Network-free checks of Jev transport, extraction, and graph validation."""

from __future__ import annotations

import asyncio
import io
import json
from urllib.error import HTTPError

import pytest

from corral.epistemic_analysis import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
    annotate_trace,
)
from corral.epistemic_analysis.annotators import jev
from corral.epistemic_analysis.decisions import DEFAULT_DECISIONS_MODEL
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


class FakeJev:
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
        return jev.validate_answers({"answers": answers}, questions)


def run_annotation(client=None, **kwargs):
    trace = NormalizedTrace(MESSAGES, SourceReference("test", "test", "test"))
    return asyncio.run(
        annotate_trace(
            trace,
            jev.JevAnnotator(client or FakeJev(), **kwargs).backend(),
            AnnotationConfig(DEFAULT_DECISIONS_MODEL, window=4, overlap=2),
        )
    ).to_dict()


def test_spans_preserve_verbatim_quotes_and_actions():
    spans = jev.candidate_spans(MESSAGES, max_chars=12)
    assert all(
        s.quote == MESSAGES[s.msg_idx]["content"][s.offset : s.offset + len(s.quote)]
        for s in spans
    )
    assert all(s.msg_idx >= 1 for s in spans)
    actions = [s for s in spans if s.kind == "action"]
    assert len(actions) == 2
    assert '<action_input>{"x": 1}</action_input>' in actions[0].quote
    assert [s.quote for s in spans if s.kind == "final"] == ["X"]
    assert len([s for s in spans if s.kind == "observation"]) == 2


def test_annotation_graph_support_overlap_and_observations_after_neutral_calls():
    client = FakeJev()
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
    assert result["provenance"]["annotator"] == "jev"
    assert result["provenance"]["strict_support"] is True


def test_edges_require_separately_selected_support():
    result = run_annotation(FakeJev(support=False))
    assert result["edges"] == []
    assert any("no confident supporting message" in w for w in result["qc"]["warnings"])


def test_low_confidence_decisions_are_omitted():
    result = run_annotation(FakeJev(confidence=0.2), min_confidence=0.6)
    assert not any(n["type"] in {"H", "T", "J"} for n in result["nodes"])
    assert result["edges"] == []
    assert any("Low node confidence" in w for w in result["qc"]["warnings"])


def test_client_posts_decisions_payload_batches_and_retries(monkeypatch):
    bodies = []

    def fake_urlopen(request, timeout):
        assert request.full_url == jev.DEFAULT_ENDPOINT
        assert request.get_header("Authorization") == "Bearer test-key"
        assert timeout == 10
        body = json.loads(request.data)
        bodies.append(body)
        if len(bodies) == 1:
            raise HTTPError(request.full_url, 429, "rate limit", {}, io.BytesIO())
        return io.BytesIO(
            json.dumps(
                {
                    "model": "decisions-pinned-test",
                    "answers": [
                        {
                            "name": q["name"],
                            "type": "choice",
                            "choice": "H",
                            "confidence": 0.9,
                            "probabilities": [
                                {
                                    "value": c["value"],
                                    "probability": 0.9 if c["value"] == "H" else 0.1,
                                }
                                for c in q["choices"]
                            ],
                        }
                        for q in body["questions"]
                    ],
                }
            ).encode()
        )

    async def no_sleep(delay):
        assert delay == 2

    monkeypatch.setattr(jev, "urlopen", fake_urlopen)
    monkeypatch.setattr(jev.asyncio, "sleep", no_sleep)
    client = jev.JevClient("test-key", batch_size=2, timeout_s=10, max_retries=2)
    questions = {
        f"q{i}": {
            "type": "choice",
            "instructions": "Classify",
            "criteria": {"H": "Hypothesis", "other": "None"},
        }
        for i in range(3)
    }
    result = asyncio.run(
        client.evaluate(DEFAULT_DECISIONS_MODEL, {"trace": "text"}, questions)
    )
    assert len(bodies) == 3
    assert [len(body["questions"]) for body in bodies] == [2, 2, 1]
    assert all(
        body["model"] == DEFAULT_DECISIONS_MODEL
        and json.loads(body["input"]) == {"trace": "text"}
        for body in bodies
    )
    for body in bodies:
        assert set(body) == {"model", "input", "questions"}
        for question in body["questions"]:
            original = questions[question["name"]]
            assert question["instructions"] == original["instructions"]
            assert question["choices"] == [
                {"value": k, "description": v} for k, v in original["criteria"].items()
            ]
    assert set(result) == set(questions)
    assert result["q0"]["model"] == "decisions-pinned-test"


def test_client_does_not_retry_authentication_errors(monkeypatch):
    calls = []

    def fail(body):
        calls.append(body)
        raise HTTPError(jev.DEFAULT_ENDPOINT, 401, "Unauthorized", {}, io.BytesIO())

    client = jev.JevClient("test-key")
    monkeypatch.setattr(client, "_post", fail)
    with pytest.raises(jev.JevError, match="HTTP 401"):
        asyncio.run(
            client.evaluate(DEFAULT_DECISIONS_MODEL, {}, {"q": {"type": "noul"}})
        )
    assert len(calls) == 1


def test_decisions_transport_preserves_graph_prompts_and_extraction_steps(monkeypatch):
    class ResolvedJev(FakeJev):
        async def evaluate(self, model, state, questions):
            answers = await super().evaluate(model, state, questions)
            return {key: {**answer, "model": model} for key, answer in answers.items()}

    reference = ResolvedJev()
    expected = run_annotation(reference)
    decisions = ResolvedJev()

    def respond(request, timeout):
        body = json.loads(request.data)
        questions = {
            q["name"]: {
                "type": q["type"],
                "instructions": q["instructions"],
                "criteria": {c["value"]: c["description"] for c in q["choices"]},
            }
            for q in body["questions"]
        }
        answers = asyncio.run(
            decisions.evaluate(body["model"], json.loads(body["input"]), questions)
        )
        return io.BytesIO(
            json.dumps(
                {
                    "model": body["model"],
                    "answers": [
                        {
                            **answer,
                            "name": key,
                            "probabilities": [
                                {"value": value, "probability": probability}
                                for value, probability in answer[
                                    "probabilities"
                                ].items()
                            ],
                        }
                        for key, answer in answers.items()
                    ],
                }
            ).encode()
        )

    monkeypatch.setattr(jev, "urlopen", respond)
    actual = run_annotation(jev.JevClient("test-key", batch_size=200))
    # Both transports skip HTTP requests when a stage has no questions.
    assert decisions.requests == [
        (state, questions) for state, questions in reference.requests if questions
    ]
    assert actual["nodes"] == expected["nodes"]
    assert actual["edges"] == expected["edges"]
    assert {
        key: value
        for key, value in actual["qc"].items()
        if key != "request_diagnostics"
    } == {
        key: value
        for key, value in expected["qc"].items()
        if key != "request_diagnostics"
    }
    assert actual["provenance"]["api"] == "openai-decisions"


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "extra",
        "reordered",
        "refusal",
        "duplicate_probability",
        "bad_probability",
    ],
)
def test_incomplete_decisions_responses_cannot_produce_partial_graphs(
    monkeypatch, invalid
):
    questions = {
        key: {
            "type": "choice",
            "instructions": "Classify",
            "criteria": {"H": "Hypothesis", "N": "Action"},
        }
        for key in ("q1", "q2")
    }
    payload = {
        "answers": [
            {
                "name": key,
                "type": "choice",
                "choice": "H",
                "confidence": 0.9,
                "probabilities": [
                    {"value": "H", "probability": 0.9},
                    {"value": "N", "probability": 0.1},
                ],
            }
            for key in questions
        ]
    }
    if invalid == "missing":
        payload["answers"].pop()
    elif invalid == "extra":
        payload["answers"].append(payload["answers"][0])
    elif invalid == "reordered":
        payload["answers"].reverse()
    elif invalid == "refusal":
        payload["answers"][0] = {"name": "q1", "type": "refusal"}
    elif invalid == "duplicate_probability":
        payload["answers"][0]["probabilities"].append({"value": "H", "probability": 0})
    else:
        payload["answers"][0]["probabilities"][0]["probability"] = float("nan")
    client = jev.JevClient("test-key", max_retries=1)
    monkeypatch.setattr(client, "_post", lambda body: payload)
    with pytest.raises(jev.JevError):
        asyncio.run(client.evaluate(DEFAULT_DECISIONS_MODEL, {}, questions))


def test_decisions_question_limit_splits_only_batches_and_preserves_input(monkeypatch):
    bodies = []

    def post(body):
        bodies.append(body)
        return {
            "answers": [
                {"name": q["name"], "type": "predicate", "probability": 0.5}
                for q in body["questions"]
            ]
        }

    client = jev.JevClient("test-key", batch_size=500)
    monkeypatch.setattr(client, "_post", post)
    questions = {
        f"q{i}": {"type": "noul", "instructions": "Check the evidence."}
        for i in range(201)
    }
    result = asyncio.run(
        client.evaluate(DEFAULT_DECISIONS_MODEL, {"text": "🧪 evidence"}, questions)
    )
    assert list(result) == list(questions)
    assert [len(body["questions"]) for body in bodies] == [200, 1]
    assert all(json.loads(body["input"]) == {"text": "🧪 evidence"} for body in bodies)


def test_decisions_uses_openai_credentials_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "old-typesafe-key")
    monkeypatch.setenv("JEV_API_KEY", "old-jev-key")
    assert jev.JevClient().api_key == "openai-test-key"
    assert jev.JevClient("explicit-key").api_key == "explicit-key"
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(jev.JevError, match="OPENAI_API_KEY"):
        asyncio.run(
            jev.JevClient().evaluate(
                DEFAULT_DECISIONS_MODEL, {}, {"q": {"type": "noul"}}
            )
        )


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
    with pytest.raises(jev.JevError):
        jev.validate_answers({"answers": {"q": answer}}, {"q": question})


@pytest.mark.parametrize("probability", [float("nan"), -0.1, 1.1, True])
def test_invalid_commitment_probabilities_fail(probability):
    with pytest.raises(jev.JevError, match="Noul"):
        jev.validate_answers(
            {"answers": {"q": {"type": "noul", "noul": probability}}},
            {"q": {"type": "noul"}},
        )
