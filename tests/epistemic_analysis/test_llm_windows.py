"""Repeated context must not create repeated epistemic events."""

import asyncio
import copy
import re
from dataclasses import replace

import pytest

from corral.epistemic_analysis import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
    annotate_trace,
)
from corral.epistemic_analysis.annotators import llm
from corral.epistemic_analysis.pipeline import can_reuse


def node(kind, index, quote, text):
    return {
        "type": kind,
        "time": index,
        "text": text,
        "support": [{"msg_idx": index, "quote": quote}],
    }


@pytest.mark.parametrize(("window", "overlap"), [(2, 1), (3, 1), (3, 0), (1, 0)])
def test_preamble_and_overlap_do_not_repeat_actions(window, overlap):
    action = "<action>read_file</action>"
    messages = [
        {"role": "user", "content": "Read the instructions."},
        {"role": "assistant", "content": action},
        {"role": "user", "content": "Observation: instructions."},
        {"role": "assistant", "content": action},
        {"role": "user", "content": "Observation: more instructions."},
        {"role": "assistant", "content": "Finished."},
    ]
    calls = []

    async def respond(**kwargs):
        calls.append(kwargs)
        visible = [
            int(i) for i in re.findall(r"^\[(\d+)\] role=", kwargs["user"], re.M)
        ]
        return {
            "nodes": [
                node(
                    "T" if len(calls) == 1 or i != 1 else "N",
                    i,
                    action,
                    f"Different paraphrase {len(calls)} of action {i}",
                )
                for i in visible
                if messages[i]["content"] == action
            ]
        }

    nodes, _ = asyncio.run(
        llm._extract_nodes(
            messages,
            "fixture",
            window,
            overlap,
            100,
            call=respond,
        )
    )
    assert [n["time"] for n in nodes] == [1, 3]
    assert len({n["node_id"] for n in nodes}) == 2
    assert nodes[0]["type"] == ("N" if window == 1 else "T")


def test_preserves_distinct_instances_and_multiple_labels_in_one_message():
    text = "X may explain this. Y may explain this. I retain X."
    messages = [{"role": "assistant", "content": text}]
    expected = [
        node("H", 0, "X may explain this.", "X"),
        node("H", 0, "Y may explain this.", "Y"),
        node("J", 0, "I retain X.", "Judgment about X"),
        node("C", 0, "I retain X.", "Commitment to X"),
    ]

    async def respond(**kwargs):
        return {"nodes": copy.deepcopy(expected)}

    nodes, _ = asyncio.run(
        llm._extract_nodes(messages, "fixture", 2, 1, 100, call=respond)
    )
    assert [(n["type"], n["text"]) for n in nodes] == [
        (n["type"], n["text"]) for n in expected
    ]


def test_edges_receive_one_test_per_action_and_old_cache_is_invalid(monkeypatch):
    messages = [
        {"role": "user", "content": "Measure X."},
        {"role": "assistant", "content": "<action>measure</action>"},
        {"role": "tool", "content": "Observation: 4."},
        {"role": "assistant", "content": "X is four."},
    ]
    edge_inputs = []
    node_calls = 0

    async def respond(**kwargs):
        nonlocal node_calls
        if kwargs["system"] == llm.PASS_A_SYSTEM:
            node_calls += 1
            return {
                "nodes": [
                    node(
                        "T",
                        1,
                        messages[1]["content"],
                        f"Measure X, wording {node_calls}",
                    )
                ]
            }
        edge_inputs.append(kwargs["user"])
        return {"edges": []}

    monkeypatch.setattr(llm, "llm_json_call_async", respond)
    source = NormalizedTrace(
        messages, SourceReference("fixture", "fixture.json", "hash")
    )
    backend, config = llm.LLMAnnotator().backend(), AnnotationConfig("fixture")
    backend = replace(backend, extract_trace_checks=None)
    graph = asyncio.run(annotate_trace(source, backend, config))
    assert [(n["type"], n["time"]) for n in graph.nodes] == [("T", 1), ("E", 2)]
    assert edge_inputs
    assert all(text.count('"type": "T"') == 1 for text in edge_inputs)
    assert can_reuse(graph, source, backend, config)
    legacy = replace(
        backend,
        provenance={
            k: v for k, v in backend.provenance.items() if k != "node_window_policy"
        },
    )
    assert not can_reuse(graph, source, legacy, config)


def test_support_outside_request_and_future_messages_are_rejected():
    messages = [{"role": "assistant", "content": f"Claim {i}."} for i in range(4)]
    calls = 0

    async def respond(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"nodes": [node("H", 3, "Claim 3.", "Unseen future message")]}
        return {
            "nodes": [
                {**node("H", calls, f"Claim {calls}.", "Wrong time"), "time": 0},
                {
                    **node("C", 2, "Claim 2.", "Preamble-only support"),
                    "support": [{"msg_idx": 0, "quote": "Claim 0."}],
                },
            ]
        }

    nodes, _ = asyncio.run(
        llm._extract_nodes(messages, "fixture", 2, 1, 100, call=respond)
    )
    assert [n["time"] for n in nodes] == [2, 3]


@pytest.mark.parametrize(
    "payload", [{}, {"nodes": None}, {"nodes": {}}, {"nodes": [None]}]
)
def test_malformed_node_response_is_not_an_abstention(payload):
    async def respond(**kwargs):
        return payload

    with pytest.raises(llm.LLMError):
        asyncio.run(
            llm._extract_nodes(
                [{"role": "assistant", "content": "A claim."}],
                "fixture",
                16,
                4,
                100,
                call=respond,
            )
        )


def test_empty_nodes_are_allowed_and_revised_prompt_is_fingerprinted():
    calls = []

    async def respond(**kwargs):
        calls.append(kwargs)
        return {"nodes": []}

    nodes, warnings = asyncio.run(
        llm._extract_nodes(
            [{"role": "assistant", "content": "A fragment"}],
            "fixture",
            16,
            4,
            100,
            call=respond,
        )
    )
    assert nodes == []
    assert warnings == []
    assert "An empty or unlabelled message is allowed" in calls[0]["system"]
    assert "do not reconstruct" in calls[0]["system"].lower()
    assert "in [0, 1)" in calls[0]["user"]
    provenance = llm.LLMAnnotator().backend().provenance
    assert provenance["prompt_version"] == "transcript-labels-v2"
    assert len(provenance["prompt_hash"]) == 64
