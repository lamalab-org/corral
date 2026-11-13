"""Task facts and overlapping H/J roles retain distinct source semantics."""

import asyncio
from dataclasses import replace

from corral.epistemic_analysis.annotators.jev import JevAnnotator, candidate_spans
from corral.epistemic_analysis.context import annotation_context, build_annotation_view
from corral.epistemic_analysis.graph import (
    _filter_invalid_edges,
    validate_support_quotes,
)
from corral.epistemic_analysis.ontology import relation_allowed
from corral.epistemic_analysis.overflow import long_message_view, tool_excerpt_view
from corral.epistemic_analysis.pipeline import annotate_trace
from corral.epistemic_analysis.schema import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
)


class Choices:
    def __init__(self):
        self.bodies = []

    async def evaluate(self, model, state, questions):
        self.bodies.append((state, questions))
        answers = {}
        for key, question in questions.items():
            if key.startswith("same_"):
                choice = "distinct"
            elif key.startswith("edge_"):
                choice = "unrelated"
            else:
                span, kind = key.rsplit("_", 1)
                candidate = state["candidates"][span]
                quote = candidate["quote"]
                kinds = set()
                if candidate["kind"] == "observation" or (
                    candidate["kind"] == "task" and "measured" in quote
                ):
                    kinds.add("E")
                if candidate["kind"] == "prose":
                    if "consistent" in quote:
                        kinds.add("J")
                    if "candidate" in quote:
                        kinds.add("H")
                    if "suggests" in quote:
                        kinds.update({"H", "J"})
                choice = "present" if kind in kinds else "absent"
            answers[key] = {
                "choice": choice,
                "confidence": 1.0,
                "probabilities": {k: float(k == choice) for k in question["criteria"]},
            }
        return answers


def run(trace, *, view=None):
    client = Choices()
    backend = JevAnnotator(client).backend(stop_at_submission=False)
    graph = asyncio.run(
        annotate_trace(
            trace,
            backend,
            AnnotationConfig("test", window=32, overlap=16, tool_output_max_chars=0),
            prepared_view=view,
        )
    )
    return graph, client


def test_mixed_task_prompt_labels_exact_given_fact_and_provenance():
    trace = NormalizedTrace(
        [
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": "Identify the sample. Its measured mass is 180 Da. Additional data are in spectra.csv.",
            },
            {
                "role": "assistant",
                "content": "The band is consistent with a carbonyl; a plausible candidate is acetophenone.",
            },
            {"role": "tool", "content": "Spectrum data."},
        ],
        SourceReference("trace", "trace.json", "hash"),
    )
    graph, client = run(trace)
    given = [n for n in graph.nodes if n.get("evidence_origin") == "task_given"]
    observed = [
        n for n in graph.nodes if n.get("evidence_origin") == "tool_observation"
    ]
    assert len(given) == len(observed) == 1
    assert given[0]["text"] == "Its measured mass is 180 Da."
    assert given[0]["evidence_source_id"] != observed[0]["evidence_source_id"]
    h, j = (
        next(n for n in graph.nodes if n["type"] == "H"),
        next(n for n in graph.nodes if n["type"] == "J"),
    )
    assert h["text"] == "a plausible candidate is acetophenone."
    assert j["text"] == "The band is consistent with a carbonyl;"
    assert h["support"] != j["support"]
    assert all(
        validate_support_quotes(graph.messages, n["support"])[0] for n in graph.nodes
    )
    task_target = {"type": "E", "evidence_origin": "task_given"}
    assert not relation_allowed("observes", {"type": "T"}, task_target)
    assert relation_allowed("informs", task_target, {"type": "H"})
    for state, questions in client.bodies:
        if "nodes" in state:
            assert any(
                n.get("evidence_origin") == "task_given"
                for n in state["nodes"].values()
            )
        for key in questions:
            assert not key.endswith("_T") or not key.startswith(("m0_", "m1_"))


def test_shared_h_j_passage_keeps_both_roles_without_artificial_sequence():
    trace = NormalizedTrace(
        [{"role": "assistant", "content": "The band suggests a ketone."}],
        SourceReference("trace", "trace.json", "hash"),
    )
    graph, client = run(trace)
    h, j = (
        next(n for n in graph.nodes if n["type"] == "H"),
        next(n for n in graph.nodes if n["type"] == "J"),
    )
    assert h["shared_support_id"] == j["shared_support_id"]
    assert h["epistemic_role"] != j["epistemic_role"]
    assert h["shared_support_with"] == [j["node_id"]]
    assert not relation_allowed("informs", j, h)
    edge = {"src": j["node_id"], "dst": h["node_id"], "relation": "informs"}
    assert _filter_invalid_edges([edge], graph.nodes)[0] == []
    for _, questions in client.bodies:
        question = questions.get(f"edge_{j['node_id']}_{h['node_id']}")
        if question:
            assert "informs" not in question["criteria"]


def test_shared_task_evidence_keeps_same_identity_across_scientist_units():
    base = NormalizedTrace(
        [
            {"role": "user", "content": "The measured mass is 180 Da."},
            {"role": "assistant", "content": "A candidate is X."},
        ],
        SourceReference("parent", "trace.json", "hash"),
        message_sources=[
            {
                "origin": "reconstructed_task_prompt",
                "parent_msg_idx": 0,
                "context_only": True,
            },
            {"parent_msg_idx": 12},
        ],
    )
    records = []
    for unit in ("main", "node_1"):
        trace = replace(
            base,
            source=replace(
                base.source,
                identity=f"parent#{unit}",
                details={"scientist_unit": {"kind": unit}},
            ),
        )
        graph, _ = run(trace)
        records.append(next(n for n in graph.nodes if n["type"] == "E"))
    assert records[0]["evidence_id"] == records[1]["evidence_id"]
    assert records[0]["evidence_provenance"]["context_only"]
    assert records[0]["evidence_provenance"]["reconstructed_task_prompt"]


def test_task_claim_ranges_do_not_cross_trim_markers():
    text = (
        "The measured mass is 180 Da.\n"
        + "\n" * 10000
        + "The measured temperature is 300 K."
    )
    trace = NormalizedTrace(
        [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "A candidate is X."},
        ],
        SourceReference("trace", "trace.json", "hash"),
    )
    view = long_message_view(trace, tool_excerpt_view(trace))
    graph, _ = run(trace, view=view)
    facts = [n for n in graph.nodes if n["type"] == "E"]
    assert len(facts) == 2
    assert facts[-1]["support"][0]["start"] == text.index("The measured temperature")
    assert all("ANNOTATION" not in n["text"] for n in facts)


def test_metadata_identified_observations_remain_tool_evidence():
    trace = NormalizedTrace(
        [
            {"role": "assistant", "content": "Read the data."},
            {"role": "user", "content": "measured spectrum"},
        ],
        SourceReference("trace", "trace.json", "hash"),
        message_sources=[{}, {"tool_call_id": "call-1"}],
    )
    graph, _ = run(trace)
    assert (
        next(n for n in graph.nodes if n["type"] == "E")["evidence_origin"]
        == "tool_observation"
    )


def test_clause_splitting_does_not_split_nested_structured_data():
    text = 'A candidate is {"reason": "first clause; second clause", "mass": 180.2}.'
    trace = NormalizedTrace(
        [{"role": "assistant", "content": text}],
        SourceReference("trace", "trace.json", "hash"),
    )
    config = AnnotationConfig("test", tool_output_max_chars=0)
    view = build_annotation_view(trace, config, stop_at_submission=False)
    with annotation_context(view, config, []):
        spans = candidate_spans(view.messages, 1200)
    assert len(spans) == 1
    assert spans[0].quote == text
