"""Graph preprocessing keeps original source coordinates and excludes cut markers."""

import asyncio
import copy
import json

import pytest

from corral.epistemic_analysis.annotators.graph import (
    AnnotationBackend,
    candidate_spans,
    window_state,
)
from corral.epistemic_analysis.context import (
    annotation_context,
    build_annotation_view,
    through_submission,
)
from corral.epistemic_analysis.overflow import (
    long_message_view,
    repetition_runs,
    tool_excerpt_view,
)
from corral.epistemic_analysis.pipeline import annotate_trace, can_reuse
from corral.epistemic_analysis.schema import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
)


def test_trimmed_assistant_tail_is_annotated_at_original_offset():
    text = "First hypothesis.\n" + "\n" * 10000 + "Last hypothesis."
    trace = NormalizedTrace(
        messages=[{"role": "assistant", "content": text}],
        source=SourceReference("example", "example.json", "hash"),
    )
    view = long_message_view(trace, tool_excerpt_view(trace))
    config = AnnotationConfig(model="test", tool_output_max_chars=0)
    with annotation_context(view, config, []):
        spans = candidate_spans(view.messages, 1200)
        assert [s.quote for s in spans] == ["First hypothesis.", "Last hypothesis."]
        assert spans[-1].support[0]["start"] == text.index("Last hypothesis.")

    async def nodes(messages, **kwargs):
        return [
            {
                "node_id": str(i),
                "type": "H",
                "time": s.msg_idx,
                "text": s.quote,
                "support": s.support,
            }
            for i, s in enumerate(candidate_spans(messages, 1200))
        ], []

    async def edges(**kwargs):
        return [], []

    backend = AnnotationBackend(extract_nodes=nodes, extract_edges=edges, provenance={})
    graph = asyncio.run(annotate_trace(trace, backend, config, prepared_view=view))
    assert len(graph.nodes) == 2
    assert graph.messages == trace.messages
    assert graph.nodes[-1]["support"][0]["start"] == text.index("Last hypothesis.")
    assert graph.qc["context_coverage"]["omitted_message_chars"] > 0
    assert not can_reuse(graph, trace, backend, config)

    changed = copy.deepcopy(trace)
    changed.messages[0]["content"] += "changed"
    with pytest.raises(ValueError, match="different source"):
        asyncio.run(annotate_trace(changed, backend, config, prepared_view=view))


def test_graph_priority_queue_has_no_behavioral_barrier(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "scripts/annotations")
    )
    from run_graph_annotation_batch import predecessor_status

    assert predecessor_status({"predecessor": None})["ready"]


@pytest.mark.parametrize("prepared", [False, True])
def test_graph_shortens_repeated_phrases_before_both_passes(prepared):
    phrase = "αβ try again! "
    text = "Leading claim.\n" + phrase * 900 + "\nFinal claim."
    trace = NormalizedTrace(
        messages=[
            {"role": "system", "content": phrase * 900},
            {"role": "assistant", "content": text},
        ],
        source=SourceReference("test", "test.json", "hash"),
    )
    config = AnnotationConfig(model="test")
    original = build_annotation_view(trace, config)
    seen = []

    async def nodes(messages, **kwargs):
        seen.append(messages)
        signals = window_state(messages, 1, 2)["repetition_signals"]
        assert signals["m1"][0]["repetitions"] == 900
        tail = next(
            s for s in candidate_spans(messages, 1200) if s.quote == "Final claim."
        )
        return [{"node_id": "N1", "type": "H", "time": 1, "support": tail.support}], []

    async def edges(messages, **kwargs):
        seen.append(messages)
        return [], []

    backend = AnnotationBackend(nodes, edges, {})
    graph = asyncio.run(
        annotate_trace(
            trace, backend, config, prepared_view=original if prepared else None
        )
    )
    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert seen[0][0] == trace.messages[0]
    assert seen[0][1]["content"].startswith("Leading claim.")
    assert len(seen[0][1]["content"]) < 1200
    assert original.messages[1]["content"] == graph.messages[1]["content"] == text
    assert graph.nodes[0]["support"][0]["start"] == text.index("Final claim.")
    mapping = graph.annotation_view["mappings"][1]
    assert mapping["repetition_signals"][0]["repetitions"] == 900
    assert mapping["omitted_chars"] == len(phrase) * 900 - 1024
    if not prepared:
        assert can_reuse(graph, trace, backend, config)


def test_detects_multiple_periods_and_leaves_short_repetitions_visible():
    first, second = "z\n\n", "a" * 600 + "b" * 20
    text = first * 2000 + "interlude!" + second * 9 + "unique" + "abc" * 50
    runs = repetition_runs(text)
    assert [text[r["start"] : r["end"]] for r in runs] == [first * 2000, second * 9]
    assert [r["repetitions"] for r in runs] == [2000, 9]


def test_spacing_variants_preserve_distinct_claims_and_dispatch_fields():
    loop = "".join(" " * (i % 4) + "`\n" + "\n" * (i % 7) for i in range(1000))
    fields = "\n[saved message fields]\n" + json.dumps(
        {"tool_calls": [{"function": {"name": "measure", "arguments": "x" * 6000}}]}
    )
    text = "Before.\n" + loop + "An intervening claim.\n" + loop + "After." + fields
    trace = NormalizedTrace(
        [{"role": "assistant", "content": text}],
        SourceReference("test", "test", "hash"),
    )
    config = AnnotationConfig(model="test")
    view = long_message_view(trace, build_annotation_view(trace, config))
    assert long_message_view(trace, view) is view
    assert view.messages[0]["content"].endswith("After." + fields)
    assert [s["repetitions"] for s in view.mappings[0]["repetition_signals"]] == [
        1000,
        1000,
    ]
    with annotation_context(view, config, []):
        spans = candidate_spans(view.messages, 1200)
        claim = next(s for s in spans if s.quote == "An intervening claim.")
        assert claim.support[0]["start"] == text.index(claim.quote)
        assert any(s.tool_name == "measure" for s in spans)


@pytest.mark.parametrize("precut", [False, True])
def test_prepared_graph_view_stops_at_original_submission_even_if_trimmed_away(precut):
    trace = NormalizedTrace(
        messages=[
            {"role": "assistant", "content": "A candidate explanation."},
            {
                "role": "assistant",
                "content": "Answer.\n[saved message fields]\n"
                + json.dumps({"tool_calls": [{"function": {"name": "submit_answer"}}]}),
            },
            {"role": "tool", "content": "POST_SUBMISSION_ACCEPTANCE"},
            {"role": "assistant", "content": "POST_SUBMISSION_COMMENTARY"},
        ],
        source=SourceReference("example", "example.json", "hash"),
    )
    view = tool_excerpt_view(trace)
    # A prepared excerpt may hide the dispatch fields; use the source boundary.
    view.messages[1] = {"role": "assistant", "content": "Answer."}
    view.mappings[1]["retained_ranges"] = [
        {"original_start": 0, "original_end": 7, "view_start": 0, "view_end": 7}
    ]
    if precut:
        view = through_submission(view, trace)
    seen = []

    async def nodes(messages, **kwargs):
        seen.append(copy.deepcopy(messages))
        return [], []

    async def edges(messages, **kwargs):
        seen.append(copy.deepcopy(messages))
        return [], []

    graph = asyncio.run(
        annotate_trace(
            trace,
            AnnotationBackend(extract_nodes=nodes, extract_edges=edges, provenance={}),
            AnnotationConfig(model="test", tool_output_max_chars=0),
            prepared_view=view,
        )
    )
    assert seen
    assert all(len(messages) == 2 for messages in seen)
    assert "POST_SUBMISSION" not in json.dumps(seen)
    assert len(graph.annotation_view["messages"]) == 2
    assert len(graph.messages) == 4
    assert len(view.messages) == (2 if precut else 4)
