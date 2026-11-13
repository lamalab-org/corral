"""Graph preprocessing keeps original source coordinates and excludes cut markers."""

import asyncio
import copy
import json

import pytest

from corral.epistemic_analysis.annotators.base import AnnotationBackend
from corral.epistemic_analysis.annotators.jev import candidate_spans
from corral.epistemic_analysis.context import annotation_context, through_submission
from corral.epistemic_analysis.overflow import long_message_view, tool_excerpt_view
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

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_graph_annotation_batch import predecessor_status

    assert predecessor_status({"predecessor": None})["ready"]


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
