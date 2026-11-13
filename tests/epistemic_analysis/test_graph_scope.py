"""Saved graph correction preserves provenance and removes all later support."""

import copy

from corral.epistemic_analysis.graph_scope import scope_saved_graph


def test_scope_removes_acceptance_and_later_labels_without_claiming_a_rerun():
    messages = [
        {"role": "tool", "content": "Measured 5 Hz."},
        {"role": "assistant", "content": "<action>submit_answer</action>"},
        {"role": "tool", "content": "answer accepted"},
        {"role": "assistant", "content": "That proves my answer."},
    ]
    nodes = [
        {
            "node_id": str(i),
            "time": i,
            "type": kind,
            "support": [{"msg_idx": i, "quote": messages[i]["content"]}],
        }
        for i, kind in enumerate(["E", "F", "E", "J"])
    ]
    graph = {
        "messages": messages,
        "nodes": nodes,
        "edges": [
            {
                "src": "0",
                "dst": "1",
                "relation": "informs",
                "support": [{"msg_idx": 0, "quote": "Measured 5 Hz."}],
            },
            {
                "src": "2",
                "dst": "3",
                "relation": "informs",
                "support": [{"msg_idx": 2, "quote": "answer accepted"}],
            },
        ],
        "annotation_view": {"messages": messages},
        "annotation_requests": [{"original_request_history": True}],
    }
    original = copy.deepcopy(graph)
    result, scope = scope_saved_graph(graph, "original-sha")
    assert [n["type"] for n in result["nodes"]] == ["E", "F"]
    assert len(result["edges"]) == 1
    assert scope["removed_evidence_count"] == 1
    assert scope["message_count"] == 2
    assert scope["original_model_saw_post_submission"]
    assert not scope["model_rerun"]
    assert result["annotation_view"] == graph["annotation_view"]
    assert result["annotation_requests"] == graph["annotation_requests"]
    assert result["messages"] == messages
    assert result["qc"]["review_issues"]
    assert graph == original
