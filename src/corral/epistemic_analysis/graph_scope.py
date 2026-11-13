"""Apply an explicit scope to saved graphs without claiming a model rerun."""

from __future__ import annotations

import copy

from .context import SUBMISSION_SCOPE_VERSION, submission_end
from .pipeline import message_coverage
from .sources.inputs import CheckInput


def scope_saved_graph(graph, source_sha256):
    """Keep the source and original request history; remove out-of-scope labels."""
    end = submission_end(CheckInput.from_document(graph).trace)
    if end == len(graph["messages"]):
        return copy.deepcopy(graph), None
    result = copy.deepcopy(graph)

    def retained(item):
        return all(0 <= s["msg_idx"] < end for s in item.get("support", []))

    nodes = [n for n in result["nodes"] if 0 <= n["time"] < end and retained(n)]
    identities = {n["node_id"] for n in nodes}
    for node in nodes:
        if "shared_support_with" in node:
            node["shared_support_with"] = [
                i for i in node["shared_support_with"] if i in identities
            ]
    edges = [
        e
        for e in result["edges"]
        if e["src"] in identities and e["dst"] in identities and retained(e)
    ]
    scope = {
        "policy": SUBMISSION_SCOPE_VERSION,
        "application": "filtered_saved_labels",
        "source_graph_sha256": source_sha256,
        "message_count": end,
        "submission_msg_idx": end - 1,
        "excluded_message_count": len(graph["messages"]) - end,
        "removed_node_count": len(graph["nodes"]) - len(nodes),
        "removed_evidence_count": sum(
            n["type"] == "E" for n in graph["nodes"] if n["node_id"] not in identities
        ),
        "removed_edge_count": len(graph["edges"]) - len(edges),
        "model_rerun": False,
        "original_model_saw_post_submission": len(
            graph.get("annotation_view", {}).get("messages", graph["messages"])
        )
        > end,
    }
    result["nodes"], result["edges"] = nodes, edges
    result.setdefault("provenance", {})["graph_scope"] = scope
    note = (
        "Graph labels and patterns stop at the submit_answer call. Acceptance feedback "
        "and later messages are excluded. Existing labels were filtered, not rerun; "
        "earlier labels were originally generated with post-submission context."
    )
    qc = result.setdefault("qc", {})
    qc.setdefault("warnings", []).append(note)
    qc.setdefault("review_issues", []).append(note)
    qc["message_coverage"] = message_coverage(graph["messages"][:end], nodes)
    qc["uncovered_messages"] = qc["message_coverage"]["unlabelled_messages"]
    result["annotation_status"] = "incomplete"
    return result, scope
