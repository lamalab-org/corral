"""Read-only graph motifs and inspectable repeated-test candidates.

These are structural measurements, not behavioral-check predictions. In
particular, repeated calls never produce a positive p-hacking label.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from .decision_applicability import ValidationPolicy, extract_facts
from .epistemic_prompts import ANTIPATTERN_DESCRIPTIONS, SUBGRAPH_DESCRIPTIONS
from .patterns import (
    analyze_answer_provenance,
    detect_antipatterns_local,
    detect_subgraphs_local,
)
from .schema import content_hash
from .sources.inputs import CheckInput

PATTERN_REVIEW_VERSION = "graph-motifs-and-answer-provenance-v3"
STATISTICAL_MARKER = re.compile(
    r"\bp[-_ ]?values?\b|\bp\s*[<=>]\s*0\.\d+|\bstatistically significant\b",
    re.IGNORECASE,
)
GENERIC_TOOLS = {"terminal", "execute_python_script", "bash", "shell", "exec_command"}


def _short(value, limit=320):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def _matches_call(node, event):
    """Link a retained test to a recorded dispatch, never just a proposed plan."""
    if node["time"] != event["call_index"]:
        return False
    name = re.escape(event["name"])
    text = " ".join(s.get("quote", "") for s in node.get("support", []))
    return bool(
        re.search(r"<action>\s*" + name + r"\s*</action>", text)
        or re.search(r'"(?:name|tool_name)"\s*:\s*"' + name + r'"', text)
        or re.search(r"<\|tool_call_begin\|>(?:functions\.)?" + name + r"[:<]", text)
    )


def repeated_test_candidates(graph):
    if not graph.get("messages"):
        return [], {"paired_calls": 0, "matched_test_nodes": 0}
    end = (
        graph.get("provenance", {})
        .get("graph_scope", {})
        .get(
            "message_count",
            len(graph.get("annotation_view", {}).get("messages", graph["messages"])),
        )
    )
    scoped = {**graph, "messages": graph["messages"][:end]}
    if "message_sources" in graph:
        scoped["message_sources"] = graph["message_sources"][:end]
    events = extract_facts(CheckInput.from_document(scoped), ValidationPolicy())[
        "events"
    ]
    events = [
        e for e in events if "result_index" in e and isinstance(e["arguments"], dict)
    ]
    by_message = defaultdict(list)
    for event in events:
        by_message[event["call_index"]].append(event)
    nodes = {n["node_id"]: n for n in graph["nodes"]}
    calls = {}
    for node in nodes.values():
        if node["type"] != "T":
            continue
        matches = [e for e in by_message[node["time"]] if _matches_call(node, e)]
        if len(matches) == 1:
            calls[node["node_id"]] = matches[0]
    tests = defaultdict(set)
    outgoing = defaultdict(list)
    for edge in graph["edges"]:
        outgoing[edge["src"]].append(edge)
        if edge["relation"] != "tests":
            continue
        first, second = nodes.get(edge["src"]), nodes.get(edge["dst"])
        if first and second and first["type"] == "T" and second["type"] == "H":
            first, second = second, first
        if (
            first
            and second
            and first["type"] == "H"
            and second["type"] == "T"
            and second["node_id"] in calls
            and first["time"] <= second["time"]
        ):
            tests[first["node_id"]].add(second["node_id"])
    groups = []
    for hypothesis, test_ids in tests.items():
        by_tool = defaultdict(dict)
        for identity in sorted(test_ids):
            event = calls[identity]
            # Multiple labels on one dispatch must not become repeated tests.
            key = (event["actor"], event["branch"], event["call_id"])
            tool = (event["name"], event["actor"], event["branch"])
            by_tool[tool].setdefault(key, {"event": event, "test_ids": []})[
                "test_ids"
            ].append(identity)
        for (tool, _actor, _branch), unique in by_tool.items():
            if len(unique) < 2:
                continue
            ordered = sorted(
                unique.values(), key=lambda item: item["event"]["call_order"]
            )
            arguments = [item["event"]["arguments"] for item in ordered]
            keys = sorted({key for value in arguments for key in value})
            changed = [
                key
                for key in keys
                if len(
                    {
                        content_hash({"present": key in value, "value": value.get(key)})
                        for value in arguments
                    }
                )
                > 1
            ]
            first, last = (
                ordered[0]["event"]["call_index"],
                ordered[-1]["event"]["call_index"],
            )
            ids = {identity for item in ordered for identity in item["test_ids"]}
            feedback = []
            for source in ids:
                for observed in outgoing[source]:
                    if observed["relation"] != "observes":
                        continue
                    for interpreted in outgoing[observed["dst"]]:
                        if (
                            interpreted["relation"] != "informs"
                            or nodes[interpreted["dst"]]["type"] != "J"
                        ):
                            continue
                        for redesigned in outgoing[interpreted["dst"]]:
                            target = redesigned["dst"]
                            if (
                                redesigned["relation"] == "tests"
                                and target in ids
                                and calls[target]["call_order"]
                                > calls[source]["call_order"]
                                and nodes[observed["dst"]]["time"]
                                == calls[source]["result_index"]
                                and calls[source]["result_index"]
                                <= nodes[interpreted["dst"]]["time"]
                                <= calls[target]["call_index"]
                            ):
                                feedback.append(
                                    [
                                        source,
                                        observed["dst"],
                                        interpreted["dst"],
                                        target,
                                    ]
                                )
            markers = []
            for index in range(
                first, max(item["event"]["result_index"] for item in ordered) + 1
            ):
                message = graph["messages"][index]
                if message["role"] not in {"assistant", "tool"}:
                    continue
                for match in STATISTICAL_MARKER.finditer(message["content"]):
                    start, end = (
                        max(0, match.start() - 80),
                        min(len(message["content"]), match.end() + 100),
                    )
                    markers.append(
                        {
                            "msg_idx": index,
                            "start": start,
                            "end": end,
                            "quote": message["content"][start:end],
                        }
                    )
            groups.append(
                {
                    "id": content_hash(
                        {"hypothesis": hypothesis, "tool": tool, "calls": list(unique)}
                    )[:20],
                    "hypothesis_id": hypothesis,
                    "hypothesis": nodes[hypothesis]["text"],
                    "tool": tool,
                    "generic_executor": tool in GENERIC_TOOLS,
                    "test_ids": sorted(
                        ids, key=lambda identity: (nodes[identity]["time"], identity)
                    ),
                    "call_count": len(ordered),
                    "argument_variants": len(
                        {content_hash(value) for value in arguments}
                    ),
                    "changed_argument_keys": changed,
                    "candidate_representation_changed": bool(
                        set(changed)
                        & {"smiles", "molecule_smiles", "hypothesis", "candidate"}
                    ),
                    "hypothesis_revision_recorded": any(
                        e["relation"] == "updates_to"
                        and first <= nodes[e["dst"]]["time"] <= last
                        for e in outgoing[hypothesis]
                    ),
                    "feedback_chains": feedback,
                    "statistical_markers": markers,
                    "p_hacking_status": "unassessed",
                    "calls": [
                        {
                            "test_ids": item["test_ids"],
                            "call_id": item["event"]["call_id"],
                            "msg_idx": item["event"]["call_index"],
                            "result_msg_idx": item["event"]["result_index"],
                            "arguments": {
                                key: _short(value)
                                for key, value in item["event"]["arguments"].items()
                            },
                            "argument_hash": content_hash(item["event"]["arguments"]),
                        }
                        for item in ordered
                    ],
                }
            )
    return groups, {"paired_calls": len(events), "matched_test_nodes": len(calls)}


def analyze_patterns(graph, graph_sha256):
    nodes, edges = graph["nodes"], graph["edges"]
    items = []
    for category, descriptions, detector in (
        ("reasoning", SUBGRAPH_DESCRIPTIONS, detect_subgraphs_local),
        ("review_signal", ANTIPATTERN_DESCRIPTIONS, detect_antipatterns_local),
    ):
        counts = detector(nodes, edges) if nodes else {}
        items.extend(
            {
                "name": name,
                "category": category,
                "description": description,
                "count": counts.get(name, 0),
                "status": "unassessed"
                if not nodes
                else "matched"
                if counts[name]
                else "not_found",
            }
            for name, description in descriptions.items()
        )
    repeated, coverage = repeated_test_candidates(graph)
    provenance = analyze_answer_provenance(graph)
    items.extend(provenance.pop("patterns"))
    return {
        "version": PATTERN_REVIEW_VERSION,
        "graph_sha256": graph_sha256,
        "scope": "independent_agent_unit"
        if graph.get("provenance", {}).get("preprocessing", {}).get("scientist_unit")
        else "trace",
        "basis": "Deterministic matches in saved graph annotations; no additional model call.",
        "qualification": "Graph matches are structural signals. Missing links may reflect annotation gaps; repeated tests do not establish p-hacking, and multiple tests need not be independent.",
        "reduced_context": graph.get("qc", {})
        .get("context_coverage", {})
        .get("reduced_context", False),
        "patterns": items,
        "answer_provenance": provenance,
        "repeated_tests": repeated,
        "call_coverage": coverage,
        "reviews": [],
        "graph_scope": graph.get("provenance", {}).get("graph_scope"),
    }
