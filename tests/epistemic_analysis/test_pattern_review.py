"""Repeated tests remain source-linked screening candidates, not p-hacking verdicts."""

import json
from copy import deepcopy
from dataclasses import asdict

from corral.epistemic_analysis.annotation_transcript import SAVED_FIELDS
from corral.epistemic_analysis.pattern_review import analyze_patterns
from corral.epistemic_analysis.schema import NormalizedTrace, SourceReference
from corral.epistemic_analysis.viewer.graph_run import aggregate_pattern_reports


def graph_fixture():
    messages = [{"role": "assistant", "content": "The treatment changes the outcome."}]
    sources = [{"agent_run_id": "actor"}]
    nodes = [
        {
            "node_id": "H",
            "type": "H",
            "time": 0,
            "text": messages[0]["content"],
            "support": [],
        }
    ]
    edges = []
    for n, value in enumerate((0.2, 0.01)):
        index = len(messages)
        arguments = {"threshold": n}
        quote = (
            "<action>analyze</action>\n<action_input>"
            + json.dumps(arguments)
            + "</action_input>"
        )
        call = {
            "id": f"call-{n}",
            "function": {"name": "analyze", "arguments": json.dumps(arguments)},
        }
        content = (
            quote
            + SAVED_FIELDS
            + json.dumps({"tool_calls": [call], "agent_run_id": "actor"})
        )
        messages.extend(
            [
                {"role": "assistant", "content": content},
                {"role": "tool", "content": f"p-value = {value}"},
            ]
        )
        sources.extend(
            [
                {"agent_run_id": "actor"},
                {
                    "agent_run_id": "actor",
                    "tool_call_id": call["id"],
                    "name": "analyze",
                },
            ]
        )
        nodes.extend(
            [
                {
                    "node_id": f"T{n}",
                    "type": "T",
                    "time": index,
                    "text": quote,
                    "support": [
                        {
                            "msg_idx": index,
                            "start": 0,
                            "end": len(quote),
                            "quote": quote,
                        }
                    ],
                },
                {
                    "node_id": f"E{n}",
                    "type": "E",
                    "time": index + 1,
                    "text": "result",
                    "support": [],
                },
            ]
        )
        edges.extend(
            [
                {"src": "H", "dst": f"T{n}", "relation": "tests"},
                {"src": f"T{n}", "dst": f"E{n}", "relation": "observes"},
            ]
        )
    trace = NormalizedTrace(
        messages, SourceReference("test", "test.json", "hash"), message_sources=sources
    )
    return {**asdict(trace), "nodes": nodes, "edges": edges}


def test_repeated_paired_calls_show_changed_arguments_without_p_hacking_verdict():
    graph = graph_fixture()
    original = deepcopy(graph)
    analysis = analyze_patterns(graph, "signature")
    (candidate,) = analysis["repeated_tests"]
    assert candidate["call_count"] == 2
    assert candidate["changed_argument_keys"] == ["threshold"]
    assert candidate["statistical_markers"]
    assert candidate["p_hacking_status"] == "unassessed"
    assert [call["result_msg_idx"] for call in candidate["calls"]] == [2, 4]
    assert graph == original


def test_duplicate_test_labels_on_one_call_do_not_make_repetition():
    graph = graph_fixture()
    t0 = next(n for n in graph["nodes"] if n["node_id"] == "T0")
    graph["nodes"] = [
        n if n["node_id"] != "T1" else {**t0, "node_id": "T1"} for n in graph["nodes"]
    ]
    assert not analyze_patterns(graph, "signature")["repeated_tests"]


def test_unexecuted_plan_and_unpaired_output_are_not_repeated_tests():
    graph = graph_fixture()
    graph["message_sources"][-1]["tool_call_id"] = "another-call"
    assert not analyze_patterns(graph, "signature")["repeated_tests"]
    graph["messages"][-2]["content"] = graph["messages"][-2]["content"].split(
        SAVED_FIELDS
    )[0]
    assert not analyze_patterns(graph, "signature")["repeated_tests"]


def test_tool_result_outside_graph_scope_cannot_complete_a_repeated_test():
    graph = graph_fixture()
    graph["provenance"] = {"graph_scope": {"message_count": 4}}
    analysis = analyze_patterns(graph, "signature")
    assert not analysis["repeated_tests"]
    assert analysis["call_coverage"]["paired_calls"] == 1


def test_same_tool_under_different_hypotheses_does_not_match():
    graph = graph_fixture()
    graph["nodes"].append({**graph["nodes"][0], "node_id": "other"})
    graph["edges"][2]["src"] = "other"
    assert not analyze_patterns(graph, "signature")["repeated_tests"]


def test_hypothesis_after_tests_cannot_be_a_fixed_prior_hypothesis():
    graph = graph_fixture()
    graph["nodes"][0]["time"] = 3
    assert not analyze_patterns(graph, "signature")["repeated_tests"]


def test_feedback_requires_a_distinct_later_call_and_intervening_judgment():
    graph = graph_fixture()
    judgment = {
        "node_id": "J",
        "type": "J",
        "time": 2,
        "text": "Interpret result",
        "support": [],
    }
    graph["nodes"].append(judgment)
    graph["edges"].extend(
        [
            {"src": "E0", "dst": "J", "relation": "informs"},
            {"src": "J", "dst": "T1", "relation": "tests"},
        ]
    )
    assert analyze_patterns(graph, "signature")["repeated_tests"][0][
        "feedback_chains"
    ] == [["T0", "E0", "J", "T1"]]
    judgment["time"] = 4
    assert not analyze_patterns(graph, "signature")["repeated_tests"][0][
        "feedback_chains"
    ]


def test_empty_graph_patterns_remain_unassessed():
    result = analyze_patterns({"nodes": [], "edges": [], "messages": []}, "signature")
    assert all(p["status"] == "unassessed" for p in result["patterns"])


def test_parent_patterns_or_units_without_inventing_cross_unit_patterns():
    analysis = analyze_patterns(graph_fixture(), "signature")
    pattern = analysis["patterns"][0]
    pattern["status"], pattern["count"] = "matched", 1
    parent = aggregate_pattern_reports(
        [
            {"id": "main", "label": "Main", "analysis": analysis},
            {"id": "node", "label": "Node", "analysis": None},
        ]
    )
    assert parent["units_total"] == 2
    assert parent["units_assessed"] == 1
    assert parent["patterns"][0]["status"] == "matched"
    assert parent["patterns"][0]["units"] == [
        {"id": "main", "label": "Main", "count": 1}
    ]
    pattern["status"], pattern["count"] = "not_found", 0
    parent = aggregate_pattern_reports(
        [
            {"id": "main", "label": "Main", "analysis": analysis},
            {"id": "node", "label": "Node", "analysis": None},
        ]
    )
    assert parent["patterns"][0]["status"] == "unassessed"
