"""Source-linked graph patterns and claim-specific answer provenance."""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from corral.epistemic_analysis.annotation_transcript import SAVED_FIELDS
from corral.epistemic_analysis.graph import _filter_invalid_edges, edge_pair_coverage
from corral.epistemic_analysis.ontology import relation_allowed
from corral.epistemic_analysis.pattern_review import analyze_patterns
from corral.epistemic_analysis.patterns import (
    MOTIFS,
    analyze_answer_provenance,
    detect_antipatterns_local,
    detect_subgraphs_local,
)
from corral.epistemic_analysis.schema import (
    ANSWER_PROVENANCE_VERSION,
    NormalizedTrace,
    SourceReference,
)


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
    pytest.importorskip(
        "corral.epistemic_analysis.viewer",
        reason="Local annotation viewer is not installed",
    )
    from corral.epistemic_analysis.viewer.graph_run import aggregate_pattern_reports

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


def provenance_graph_fixture(types="EJHF", relations=None):
    nodes = []
    messages = []
    for index, kind in enumerate(types):
        quote = f"Recorded {kind} claim {index}."
        messages.append(
            {"role": "tool" if kind == "E" else "assistant", "content": quote}
        )
        nodes.append(
            {
                "node_id": f"n{index}",
                "type": kind,
                "time": index,
                "text": quote,
                "support": [{"msg_idx": index, "quote": quote}],
                **({"claim_scope": "single"} if kind == "F" else {}),
                **({"evidence_origin": "tool_observation"} if kind == "E" else {}),
            }
        )
    if relations is None:
        relations = [
            (i, i + 1, "submitted_as" if types[i : i + 2] == "HF" else "informs")
            for i in range(len(types) - 1)
        ]
    edges = [
        {
            "src": f"n{src}",
            "dst": f"n{dst}",
            "relation": relation,
            "time": dst,
            "support": deepcopy(nodes[dst]["support"]),
        }
        for src, dst, relation in relations
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "messages": messages,
        "annotation_status": "complete",
        "reconstruction_status": "complete",
        "provenance": {"answer_provenance_policy": ANSWER_PROVENANCE_VERSION},
        "qc": {
            "context_coverage": {"reduced_context": False},
            "request_diagnostics": [edge_pair_coverage(nodes, 20, 5)],
            "review_issues": [],
            "warnings": [],
        },
    }


def provenance_statuses(graph):
    return {
        p["name"]: p["status"] for p in analyze_answer_provenance(graph)["patterns"]
    }


@pytest.mark.parametrize("types", ["EF", "EJF", "EHF", "EJHF", "EJJHF"])
def test_supported_paths_include_direct_reporting_and_interpreted_claims(types):
    graph = provenance_graph_fixture(types)
    result = analyze_answer_provenance(graph)
    assert result["claims"][0]["status"] == "traceable"
    assert result["claims"][0]["evidence_paths"][0]["evidence_id"] == "n0"
    assert provenance_statuses(graph) == {
        "traceable_conclusion": "matched",
        "conclusion_without_evidence_provenance": "not_found",
    }


@pytest.mark.parametrize("types", ["F", "HF", "JF"])
def test_fully_assessed_claim_without_evidence_matches_negative(types):
    assert provenance_statuses(provenance_graph_fixture(types)) == {
        "traceable_conclusion": "not_found",
        "conclusion_without_evidence_provenance": "matched",
    }


def test_two_claims_in_one_submission_can_match_both_patterns():
    graph = provenance_graph_fixture("EFF", [(0, 1, "informs")])
    graph["messages"][1]["content"] += " " + graph["messages"][2]["content"]
    graph["nodes"][2]["time"] = 1
    graph["nodes"][2]["support"][0]["msg_idx"] = 1
    result = analyze_answer_provenance(graph)
    assert [c["status"] for c in result["claims"]] == ["traceable", "untraced"]
    assert all(p["count"] == 1 and p["status"] == "matched" for p in result["patterns"])


@pytest.mark.parametrize(
    "gap",
    [
        "legacy",
        "context",
        "pairs",
        "annotation",
        "reconstruction",
        "review",
        "omission",
        "scope",
        "support",
        "origin",
    ],
)
def test_missing_coverage_and_unknown_claims_never_become_negative(gap):
    graph = provenance_graph_fixture("EHF", [(1, 2, "submitted_as")])
    if gap == "legacy":
        graph["provenance"] = {}
    elif gap == "context":
        graph["qc"]["context_coverage"]["reduced_context"] = True
    elif gap == "pairs":
        graph["qc"]["request_diagnostics"] = [edge_pair_coverage(graph["nodes"], 1, 0)]
    elif gap in {"annotation", "reconstruction"}:
        graph[gap + "_status"] = "incomplete"
    elif gap == "review":
        graph["qc"]["review_issues"] = ["Review needed"]
    elif gap == "omission":
        graph["qc"]["warnings"] = ["Omitted edge: no confident support"]
    elif gap == "scope":
        graph["nodes"][-1]["claim_scope"] = "compound"
    elif gap == "support":
        graph["nodes"][0]["support"][0]["quote"] = "Invented quote"
    elif gap == "origin":
        graph["nodes"][0].pop("evidence_origin")
    assert set(provenance_statuses(graph).values()) == {"unassessed"}


def test_positive_path_remains_visible_with_incomplete_coverage_elsewhere():
    graph = provenance_graph_fixture()
    graph["qc"]["context_coverage"]["reduced_context"] = True
    assert provenance_statuses(graph)["traceable_conclusion"] == "matched"


def test_task_supplied_evidence_can_ground_a_reported_result():
    graph = provenance_graph_fixture("EF")
    graph["nodes"][0]["evidence_origin"] = "task_given"
    graph["messages"][0]["role"] = "user"
    assert provenance_statuses(graph)["traceable_conclusion"] == "matched"


def test_one_incoming_link_cannot_ground_an_entire_compound_answer():
    graph = provenance_graph_fixture("EF")
    graph["nodes"][-1]["claim_scope"] = "compound"
    assert set(provenance_statuses(graph).values()) == {"unassessed"}


def test_a_test_result_is_not_provenance_until_it_is_used():
    graph = provenance_graph_fixture(
        "HTEF", [(0, 1, "tests"), (1, 2, "observes"), (0, 3, "submitted_as")]
    )
    assert (
        provenance_statuses(graph)["conclusion_without_evidence_provenance"]
        == "matched"
    )


def test_invalid_edge_quote_is_unassessed_and_informs_cycles_terminate():
    graph = provenance_graph_fixture("EJF")
    graph["edges"][0]["support"][0]["quote"] = "Invented evidence use"
    assert set(provenance_statuses(graph).values()) == {"unassessed"}
    cycle = provenance_graph_fixture(
        "JJF", [(0, 1, "informs"), (1, 0, "informs"), (1, 2, "informs")]
    )
    assert (
        provenance_statuses(cycle)["conclusion_without_evidence_provenance"]
        == "matched"
    )


@pytest.mark.parametrize(
    "relation", ["contradicts", "tests", "updates_to", "competes_with"]
)
def test_arbitrary_reachability_is_not_evidence_provenance(relation):
    graph = provenance_graph_fixture("EHF", [(0, 1, relation), (1, 2, "submitted_as")])
    assert provenance_statuses(graph)["traceable_conclusion"] != "matched"


def test_post_submission_feedback_cannot_ground_an_answer():
    graph = provenance_graph_fixture("FE", [(1, 0, "informs")])
    assert provenance_statuses(graph)["traceable_conclusion"] == "not_found"
    assert _filter_invalid_edges(graph["edges"], graph["nodes"])[0] == []


def test_no_submission_and_boilerplate_are_not_failures():
    assert set(provenance_statuses(provenance_graph_fixture("EJ")).values()) == {
        "unassessed"
    }
    graph = provenance_graph_fixture("F")
    graph["nodes"][0]["claim_scope"] = "nonclaim"
    assert set(provenance_statuses(graph).values()) == {"not_found"}


def test_new_graph_reports_preserve_existing_motif_catalog_and_counts():
    graph = provenance_graph_fixture()
    before = {
        **detect_subgraphs_local(graph["nodes"], graph["edges"]),
        **detect_antipatterns_local(graph["nodes"], graph["edges"]),
    }
    result = analyze_patterns(graph, "graph-hash")
    assert len(MOTIFS) == 18
    assert len(result["patterns"]) == 20
    assert {
        p["name"]: p["count"] for p in result["patterns"] if p["name"] in before
    } == before
    assert result["answer_provenance"]["claims"][0]["status"] == "traceable"


@pytest.mark.parametrize(
    ("src", "rel", "allowed"),
    [
        ("E", "informs", True),
        ("J", "informs", True),
        ("H", "submitted_as", True),
        ("T", "informs", False),
        ("H", "informs", False),
        ("C", "submitted_as", False),
    ],
)
def test_final_edge_ontology_is_typed(src, rel, allowed):
    assert relation_allowed(rel, {"type": src}, {"type": "F"}) is allowed
