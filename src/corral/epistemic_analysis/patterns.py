"""Patterns for epistemic trace analysis."""

from __future__ import annotations

from typing import Any

from corral.epistemic_analysis.graph import (
    build_index,
    earliest_time_of_type,
    node_time,
)

SUBGRAPH_DESCRIPTIONS: dict[str, str] = {
    "refutation_driven_belief_revision": (
        "Evidence triggers a belief update to a new hypothesis "
        "[H -tests-> T -observes-> E -informs-> J, H -updates_to-> H2]."
    ),
    "fixed_hypothesis_test_tuning": (
        "Hypothesis is held fixed while tests are iteratively adjusted "
        "[H -tests-> T -observes-> E -informs-> J -tests-> T2]."
    ),
    "explore_then_test_transition": (
        "Exploration precedes hypothesis formation, which then drives testing "
        "[T -observes-> E ... H ... H -tests-> T]."
    ),
    "hypothesis_reranking": (
        "Competing hypotheses are compared as new evidence arrives "
        "[H1 -competes_with- H2, both tested]."
    ),
    "evidence_led_hypothesis_generation": (
        "Evidence is observed first; a hypothesis is formed afterward "
        "[E -informs-> J ... H ... H -tests-> T]."
    ),
    "convergent_multi_test_evidence": (
        "One hypothesis is evaluated via multiple independent tests "
        "[H -tests-> T1/T2/T3..., each -> E]."
    ),
    "evidence_guided_test_redesign": (
        "A judgment motivates a new test, which then produces new evidence "
        "[J -tests-> T -observes-> E]."
    ),
}


SUBGRAPH_NAMES = list(SUBGRAPH_DESCRIPTIONS.keys())


(
    SG_REFUTATION_DRIVEN_BELIEF_REVISION,
    SG_FIXED_HYPOTHESIS_TEST_TUNING,
    SG_EXPLORE_THEN_TEST_TRANSITION,
    SG_HYPOTHESIS_RERANKING,
    SG_EVIDENCE_LED_HYPOTHESIS_GENERATION,
    SG_CONVERGENT_MULTI_TEST_EVIDENCE,
    SG_EVIDENCE_GUIDED_TEST_REDESIGN,
) = SUBGRAPH_NAMES


ANTIPATTERN_DESCRIPTIONS: dict[str, str] = {
    "untested_claim": ("Hypothesis never linked to a test [H with no tests]."),
    "evidence_non_uptake": (
        "Evidence collected but never used [E with no informs to J or H]."
    ),
    "unsupported_judgment": (
        "Judgment made without supporting evidence [J with no E via informs]."
    ),
    "stalled_revision": (
        "Revised hypothesis never tested "
        "[H target of updates_to with no outgoing tests]."
    ),
    "contradiction_without_repair": (
        "Contradiction unresolved by any update or alternative "
        "[E -contradicts-> H, no updates_to/competes_with]."
    ),
    "premature_commitment": (
        "Commitment (explicit or inferred) to a hypothesis without testing it first "
        "[J -informs-> C, J -informs-> H, H with no tests]."
    ),
    "uninformative_test": ("Test produces no observed evidence [T with no E]."),
    "fixed_belief_trace": (
        "No hypothesis revision in the entire trace [No updates_to edges in trace]."
    ),
    "disconnected_evidence": ("Evidence node with no edges [Isolated E]."),
    "one_sided_confirmation": (
        "Commitment (explicit or inferred) without contradicting evidence "
        "[J -informs-> C, J -informs-> H, H has support but no contradicts]."
    ),
    "precommitted_test_plan": (
        "Commitment (explicit or inferred) before evidence collection "
        "[C before E; then H -tests-> T]."
    ),
}


ANTIPATTERN_NAMES = list(ANTIPATTERN_DESCRIPTIONS.keys())


(
    AP_UNTESTED_CLAIM,
    AP_EVIDENCE_NON_UPTAKE,
    AP_UNSUPPORTED_JUDGMENT,
    AP_STALLED_REVISION,
    AP_CONTRADICTION_WITHOUT_REPAIR,
    AP_PREMATURE_COMMITMENT,
    AP_UNINFORMATIVE_TEST,
    AP_FIXED_BELIEF_TRACE,
    AP_DISCONNECTED_EVIDENCE,
    AP_ONE_SIDED_CONFIRMATION,
    AP_PRECOMMITTED_TEST_PLAN,
) = ANTIPATTERN_NAMES


ANTIPATTERN_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        AP_UNTESTED_CLAIM,
        AP_CONTRADICTION_WITHOUT_REPAIR,
        AP_ONE_SIDED_CONFIRMATION,
    ],
    "evidence_handling": [
        AP_EVIDENCE_NON_UPTAKE,
        AP_DISCONNECTED_EVIDENCE,
        AP_UNSUPPORTED_JUDGMENT,
        AP_UNINFORMATIVE_TEST,
    ],
    "experimental_strategy": [
        AP_STALLED_REVISION,
        AP_FIXED_BELIEF_TRACE,
        AP_PREMATURE_COMMITMENT,
        AP_PRECOMMITTED_TEST_PLAN,
    ],
}


ANTIPATTERN_FAMILY_NAMES = list(ANTIPATTERN_FAMILIES.keys())


SUBGRAPH_FAMILIES: dict[str, list[str]] = {
    "hypothesis_generation": [
        SG_REFUTATION_DRIVEN_BELIEF_REVISION,
        SG_HYPOTHESIS_RERANKING,
        SG_EVIDENCE_LED_HYPOTHESIS_GENERATION,
    ],
    "evidence_handling": [
        SG_CONVERGENT_MULTI_TEST_EVIDENCE,
        SG_EXPLORE_THEN_TEST_TRANSITION,
    ],
    "experimental_strategy": [
        SG_FIXED_HYPOTHESIS_TEST_TUNING,
        SG_EVIDENCE_GUIDED_TEST_REDESIGN,
    ],
}


SUBGRAPH_FAMILY_NAMES = list(SUBGRAPH_FAMILIES.keys())


def _nodes_of_type(ntype: str, node_type_map: dict[str, str]) -> list[str]:
    return [nid for nid, t in node_type_map.items() if t == ntype]


def _has_edge(
    src: str, dst: str, relation: str, out_edges: dict[str, list[dict[str, Any]]]
) -> bool:
    for e in out_edges.get(src, []):
        if e.get("relation") == relation and e.get("dst") == dst:
            return True
    return False


def _has_informs_link(
    a: str, b: str, out_edges: dict[str, list[dict[str, Any]]]
) -> bool:
    return _has_edge(a, b, "informs", out_edges) or _has_edge(
        b, a, "informs", out_edges
    )


def _neighbours(
    src: str,
    relation: str,
    dst_type: str,
    out_edges: dict[str, list[dict[str, Any]]],
    node_type_map: dict[str, str],
) -> list[str]:
    return [
        e["dst"]
        for e in out_edges.get(src, [])
        if e.get("relation") == relation and node_type_map.get(e.get("dst")) == dst_type
    ]


def _match_popperian(node_type_map, _node_by_id, out_edges):
    count = 0
    for h1 in _nodes_of_type("H", node_type_map):
        matched = False
        for t in _neighbours(h1, "tests", "T", out_edges, node_type_map):
            for ev in _neighbours(t, "observes", "E", out_edges, node_type_map):
                for j in _nodes_of_type("J", node_type_map):
                    if not _has_informs_link(ev, j, out_edges):
                        continue
                    for h2 in _neighbours(
                        h1, "updates_to", "H", out_edges, node_type_map
                    ):
                        if h2 != h1:
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                break
        if matched:
            count += 1
    return count


def _match_ml_make_it_work(node_type_map, _node_by_id, out_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        matched = False
        for t1 in _neighbours(h, "tests", "T", out_edges, node_type_map):
            for ev in _neighbours(t1, "observes", "E", out_edges, node_type_map):
                for j in _nodes_of_type("J", node_type_map):
                    if not _has_informs_link(ev, j, out_edges):
                        continue
                    j_to_Ts = _neighbours(j, "tests", "T", out_edges, node_type_map)
                    if not j_to_Ts:
                        continue
                    h_to_Hs = _neighbours(
                        h, "updates_to", "H", out_edges, node_type_map
                    )
                    if h_to_Hs:
                        continue
                    matched = True
                    break
                if matched:
                    break
            if matched:
                break
        if matched:
            count += 1
    return count


def _match_exploratory_to_confirmatory(node_type_map, node_by_id, out_edges):
    count = 0
    for t0 in _nodes_of_type("T", node_type_map):
        t0_time = node_time(node_by_id[t0])
        evs = _neighbours(t0, "observes", "E", out_edges, node_type_map)
        if not evs:
            continue
        for h1 in _nodes_of_type("H", node_type_map):
            if node_time(node_by_id[h1]) <= t0_time:
                continue
            if _neighbours(h1, "tests", "T", out_edges, node_type_map):
                count += 1
                break
    return count


def _match_bayesian(node_type_map, _node_by_id, out_edges):
    count = 0
    Hs = _nodes_of_type("H", node_type_map)
    for i, h1 in enumerate(Hs):
        for h2 in Hs[i + 1 :]:
            competes = _has_edge(h1, h2, "competes_with", out_edges) or _has_edge(
                h2, h1, "competes_with", out_edges
            )
            if not competes:
                continue
            if _neighbours(h1, "tests", "T", out_edges, node_type_map) and _neighbours(
                h2, "tests", "T", out_edges, node_type_map
            ):
                count += 1
    return count


def _match_abductive(node_type_map, node_by_id, out_edges):
    count = 0
    for e0 in _nodes_of_type("E", node_type_map):
        e0_time = node_time(node_by_id[e0])
        matched = False
        for j0 in _nodes_of_type("J", node_type_map):
            if not _has_informs_link(e0, j0, out_edges):
                continue
            for h1 in _nodes_of_type("H", node_type_map):
                if node_time(node_by_id[h1]) <= e0_time:
                    continue
                if _neighbours(h1, "tests", "T", out_edges, node_type_map):
                    matched = True
                    break
            if matched:
                break
        if matched:
            count += 1
    return count


def _match_triangulation(node_type_map, _node_by_id, out_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        test_targets = _neighbours(h, "tests", "T", out_edges, node_type_map)
        test_with_ev = [
            t
            for t in test_targets
            if _neighbours(t, "observes", "E", out_edges, node_type_map)
        ]
        if len(set(test_with_ev)) >= 3:
            count += 1
    return count


def _match_preregistered(node_type_map, node_by_id, out_edges, _in_edges=None):
    count = 0
    Cs = _nodes_of_type("C", node_type_map)
    Es = _nodes_of_type("E", node_type_map)
    for h in _nodes_of_type("H", node_type_map):
        matched = False
        for t in _neighbours(h, "tests", "T", out_edges, node_type_map):
            t_time = node_time(node_by_id[t])
            for c in Cs:
                c_time = node_time(node_by_id[c])
                if c_time > t_time:
                    continue
                for ev in Es:
                    if node_time(node_by_id[ev]) > c_time:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if matched:
            count += 1
    return count


def _match_active_learning(node_type_map, _node_by_id, out_edges):
    count = 0
    for j in _nodes_of_type("J", node_type_map):
        for t in _neighbours(j, "tests", "T", out_edges, node_type_map):
            if _neighbours(t, "observes", "E", out_edges, node_type_map):
                count += 1
                break
    return count


_SUBGRAPH_MATCHERS = {
    SG_REFUTATION_DRIVEN_BELIEF_REVISION: _match_popperian,
    SG_FIXED_HYPOTHESIS_TEST_TUNING: _match_ml_make_it_work,
    SG_EXPLORE_THEN_TEST_TRANSITION: _match_exploratory_to_confirmatory,
    SG_HYPOTHESIS_RERANKING: _match_bayesian,
    SG_EVIDENCE_LED_HYPOTHESIS_GENERATION: _match_abductive,
    SG_CONVERGENT_MULTI_TEST_EVIDENCE: _match_triangulation,
    SG_EVIDENCE_GUIDED_TEST_REDESIGN: _match_active_learning,
}


def detect_subgraphs_local(nodes: list, edges: list) -> dict[str, int]:
    node_by_id, out_edges, _in = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    return {
        name: matcher(node_type_map, node_by_id, out_edges)
        for name, matcher in _SUBGRAPH_MATCHERS.items()
    }


def _typed_edge_exists(src_type, dst_type, relation, node_by_id, out_edges):
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    for nid, ntype in node_type_map.items():
        if ntype != src_type:
            continue
        for e in out_edges.get(nid, []):
            if (
                e.get("relation") == relation
                and node_type_map.get(e.get("dst")) == dst_type
            ):
                return True
    return False


def _fan_out_global(src_type, dst_type, relation, node_by_id, out_edges):
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    best = 0
    for nid, ntype in node_type_map.items():
        if ntype != src_type:
            continue
        targets = {
            e.get("dst")
            for e in out_edges.get(nid, [])
            if e.get("relation") == relation
            and node_type_map.get(e.get("dst")) == dst_type
        }
        best = max(best, len(targets))
    return best


def detect_subgraphs_global(nodes: list, edges: list) -> dict[str, int]:
    node_by_id, out_edges, _in_edges = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}

    n_H = sum(1 for t in node_type_map.values() if t == "H")
    n_T = sum(1 for t in node_type_map.values() if t == "T")

    t_first_H = earliest_time_of_type(nodes, "H")
    t_first_T = earliest_time_of_type(nodes, "T")
    t_first_E = earliest_time_of_type(nodes, "E")

    has_HT_tests = _typed_edge_exists("H", "T", "tests", node_by_id, out_edges)
    has_TE_observes = _typed_edge_exists("T", "E", "observes", node_by_id, out_edges)
    has_HH_updates = _typed_edge_exists("H", "H", "updates_to", node_by_id, out_edges)
    has_JT_tests = _typed_edge_exists("J", "T", "tests", node_by_id, out_edges)
    has_HH_competes = _typed_edge_exists(
        "H", "H", "competes_with", node_by_id, out_edges
    )
    fan_out_H_T = _fan_out_global("H", "T", "tests", node_by_id, out_edges)
    has_EJ_informs = _typed_edge_exists(
        "E", "J", "informs", node_by_id, out_edges
    ) or _typed_edge_exists("J", "E", "informs", node_by_id, out_edges)

    results: dict[str, int] = {}
    results[SG_REFUTATION_DRIVEN_BELIEF_REVISION] = int(
        has_HT_tests and has_TE_observes and has_HH_updates and n_H >= 2
    )
    results[SG_FIXED_HYPOTHESIS_TEST_TUNING] = int(
        n_H <= 1
        and n_T >= 2
        and has_HT_tests
        and has_TE_observes
        and has_EJ_informs
        and not has_HH_updates
    )
    results[SG_EXPLORE_THEN_TEST_TRANSITION] = int(
        t_first_T is not None
        and t_first_H is not None
        and t_first_T < t_first_H
        and has_HT_tests
    )
    results[SG_HYPOTHESIS_RERANKING] = int(n_H >= 2 and has_HH_competes)
    results[SG_EVIDENCE_LED_HYPOTHESIS_GENERATION] = int(
        t_first_E is not None and t_first_H is not None and t_first_E < t_first_H
    )
    results[SG_CONVERGENT_MULTI_TEST_EVIDENCE] = int(fan_out_H_T >= 3)
    results[SG_EVIDENCE_GUIDED_TEST_REDESIGN] = int(has_JT_tests and has_TE_observes)
    return results


def _ap_untested_hypothesis(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if not tested:
            count += 1
    return count


def _ap_evidence_ignored(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for ev in _nodes_of_type("E", node_type_map):
        used = any(
            e.get("relation") == "informs"
            and node_type_map.get(e.get("dst")) in {"J", "H"}
            for e in out_edges.get(ev, [])
        )
        if not used:
            used = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) in {"J", "H"}
                for e in in_edges.get(ev, [])
            )
        if not used:
            count += 1
    return count


def _ap_judgment_without_evidence(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for j in _nodes_of_type("J", node_type_map):
        has_e = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(j, [])
        )
        if not has_e:
            has_e = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(j, [])
            )
        if not has_e:
            count += 1
    return count


def _ap_dead_end_update(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for h in _nodes_of_type("H", node_type_map):
        is_revised = any(e.get("relation") == "updates_to" for e in in_edges.get(h, []))
        if not is_revised:
            continue
        has_test = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not has_test:
            count += 1
    return count


def _ap_unresolved_contradiction(node_type_map, node_by_id, out_edges, in_edges):
    count = 0
    for nid in list(node_type_map):
        for e in out_edges.get(nid, []):
            if e.get("relation") != "contradicts":
                continue
            dst = e.get("dst")
            if dst is None or node_type_map.get(dst) != "H":
                continue
            h_t = node_time(node_by_id.get(dst, {}))
            resolved = False
            for ie in in_edges.get(dst, []):
                src = ie.get("src")
                if (
                    ie.get("relation") == "updates_to"
                    and node_type_map.get(src) == "H"
                    and node_time(node_by_id.get(src, {})) >= h_t
                ):
                    resolved = True
                    break
            if not resolved:
                for oe in out_edges.get(dst, []):
                    if oe.get("relation") == "competes_with":
                        resolved = True
                        break
                if not resolved:
                    for ie in in_edges.get(dst, []):
                        if ie.get("relation") == "competes_with":
                            resolved = True
                            break
            if not resolved:
                count += 1
    return count


def _ap_hypothesis_to_commitment_shortcut(
    node_type_map, _node_by_id, out_edges, in_edges
):
    # H->C is not a permitted edge; the commitment is reached via a shared J
    # that informs both the hypothesis and the commitment node.
    count = 0
    h_linked_to_c: set[str] = set()
    for j in _nodes_of_type("J", node_type_map):
        informs_c = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(j, [])
        )
        if not informs_c:
            continue
        for e in out_edges.get(j, []):
            if (
                e.get("relation") == "informs"
                and node_type_map.get(e.get("dst")) == "H"
            ):
                h_linked_to_c.add(e["dst"])
    for h in h_linked_to_c:
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if not tested:
            count += 1
    return count


def _ap_test_without_evidence(node_type_map, _node_by_id, out_edges, in_edges):
    count = 0
    for t in _nodes_of_type("T", node_type_map):
        has_ev = any(
            e.get("relation") in ("observes", "tests")
            and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(t, [])
        )
        if not has_ev:
            has_ev = any(
                e.get("relation") in ("observes", "tests")
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(t, [])
            )
        if not has_ev:
            count += 1
    return count


def _ap_no_belief_revision(_node_type_map, _node_by_id, out_edges, _in_edges):
    has_update = any(
        e.get("relation") == "updates_to"
        for edges_list in out_edges.values()
        for e in edges_list
    )
    return 0 if has_update else 1


def _ap_orphan_evidence(node_type_map, _node_by_id, out_edges, in_edges):
    return sum(
        1
        for ev in _nodes_of_type("E", node_type_map)
        if not out_edges.get(ev) and not in_edges.get(ev)
    )


def _ap_confirmation_only(node_type_map, _node_by_id, out_edges, in_edges):
    # H->C is not a permitted edge; the path must run through a shared J that
    # informs both H and C.  We additionally require supporting evidence for H
    # and the absence of any contradicting evidence to distinguish from genuine
    # belief-revision traces.
    count = 0
    h_linked_to_c: set[str] = set()
    for j in _nodes_of_type("J", node_type_map):
        informs_c = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(j, [])
        )
        if not informs_c:
            continue
        for e in out_edges.get(j, []):
            if (
                e.get("relation") == "informs"
                and node_type_map.get(e.get("dst")) == "H"
            ):
                h_linked_to_c.add(e["dst"])
    for h in h_linked_to_c:
        # Check H has supporting evidence (E->H or E->J->H via informs)
        has_support = False
        for ie in in_edges.get(h, []):
            src = ie.get("src")
            if ie.get("relation") == "informs" and node_type_map.get(src) == "J":
                for je in in_edges.get(src, []):
                    if (
                        je.get("relation") == "informs"
                        and node_type_map.get(je.get("src")) == "E"
                    ):
                        has_support = True
                        break
            if ie.get("relation") == "informs" and node_type_map.get(src) == "E":
                has_support = True
            if has_support:
                break
        if not has_support:
            continue
        has_contradiction = any(
            ie.get("relation") == "contradicts" for ie in in_edges.get(h, [])
        )
        if not has_contradiction:
            has_contradiction = any(
                oe.get("relation") == "contradicts" for oe in out_edges.get(h, [])
            )
        if not has_contradiction:
            count += 1
    return count


_ANTIPATTERN_MATCHERS = {
    AP_UNTESTED_CLAIM: _ap_untested_hypothesis,
    AP_EVIDENCE_NON_UPTAKE: _ap_evidence_ignored,
    AP_UNSUPPORTED_JUDGMENT: _ap_judgment_without_evidence,
    AP_STALLED_REVISION: _ap_dead_end_update,
    AP_CONTRADICTION_WITHOUT_REPAIR: _ap_unresolved_contradiction,
    AP_PREMATURE_COMMITMENT: _ap_hypothesis_to_commitment_shortcut,
    AP_UNINFORMATIVE_TEST: _ap_test_without_evidence,
    AP_FIXED_BELIEF_TRACE: _ap_no_belief_revision,
    AP_DISCONNECTED_EVIDENCE: _ap_orphan_evidence,
    AP_ONE_SIDED_CONFIRMATION: _ap_confirmation_only,
    AP_PRECOMMITTED_TEST_PLAN: _match_preregistered,
}


def detect_antipatterns_local(nodes: list, edges: list) -> dict[str, int]:
    node_by_id, out_edges, in_edges = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    return {
        name: matcher(node_type_map, node_by_id, out_edges, in_edges)
        for name, matcher in _ANTIPATTERN_MATCHERS.items()
    }


def detect_antipatterns_global(
    nodes: list, edges: list
) -> tuple[dict[str, bool], dict[str, int]]:
    node_by_id, out_edges, in_edges = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}

    n_H = sum(1 for t in node_type_map.values() if t == "H")
    n_T = sum(1 for t in node_type_map.values() if t == "T")
    n_E = sum(1 for t in node_type_map.values() if t == "E")

    # Count updates_to edges for fixed_belief_trace check
    n_updates_to = sum(1 for e in edges if e.get("relation") == "updates_to")

    h_tested = 0
    for h in _nodes_of_type("H", node_type_map):
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if tested:
            h_tested += 1

    e_used = 0
    for ev in _nodes_of_type("E", node_type_map):
        used = any(
            e.get("relation") == "informs"
            and node_type_map.get(e.get("dst")) in {"J", "H"}
            for e in out_edges.get(ev, [])
        )
        if not used:
            used = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) in {"J", "H"}
                for e in in_edges.get(ev, [])
            )
        if used:
            e_used += 1

    t_with_ev = 0
    for t in _nodes_of_type("T", node_type_map):
        has_ev = any(
            e.get("relation") in ("observes", "tests")
            and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(t, [])
        )
        if not has_ev:
            has_ev = any(
                e.get("relation") in ("observes", "tests")
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(t, [])
            )
        if has_ev:
            t_with_ev += 1

    n_contradicts = 0
    n_unresolved = 0
    for nid in node_type_map:
        for e in out_edges.get(nid, []):
            if e.get("relation") != "contradicts":
                continue
            dst = e.get("dst")
            if node_type_map.get(dst) != "H":
                continue
            n_contradicts += 1
            resolved = any(
                ie.get("relation") in ("updates_to", "competes_with")
                for ie in in_edges.get(dst, [])
            )
            if not resolved:
                resolved = any(
                    oe.get("relation") == "competes_with"
                    for oe in out_edges.get(dst, [])
                )
            if not resolved:
                n_unresolved += 1

    # Premature commitment: H linked to C via shared J, but H has no tests
    h_linked_to_c: set[str] = set()
    for j in _nodes_of_type("J", node_type_map):
        j_informs_c = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(j, [])
        )
        if not j_informs_c:
            continue
        for e in out_edges.get(j, []):
            if (
                e.get("relation") == "informs"
                and node_type_map.get(e.get("dst")) == "H"
            ):
                h_linked_to_c.add(e["dst"])
    h_to_c_untested = 0
    for h in h_linked_to_c:
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if not tested:
            h_to_c_untested += 1

    e_orphan = sum(
        1
        for ev in _nodes_of_type("E", node_type_map)
        if not out_edges.get(ev) and not in_edges.get(ev)
    )

    j_without_e = 0
    for j in _nodes_of_type("J", node_type_map):
        has_e = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(j, [])
        )
        if not has_e:
            has_e = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(j, [])
            )
        if not has_e:
            j_without_e += 1

    # Stalled revision: revised H (target of updates_to) with no outgoing tests
    dead_revised = 0
    for h in _nodes_of_type("H", node_type_map):
        is_revised = any(e.get("relation") == "updates_to" for e in in_edges.get(h, []))
        if not is_revised:
            continue
        has_test = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not has_test:
            dead_revised += 1
    ap_local_counts = detect_antipatterns_local(nodes, edges)

    # Precommitted test plan: C appears before any E
    t_first_C = earliest_time_of_type(nodes, "C")
    t_first_E = earliest_time_of_type(nodes, "E")
    precommitted = (
        t_first_C is not None and t_first_E is not None and t_first_C < t_first_E
    )

    binary: dict[str, bool] = {
        AP_UNTESTED_CLAIM: (n_H - h_tested) > 0,
        AP_EVIDENCE_NON_UPTAKE: (n_E - e_used) > 0,
        AP_UNSUPPORTED_JUDGMENT: j_without_e > 0,
        AP_STALLED_REVISION: dead_revised > 0,
        AP_CONTRADICTION_WITHOUT_REPAIR: n_unresolved > 0,
        AP_PREMATURE_COMMITMENT: h_to_c_untested > 0,
        AP_UNINFORMATIVE_TEST: (n_T - t_with_ev) > 0,
        AP_FIXED_BELIEF_TRACE: n_updates_to == 0,
        AP_DISCONNECTED_EVIDENCE: e_orphan > 0,
        AP_ONE_SIDED_CONFIRMATION: ap_local_counts.get(AP_ONE_SIDED_CONFIRMATION, 0)
        > 0,
        AP_PRECOMMITTED_TEST_PLAN: precommitted,
    }
    counts: dict[str, int] = {
        AP_UNTESTED_CLAIM: n_H - h_tested,
        AP_EVIDENCE_NON_UPTAKE: n_E - e_used,
        AP_UNSUPPORTED_JUDGMENT: j_without_e,
        AP_STALLED_REVISION: dead_revised,
        AP_CONTRADICTION_WITHOUT_REPAIR: n_unresolved,
        AP_PREMATURE_COMMITMENT: h_to_c_untested,
        AP_UNINFORMATIVE_TEST: n_T - t_with_ev,
        AP_FIXED_BELIEF_TRACE: 1 if n_updates_to == 0 else 0,
        AP_DISCONNECTED_EVIDENCE: e_orphan,
        AP_ONE_SIDED_CONFIRMATION: ap_local_counts.get(AP_ONE_SIDED_CONFIRMATION, 0),
        AP_PRECOMMITTED_TEST_PLAN: 1 if precommitted else 0,
    }
    return binary, counts
