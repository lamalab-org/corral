"""Graph for epistemic trace analysis."""

from __future__ import annotations

import re
from typing import Any

from corral.epistemic_analysis._utils import (
    normalize_whitespace,
    sha1_hex,
)
from corral.epistemic_analysis.ontology import (
    ALLOWED_EDGE_TYPE_COMBOS,
    EDGE_RELATIONS_SET,
    NODE_TYPES_SET,
)

from .context import current_context, message_support, observation_indices


def is_runtime_notice(content: str) -> bool:
    """Recognize the runner's budget notice without matching ordinary reasoning."""
    return bool(
        re.fullmatch(
            r"agent exhausted its \d+ interaction budget\.?", content.strip(), re.I
        )
    )


def _postprocess_runtime_notices(
    nodes: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Give otherwise unlabelled assistant runtime notices a Neutral node."""
    result = list(nodes)
    covered = {node["time"] for node in nodes}
    identities = {node["node_id"] for node in nodes}
    next_id = 1
    warnings = []
    observations = observation_indices(messages)
    for idx, message in enumerate(messages):
        content = message.get("content", "")
        if (
            idx in covered
            or idx in observations
            or message.get("role", "").lower() != "assistant"
            or not is_runtime_notice(content)
        ):
            continue
        while f"N{next_id}" in identities:
            next_id += 1
        identity = f"N{next_id}"
        identities.add(identity)
        result.append(
            {
                "node_id": identity,
                "type": "N",
                "time": idx,
                "text": normalize_whitespace(content),
                "support": message_support(messages, idx),
                "postprocessing": "runtime_notice_neutral_fallback",
            }
        )
        warnings.append(f"Added Neutral node {identity} for runtime notice {idx}.")
    result.sort(key=lambda node: node["time"])
    return result, warnings


def validate_support_quotes(
    messages: list[dict[str, Any]],
    support: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    ok = True
    for s in support:
        if not isinstance(s, dict) or "msg_idx" not in s or "quote" not in s:
            ok = False
            warnings.append("Support item missing msg_idx or quote.")
            continue
        idx = s["msg_idx"]
        quote = s["quote"]
        if (
            not isinstance(idx, int)
            or isinstance(idx, bool)
            or idx < 0
            or idx >= len(messages)
        ):
            ok = False
            warnings.append(f"Support msg_idx out of range: {idx}")
            continue
        msg_content = str(messages[idx].get("content", "") or "")
        context = current_context()
        visible = [msg_content]
        if context is not None and messages is context.view.messages:
            visible = [item["quote"] for item in context.view.support(idx)]
        if (
            not isinstance(quote, str)
            or not quote
            or not any(quote in text for text in visible)
        ):
            ok = False
            warnings.append(f"Quote not found in a retained range of message {idx}.")
        elif "start" in s or "end" in s:
            # View coordinates are mapped by the pipeline before saving.
            if context is None or messages is not context.view.messages:
                start, end = s.get("start"), s.get("end")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(msg_content)
                    or msg_content[start:end] != quote
                ):
                    ok = False
                    warnings.append(
                        f"Invalid original support offsets in message {idx}."
                    )
    return ok, warnings


def coerce_node(
    node: dict[str, Any], messages: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[str]]:
    warns: list[str] = []
    t = node.get("type")
    if t not in NODE_TYPES_SET:
        return None, [f"Invalid node type: {t}"]
    support = node.get("support", [])
    _ok, w = validate_support_quotes(
        messages, support if isinstance(support, list) else []
    )
    warns.extend(w)

    time_idx = node.get("time")
    if not isinstance(time_idx, int):
        try:
            time_idx = min(int(x["msg_idx"]) for x in support)
        except Exception:
            time_idx = 0
            warns.append("Node time missing and could not be derived; defaulting to 0.")
        node["time"] = time_idx

    node["text"] = normalize_whitespace(str(node.get("text", "")))

    if t == "H":
        hyp = node.get("hypothesis") or {}
        canonical = normalize_whitespace(str(hyp.get("canonical", node["text"])))
        hid = hyp.get("hid")
        if not hid or not isinstance(hid, str) or len(hid) < 10:
            hid = sha1_hex(canonical)
        node["hypothesis"] = {"canonical": canonical, "hid": hid}
    else:
        node.pop("hypothesis", None)

    return node, warns


def coerce_edge(
    edge: dict[str, Any], messages: list[dict[str, Any]], node_ids: set
) -> tuple[dict[str, Any] | None, list[str]]:
    warns: list[str] = []
    rel = edge.get("relation")
    if rel not in EDGE_RELATIONS_SET:
        return None, [f"Invalid edge relation: {rel}"]
    src = edge.get("src")
    dst = edge.get("dst")
    if src not in node_ids or dst not in node_ids:
        return None, [f"Edge references unknown node_id: src={src}, dst={dst}"]

    support = edge.get("support", [])
    _ok, w = validate_support_quotes(
        messages, support if isinstance(support, list) else []
    )
    warns.extend(w)

    time_idx = edge.get("time")
    if not isinstance(time_idx, int):
        try:
            time_idx = min(int(x["msg_idx"]) for x in support)
        except Exception:
            time_idx = 0
            warns.append("Edge time missing and could not be derived; defaulting to 0.")
        edge["time"] = time_idx

    return edge, warns


def build_adjacency(
    edges: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    out_edges: dict[str, list[dict[str, Any]]] = {}
    in_edges: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        s = e["src"]
        d = e["dst"]
        out_edges.setdefault(s, []).append(e)
        in_edges.setdefault(d, []).append(e)
    return out_edges, in_edges


def build_index(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    node_by_id = {n["node_id"]: n for n in nodes if "node_id" in n}
    out_edges: dict[str, list[dict[str, Any]]] = {}
    in_edges: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        s, d = e.get("src"), e.get("dst")
        if s is None or d is None:
            continue
        out_edges.setdefault(s, []).append(e)
        in_edges.setdefault(d, []).append(e)
    return node_by_id, out_edges, in_edges


def node_time(n: dict[str, Any]) -> int:
    t = n.get("time", 0)
    try:
        return int(t)
    except Exception:
        return 0


def earliest_time_of_type(nodes: list[dict[str, Any]], t: str) -> int | None:
    ts = [node_time(n) for n in nodes if n.get("type") == t]
    return min(ts) if ts else None


def _postprocess_observation_nodes(
    nodes: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Enforce that observation messages have exactly one E node each.

    Observations remain E even after neutral tool calls. N is restricted to
    the calls themselves, as required by the node definitions.

    Args:
        nodes: Nodes extracted by Pass A for the current trace.
        messages: Full message list of the trace (role + content dicts).

    Returns:
        A tuple of (updated node list, list of QC warning strings).
    """
    warnings: list[str] = []

    # Identify observation message indices
    obs_indices = observation_indices(messages)

    if not obs_indices:
        return nodes, warnings

    task_idx = next(
        (
            i
            for i, m in enumerate(messages)
            if m.get("role", "").lower() == "user" and i not in obs_indices
        ),
        None,
    )

    # Compute max node id for generating new ones
    max_id = max(
        (
            int(n["node_id"][1:])
            for n in nodes
            if n.get("node_id", "").startswith("N") and n["node_id"][1:].isdigit()
        ),
        default=0,
    )

    obs_covered: set[int] = set()
    result: list[dict[str, Any]] = []

    for original_node in nodes:
        n = original_node
        t = n.get("time")
        ntype = n.get("type")

        if isinstance(t, int) and t in obs_indices:
            if t not in obs_covered:
                context = current_context()
                if context and context.view.mappings[t]["omitted_chars"]:
                    n = {
                        **n,
                        "text": normalize_whitespace(messages[t]["content"]),
                        "support": message_support(messages, t),
                    }
                # Keep first node but ensure correct type
                target_type = "E"
                if ntype != target_type:
                    warnings.append(
                        f"Changed node {n.get('node_id')} at observation message {t} "
                        f"from {ntype} to {target_type}."
                    )
                    node = dict(n)
                    node["type"] = target_type
                    result.append(node)
                else:
                    result.append(n)
                obs_covered.add(t)
            else:
                warnings.append(
                    f"Removed extra node {n.get('node_id')} ({ntype}) "
                    f"at observation message {t}: only one node per observation."
                )
        elif ntype == "E" and t != task_idx:
            warnings.append(
                f"Removed Evidence node {n.get('node_id')} at message {t}: "
                f"not an Observation message."
            )
        else:
            result.append(n)

    # Add missing nodes for uncovered observation messages
    for obs_idx in sorted(obs_indices - obs_covered):
        max_id += 1
        content = str(messages[obs_idx].get("content", "") or "")
        target_type = "E"
        new_node = {
            "node_id": f"N{max_id}",
            "type": target_type,
            "time": obs_idx,
            "text": normalize_whitespace(content[:200]),
            "support": message_support(messages, obs_idx)
            if current_context()
            and current_context().view.mappings[obs_idx]["omitted_chars"]
            else [{"msg_idx": obs_idx, "quote": content[:500]}],
        }
        result.append(new_node)
        warnings.append(
            f"Added {target_type} node N{max_id} for observation message {obs_idx} "
            f"(no node was previously assigned)."
        )

    result.sort(key=lambda n: (n.get("time", 0), n.get("node_id", "")))
    return result, warnings


def _filter_invalid_edges(
    edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove edges whose (relation, src_type, dst_type) is not in ALLOWED_EDGE_TYPE_COMBOS."""
    warnings: list[str] = []
    node_type_map = {n["node_id"]: n.get("type") for n in nodes if "node_id" in n}

    valid_edges: list[dict[str, Any]] = []
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        rel = e.get("relation")
        src_type = node_type_map.get(src)
        dst_type = node_type_map.get(dst)

        if src_type is None or dst_type is None:
            warnings.append(f"Removed edge {src}->{dst} ({rel}): unknown node type.")
            continue

        if (rel, src_type, dst_type) not in ALLOWED_EDGE_TYPE_COMBOS:
            warnings.append(
                f"Removed edge {src}({src_type})->{dst}({dst_type}) ({rel}): "
                f"disallowed type combination."
            )
            continue

        valid_edges.append(e)

    return valid_edges, warnings
