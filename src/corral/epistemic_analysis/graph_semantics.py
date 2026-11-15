"""Claim-sized source spans, evidence provenance, and shared H/J support."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import pairwise

from .context import task_message_indices
from .schema import content_hash

GRAPH_SEMANTICS_VERSION = "task-given-evidence-claim-spans-shared-roles-final-claims-v2"

# Conservative boundaries: do not break formulas, decimal numbers, nested data,
# or quoted strings. The model still decides whether a clause is a scientific claim.
CLAUSE_BOUNDARY = re.compile(
    r"[;:]\s+(?=[A-Za-z])"
    r"|,\s+(?=(?:and|but|whereas|while|because|since|so|therefore|thus|hence)\b)"
    r"|\s+(?=(?:therefore|thus|hence|so I|I hypothesi[sz]e|I suspect|"
    r"a plausible (?:structure|explanation)|my hypothesis)\b)",
    re.IGNORECASE,
)


def claim_ranges(text):
    """Return contiguous exact ranges; shared context remains in the window."""
    protected, stack, quote, escape = set(), [], None, False
    for index, char in enumerate(text):
        if stack or quote:
            protected.add(index)
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
        elif char == '"':
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}" and stack:
            stack.pop()
    cuts = [0]
    for match in CLAUSE_BOUNDARY.finditer(text):
        if match.start() not in protected:
            boundary = match.end()
            if (
                len(text[cuts[-1] : boundary].strip()) >= 12
                and len(text[boundary:].strip()) >= 12
            ):
                cuts.append(boundary)
    cuts.append(len(text))
    ranges = []
    for low, high in pairwise(cuts):
        start, end = low, high
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            ranges.append((start, end))
    return ranges


def _submission_string_view(raw, mapping):
    """Decode JSON escapes while retaining each character's original range."""
    text, positions = [], []
    for match in re.finditer(
        r"\\u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}"
        r"|\\u[0-9a-fA-F]{4}|\\.|[^\\]",
        raw[1:-1],
        re.DOTALL,
    ):
        start, end = match.start() + 1, match.end() + 1
        value = json.loads('"' + match.group() + '"')
        text.append(value)
        positions.extend([(mapping[start][0], mapping[end - 1][1])] * len(value))
    return "".join(text), positions


def _submission_answer_view(raw, mapping):
    decoder = json.JSONDecoder()
    position = raw.index("{") + 1
    while position < len(raw):
        while raw[position].isspace() or raw[position] == ",":
            position += 1
        if raw[position] == "}":
            return None
        key, position = decoder.raw_decode(raw, position)
        while raw[position].isspace():
            position += 1
        if raw[position] != ":":
            return None
        position += 1
        while raw[position].isspace():
            position += 1
        start = position
        value, position = decoder.raw_decode(raw, position)
        fragment, offsets = raw[start:position], mapping[start:position]
        if key == "answer":
            return (
                _submission_string_view(fragment, offsets)
                if isinstance(value, str)
                else (fragment, offsets)
            )
        if key in {"function", "arguments"}:
            if isinstance(value, str):
                fragment, offsets = _submission_string_view(fragment, offsets)
            if isinstance(value, str | dict):
                result = _submission_answer_view(fragment, offsets)
                if result is not None:
                    return result
    return None


def submission_claim_ranges(quote):
    """Return sentence/clause candidates, never synthesized or decoded quotes.

    Candidates are still classified by the annotator. Structured or inseparable
    compound answers remain eligible for an explicit compound/unknown scope.
    """
    match = re.search(r"<action_input>(.*?)</action_input>", quote, re.DOTALL)
    start, end = match.span(1) if match else (0, len(quote))
    try:
        result = _submission_answer_view(
            quote[start:end], [(i, i + 1) for i in range(start, end)]
        )
    except (ValueError, IndexError, TypeError):
        return []  # Keep the original submission available for annotation.
    if result is None:
        return []
    text, mapping = result
    # Structured results must not be split inside strings or numeric values.
    if text.lstrip().startswith(("{", "[")):
        return [(mapping[0][0], mapping[-1][1])] if mapping else []
    ranges = []
    for sentence in re.finditer(r"\S[\s\S]*?(?=\n|(?<=[.!?])\s+|$)", text):
        for low, high in claim_ranges(sentence.group()):
            left, right = sentence.start() + low, sentence.start() + high
            ranges.append((mapping[left][0], mapping[right - 1][1]))
    return ranges


def annotate_node_semantics(nodes, trace, view):
    """Attach original-source identities before edge extraction for both backends."""
    observations = {m["msg_idx"] for m in view.mappings if m["is_tool_result"]}
    tasks = task_message_indices(trace.messages, observations)
    source_identity = trace.source.identity
    if trace.source.details.get("scientist_unit"):
        source_identity = source_identity.rsplit("#", 1)[0]
    groups = defaultdict(list)
    for node in nodes:
        index = node["time"]
        source = trace.message_sources[index]
        event = {
            "source_identity": source_identity,
            "source_content_hash": trace.source.content_hash,
            "parent_msg_idx": source.get("parent_msg_idx", index),
            "message_content_hash": content_hash(trace.messages[index]["content"]),
        }
        spans = [
            {
                "parent_msg_idx": trace.message_sources[s["msg_idx"]].get(
                    "parent_msg_idx", s["msg_idx"]
                ),
                **{k: s[k] for k in ("start", "end", "quote") if k in s},
            }
            for s in node.get("support", [])
        ]
        if node["type"] == "E":
            node["evidence_origin"] = (
                "tool_observation"
                if index in observations
                else "task_given"
                if index in tasks
                else "unknown"
            )
            node["evidence_source_id"] = content_hash(event)
            node["evidence_id"] = content_hash({"event": event, "spans": spans})
            node["evidence_provenance"] = {
                **event,
                "reconstructed_task_prompt": source.get("origin")
                == "reconstructed_task_prompt",
                "context_only": source.get("context_only", False),
            }
        if node["type"] in {"H", "J"}:
            node["epistemic_role"] = (
                "candidate_claim" if node["type"] == "H" else "evidence_interpretation"
            )
            if spans:
                key = content_hash(
                    {"event": event, "spans": sorted(spans, key=content_hash)}
                )
                groups[key].append(node)
    for key, shared in groups.items():
        if {n["type"] for n in shared} == {"H", "J"}:
            for node in shared:
                node["shared_support_id"] = key
                node["shared_support_with"] = [
                    n["node_id"] for n in shared if n is not node
                ]
                node["role_overlap"] = "same_passage"
    return nodes
