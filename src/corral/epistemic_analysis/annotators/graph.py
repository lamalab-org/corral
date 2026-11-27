"""One span-classification workflow for node and edge annotation."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import re
from contextvars import ContextVar
from dataclasses import dataclass, replace
from itertools import islice
from typing import TYPE_CHECKING, Any

from corral.epistemic_analysis.context import (
    SUBMIT_ANSWER_ACTION,
    current_context,
    message_support,
    observation_indices,
    task_message_indices,
    view_instructions,
)
from corral.epistemic_analysis.decision_applicability import SAVED_FIELDS
from corral.epistemic_analysis.epistemic_prompts import (
    EDGE_ONTOLOGY,
    NODE_ONTOLOGY,
    NODE_RULES,
    graph_duplicate_question,
    graph_edge_question,
    graph_edge_support_question,
    graph_final_claim_question,
    graph_node_question,
)
from corral.epistemic_analysis.graph import (
    coerce_edge,
    coerce_node,
    edge_pair_coverage,
    is_runtime_notice,
    iter_windows,
    normalize_whitespace,
    sha1_hex,
)
from corral.epistemic_analysis.graph_semantics import (
    GRAPH_SEMANTICS_VERSION,
    claim_ranges,
    submission_claim_ranges,
)
from corral.epistemic_analysis.ontology import (
    EDGE_RELATIONS,
    NODE_TYPES,
    PASS_A_INSTRUCTIONS,
    PASS_A_SYSTEM,
    PASS_B_INSTRUCTIONS,
    relation_allowed,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


MAX_CHOICES = 255
# Edge-support decisions offer each message plus a final "none" choice.
MAX_WINDOW = MAX_CHOICES - 1
TAGGED_UNIT = re.compile(
    r"<action>.*?</action>(?:\s*<action_input>.*?</action_input>)?"
    r"|<final_answer>.*?</final_answer>",
    re.DOTALL,
)


@dataclass(frozen=True)
class AnnotationBackend:
    """Node and edge passes with reproducibility metadata."""

    extract_nodes: Callable[..., Awaitable[tuple[list[dict[str, Any]], list[str]]]]
    extract_edges: Callable[..., Awaitable[tuple[list[dict[str, Any]], list[str]]]]
    provenance: dict[str, Any]
    prepare: Callable[[], Awaitable[Any]] | None = None
    close: Callable[[], Awaitable[None]] | None = None

    def resolved_models(self) -> set[str]:
        return _resolved_models.get() or set()


_resolved_models: ContextVar[set[str] | None] = ContextVar(
    "annotation_models", default=None
)


def record_model(model: str | None) -> None:
    if model:
        _resolved_models.set((_resolved_models.get() or set()) | {model})


def reset_models() -> None:
    _resolved_models.set(set())


class AnnotationError(RuntimeError):
    """A failed request or an invalid structured decision."""


def _probability(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def validate_answers(
    payload: Any, questions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Reject incomplete responses instead of silently dropping annotations."""
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise AnnotationError("Annotation response is missing the answers map")
    for key, question in questions.items():
        answer = answers.get(key)
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise AnnotationError(f"Missing or mistyped annotation answer: {key}")
        if question["type"] == "noul":
            if not _probability(answer.get("noul")):
                raise AnnotationError(f"Invalid Noul probability: {key}")
        else:
            probabilities = answer.get("probabilities")
            choice = answer.get("choice")
            if (
                not isinstance(choice, str)
                or choice not in question["criteria"]
                or not _probability(answer.get("confidence"))
                or (
                    probabilities is not None
                    and (
                        not isinstance(probabilities, dict)
                        or set(probabilities) != set(question["criteria"])
                        or not all(_probability(p) for p in probabilities.values())
                        or not math.isclose(
                            sum(probabilities.values()), 1, abs_tol=0.01
                        )
                    )
                )
            ):
                raise AnnotationError(f"Invalid Choice answer: {key}")
    return {key: answers[key] for key in questions}


@dataclass(frozen=True)
class Span:
    msg_idx: int
    offset: int
    quote: str
    kind: str
    tool_name: str | None = None

    @property
    def key(self) -> str:
        return f"m{self.msg_idx}_s{self.offset}"

    @property
    def support(self) -> list[dict[str, Any]]:
        context = current_context()
        if context is not None:
            retained = []
            for r in context.view.mappings[self.msg_idx]["retained_ranges"]:
                start = max(self.offset, r["view_start"])
                end = min(self.offset + len(self.quote), r["view_end"])
                if start < end:
                    original = r["original_start"] + start - r["view_start"]
                    retained.append(
                        {
                            "msg_idx": self.msg_idx,
                            "quote": self.quote[
                                start - self.offset : end - self.offset
                            ],
                            "start": original,
                            "end": original + end - start,
                        }
                    )
            return retained
        return [
            {
                "msg_idx": self.msg_idx,
                "quote": self.quote,
                "start": self.offset,
                "end": self.offset + len(self.quote),
            }
        ]


def _prose_spans(
    messages: list[dict[str, Any]], max_chars: int, *, observations=None, tasks=None
) -> list[Span]:
    """Split prose into exact substrings; preserve actions and their arguments."""
    spans = []

    def add_prose(idx: int, text: str, offset: int, kind: str = "prose") -> None:
        for match in re.finditer(r"\S[\s\S]*?(?=\n|(?<=[.!?])\s+|$)", text):
            quote = match.group()
            for low, high in claim_ranges(quote):
                for start in range(low, high, max_chars):
                    part = quote[start : min(high, start + max_chars)]
                    if part.strip():
                        spans.append(
                            Span(idx, offset + match.start() + start, part, kind)
                        )

    observations = (
        observation_indices(messages) if observations is None else observations
    )
    tasks = task_message_indices(messages, observations) if tasks is None else tasks
    for idx, message in enumerate(messages):
        content = str(message.get("content", "") or "")
        if idx in observations:
            spans.append(Span(idx, 0, content, "observation"))
            continue
        if idx in tasks and content.strip():
            add_prose(idx, content, 0, "task")
            continue
        # System instructions and later user messages supply context.
        if message.get("role", "").lower() != "assistant":
            continue
        if is_runtime_notice(content):
            continue  # Assigned N deterministically after node extraction.
        cursor = 0
        for match in TAGGED_UNIT.finditer(content):
            add_prose(idx, content[cursor : match.start()], cursor)
            if match.group().startswith("<action>"):
                spans.append(Span(idx, match.start(), match.group(), "action"))
            else:
                # Classify the contents too, so F does not hide H/J/C nodes.
                body_start = match.start() + len("<final_answer>")
                body_end = match.end() - len("</final_answer>")
                add_prose(idx, content[body_start:body_end], body_start, "final")
            cursor = match.end()
        add_prose(idx, content[cursor:], cursor)
    return spans


def _saved_json_members(text):
    """Return exact top-level JSON member bounds, including escaped string values."""
    decoder = json.JSONDecoder()
    position = text.index("{") + 1
    while True:
        while text[position].isspace() or text[position] == ",":
            position += 1
        if text[position] == "}":
            return
        key, position = decoder.raw_decode(text, position)
        while text[position].isspace():
            position += 1
        if text[position] != ":":
            raise ValueError("Invalid saved JSON member")
        position += 1
        while text[position].isspace():
            position += 1
        start = position
        value, position = decoder.raw_decode(text, position)
        yield key, value, start, position


def candidate_spans(messages, max_chars):
    """Preserve original substrings; never classify serialized metadata as prose."""
    prose = copy.deepcopy(messages)
    calls = []
    for index, message in enumerate(messages):
        if message.get("role", "").lower() != "assistant":
            continue
        content, marker, suffix = str(message.get("content", "") or "").rpartition(
            SAVED_FIELDS
        )
        if not marker:
            continue
        try:
            fields = json.loads(suffix)
        except ValueError:
            continue  # Keep malformed text visible rather than discard it.
        if not isinstance(fields, dict):
            continue
        prose[index]["content"] = "" if content.strip() == "null" else content
        # ReAct dispatch fields duplicate the actual action tags, whose original
        # source spans already identify the calls, including folded call batches.
        tagged = list(
            re.finditer(
                r"<action>(.*?)</action>\s*<action_input>(.*?)</action_input>",
                content,
                re.DOTALL,
            )
        )
        duplicates = []
        for match in tagged:
            with contextlib.suppress(ValueError):
                duplicates.append((match[1].strip(), json.loads(match[2])))
        base = len(content) + len(marker)
        for key, value, start, end in _saved_json_members(suffix):
            entries = []
            if key == "function_call" and isinstance(value, dict):
                entries.append((value, start, end))
            elif key == "tool_calls" and isinstance(value, list):
                position = start + 1
                for _ in value:
                    while suffix[position].isspace() or suffix[position] == ",":
                        position += 1
                    item, following = json.JSONDecoder().raw_decode(suffix, position)
                    entries.append((item, position, following))
                    position = following
            for item, left, right in entries:
                if not isinstance(item, dict):
                    continue
                function = item.get("function", item)
                if not isinstance(function, dict) or not isinstance(
                    function.get("name"), str
                ):
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    with contextlib.suppress(ValueError):
                        arguments = json.loads(arguments)
                pair = (function["name"], arguments)
                if pair in duplicates:
                    duplicates.remove(pair)
                    continue
                calls.append(
                    Span(
                        index,
                        base + left,
                        suffix[left:right],
                        "action",
                        function["name"],
                    )
                )
    spans = sorted(
        [
            *_prose_spans(
                prose,
                max_chars,
                observations=observation_indices(messages),
                tasks=task_message_indices(messages),
            ),
            *calls,
        ],
        key=lambda s: (s.msg_idx, s.offset),
    )
    expanded = []
    for span in spans:
        ranges = (
            submission_claim_ranges(span.quote)
            if GraphExtractor.submission_span(span)
            else []
        )
        if ranges:
            expanded.extend(
                Span(
                    span.msg_idx,
                    span.offset + low,
                    span.quote[low:high],
                    "final",
                    "submit_answer",
                )
                for low, high in ranges
            )
        else:
            expanded.append(span)
    spans = expanded
    context = current_context()
    if context is None or messages is not context.view.messages:
        return spans
    retained = []
    for span in spans:
        if span.kind == "observation":
            retained.append(span)
            continue
        for r in context.view.mappings[span.msg_idx]["retained_ranges"]:
            start = max(span.offset, r["view_start"])
            end = min(span.offset + len(span.quote), r["view_end"])
            quote = span.quote[start - span.offset : end - span.offset]
            if start < end and quote.strip():
                intact = start == span.offset and end == span.offset + len(span.quote)
                retained.append(
                    replace(
                        span,
                        offset=start,
                        quote=quote,
                        kind=span.kind if intact or span.kind == "task" else "prose",
                        tool_name=span.tool_name if intact else None,
                    )
                )
    return retained


def window_state(
    messages: list[dict[str, Any]], start: int, end: int
) -> dict[str, Any]:
    from corral.epistemic_analysis.annotation_transcript import initial_instructions

    state = {
        "messages": {f"m{i}": {"msg_idx": i, **messages[i]} for i in range(start, end)},
        "window": {"start": start, "end": end},
        "node_ontology": NODE_ONTOLOGY,
        "initial_instructions": initial_instructions(messages),
    }
    if view_instructions():
        state["annotation_view_instructions"] = view_instructions()
        context = current_context()
        if context is not None and messages is context.view.messages:
            signals = {
                f"m{i}": context.view.mappings[i]["repetition_signals"]
                for i in range(start, end)
                if context.view.mappings[i].get("repetition_signals")
            }
            if signals:
                state["repetition_signals"] = signals
    return state


class GraphExtractor:
    @staticmethod
    def submission_span(span):
        return span.tool_name == "submit_answer" or (
            span.kind == "action" and SUBMIT_ANSWER_ACTION.match(span.quote)
        )

    @classmethod
    def eligible_node_types(cls, span: Span) -> tuple[str, ...]:
        """Labels permitted by the ontology, before any model decision."""
        if span.kind in {"observation", "task"}:
            return ("E",)
        return tuple(
            node_type
            for node_type in NODE_TYPES
            if node_type != "E"
            and (node_type != "N" or span.kind == "action")
            and (node_type != "F" or span.kind == "final" or cls.submission_span(span))
        )

    def __init__(
        self,
        client,
        min_confidence: float = 0.6,
        max_span_chars: int = 1200,
    ):
        self.client = client
        self.min_confidence = min_confidence
        self.max_span_chars = max_span_chars

    def backend(self, *, stop_at_submission=True) -> AnnotationBackend:
        backend = AnnotationBackend(
            extract_nodes=self.extract_nodes,
            extract_edges=self.extract_edges,
            provenance={
                "annotator": "graph",
                "candidate_extractor": GRAPH_SEMANTICS_VERSION,
                "edge_coverage": "source-aware-eligible-pairs-in-windows-v2",
                "stop_at_submission": stop_at_submission,
                "node_postprocessing": "test_precedence_neutral_fallback_v1",
                "node_decisions": "independent_choices_v1",
                "node_eligibility": "pre_question_type_constraints_v1",
                "judgment_consolidation": "same_assessment_complete_link_v1",
                "final_claim_scope_sha1": sha1_hex(
                    json.dumps(graph_final_claim_question("final"), sort_keys=True)
                ),
                "duplicate_prompt_sha1": sha1_hex(
                    json.dumps(
                        {
                            t: graph_duplicate_question("first", "second", t)
                            for t in NODE_TYPES
                        },
                        sort_keys=True,
                    )
                ),
                "min_confidence": self.min_confidence,
                "max_span_chars": self.max_span_chars,
                "endpoint": getattr(self.client, "endpoint", None),
                "batch_size": getattr(self.client, "batch_size", 32),
                "timeout_s": getattr(self.client, "timeout_s", 120),
                "max_retries": getattr(self.client, "max_retries", 5),
                "decision_prompt_sha1": sha1_hex(
                    PASS_A_SYSTEM + NODE_ONTOLOGY + NODE_RULES + EDGE_ONTOLOGY
                ),
                "ontology_sha1": sha1_hex(PASS_A_INSTRUCTIONS + PASS_B_INSTRUCTIONS),
            },
        )
        # Client metadata identifies the wire protocol independently of the
        # shared extraction algorithm. Workflow options belong to this call.
        backend.provenance.update(getattr(self.client, "provenance", {}))
        return backend

    async def extract_nodes(
        self,
        messages: list[dict[str, Any]],
        model: str,
        window: int,
        overlap: int,
        max_nodes_per_window: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        spans = candidate_spans(messages, self.max_span_chars)
        if self.client is None:
            raise AnnotationError("Annotation client is required for annotation")
        nodes: list[dict[str, Any]] = []
        warnings: list[str] = []
        seen: set[str] = set()
        next_id = 1

        def review(message):
            warnings.append(message)
            context = current_context()
            if context is not None:
                context.review_issues.append(message)

        def node_state(start, end, current):
            state = window_state(messages, start, end)
            state["annotation_rules"] = PASS_A_SYSTEM
            state["candidates"] = {
                s.key: {"msg_idx": s.msg_idx, "kind": s.kind, "quote": s.quote}
                for s in current
            }
            return state

        def node_questions(current):
            for span in current:
                for kind in self.eligible_node_types(span):
                    yield f"{span.key}_{kind}", graph_node_question(span.key, kind)

        windows = []
        for start, end in iter_windows(len(messages), window, overlap):
            current = [
                s for s in spans if start <= s.msg_idx < end and s.key not in seen
            ]
            if not current:
                continue
            seen.update(s.key for s in current)
            windows.append((start, end, current))
            preflight = getattr(self.client, "preflight_large_window", None)
            if preflight is not None:
                chars = sum(len(m["content"]) for m in messages[start:end]) + sum(
                    len(s.quote) for s in current
                )
                if chars >= self.client.provider.config.context_tokens:
                    await preflight(
                        model,
                        node_state(start, end, current),
                        dict(islice(node_questions(current), self.client.batch_size)),
                    )

        for start, end, current in windows:
            state = node_state(start, end, current)
            questions = dict(node_questions(current))
            answers = await self.client.evaluate(model, state, questions)
            extracted = []
            for span in current:
                selected = {}
                submission = self.submission_span(span)
                test_over_neutral = (
                    span.kind == "action"
                    and not submission
                    and all(
                        answers[f"{span.key}_{kind}"]["choice"] == "present"
                        for kind in ("T", "N")
                    )
                )
                for node_type in NODE_TYPES:
                    answer = answers.get(f"{span.key}_{node_type}")
                    if answer is None:
                        continue
                    if answer["choice"] != "present":
                        continue
                    if (
                        node_type == "N"
                        and submission
                        and answers.get(f"{span.key}_F", {}).get("choice") == "present"
                        and answers[f"{span.key}_F"]["confidence"]
                        >= self.min_confidence
                    ):
                        continue  # F satisfies submission coverage; N is only a fallback.
                    if answer["confidence"] < self.min_confidence and not (
                        test_over_neutral and node_type == "T"
                    ):
                        warnings.append(
                            f"Low node confidence; omitted {span.key} ({node_type})."
                        )
                        continue
                    if node_type not in self.eligible_node_types(span):
                        review(
                            f"Rejected {span.key} ({node_type}): violates node type constraints."
                        )
                        continue
                    selected[node_type] = answer
                neutral_fallback = span.kind == "action" and not selected
                if neutral_fallback:
                    selected["N"] = answers[f"{span.key}_N"]
                if selected.keys() - {"N"}:
                    selected.pop("N", None)
                if span.kind == "observation" and "E" not in selected:
                    review(
                        f"Observation {span.key} was not classified E; restored by ontology validation."
                    )
                for node_type, answer in selected.items():
                    node = {
                        "node_id": f"N{next_id}",
                        "type": node_type,
                        "time": span.msg_idx,
                        "text": normalize_whitespace(span.quote),
                        "support": message_support(messages, span.msg_idx)
                        if span.kind == "observation"
                        else span.support,
                        "jev": answer,
                        "candidate_kind": span.kind,
                    }
                    if node_type == "E":
                        node["evidence_origin"] = (
                            "task_given" if span.kind == "task" else "tool_observation"
                        )
                    if node_type == "T" and test_over_neutral:
                        node["postprocessing"] = "test_over_neutral"
                    elif node_type == "N" and neutral_fallback:
                        node["postprocessing"] = "neutral_fallback"
                    next_id += 1
                    coerced, qc = coerce_node(node, messages)
                    warnings.extend(qc)
                    if coerced is not None:
                        extracted.append(coerced)
            extracted, qc = await self._merge_duplicates(model, state, extracted)
            warnings.extend(qc)
            finals = [n for n in extracted if n["type"] == "F"]
            if finals:
                scopes = await self.client.evaluate(
                    model,
                    {
                        **state,
                        "nodes": {n["node_id"]: copy.deepcopy(n) for n in finals},
                    },
                    {
                        f"claim_{n['node_id']}": graph_final_claim_question(
                            n["node_id"]
                        )
                        for n in finals
                    },
                )
                for node in finals:
                    answer = scopes[f"claim_{node['node_id']}"]
                    node["claim_scope"] = (
                        answer["choice"]
                        if answer["confidence"] >= self.min_confidence
                        else "unknown"
                    )
                    node["claim_scope_decision"] = answer
            if len(extracted) > max_nodes_per_window:
                review(
                    f"PassA[{start}-{end}] Node cap reached; omitted {len(extracted) - max_nodes_per_window} nodes."
                )
            nodes.extend(extracted[:max_nodes_per_window])
        for idx in {
            s.msg_idx for s in spans if s.kind == "final" or self.submission_span(s)
        }:
            if not any(n["time"] == idx and n["type"] == "F" for n in nodes):
                review(f"Final-answer message {idx} has no F node.")
        nodes.sort(key=lambda n: n["time"])
        return nodes, warnings

    async def _merge_duplicates(self, model, state, nodes):
        """Consolidate assessments and merge repeated instances before edge extraction."""
        originals = {n["node_id"]: copy.deepcopy(n) for n in nodes if n["type"] == "J"}
        questions, pairs = {}, {}
        for i, first in enumerate(nodes):
            for second in nodes[i + 1 :]:
                if (first["time"], first["type"]) != (second["time"], second["type"]):
                    continue
                key = f"same_{first['node_id']}_{second['node_id']}"
                pairs[key] = (first, second)
                questions[key] = graph_duplicate_question(
                    first["node_id"], second["node_id"], first["type"]
                )
        state = {**state, "nodes": {n["node_id"]: copy.deepcopy(n) for n in nodes}}
        answers = (
            await self.client.evaluate(model, state, questions) if questions else {}
        )
        representatives = {n["node_id"]: n for n in nodes}
        warnings = []
        for key, (first, second) in pairs.items():
            answer = answers[key]
            if answer["choice"] != "same":
                continue
            if answer["confidence"] < self.min_confidence:
                warnings.append(f"Uncertain duplicate {key}; retained both instances.")
                continue
            kept, removed = (
                representatives[first["node_id"]],
                representatives[second["node_id"]],
            )
            if kept is removed:
                continue
            if kept["type"] == "J":
                left = {key for key, value in representatives.items() if value is kept}
                right = {
                    key for key, value in representatives.items() if value is removed
                }
                # A transitive chain of "same" votes must not override a distinct
                # or uncertain assessment pair elsewhere in the proposed group.
                crossing = [
                    answers[name]
                    for name, (a, b) in pairs.items()
                    if (a["node_id"] in left and b["node_id"] in right)
                    or (a["node_id"] in right and b["node_id"] in left)
                ]
                if any(
                    a["choice"] != "same" or a["confidence"] < self.min_confidence
                    for a in crossing
                ):
                    warnings.append(
                        f"Judgment group {key} has conflicting or uncertain pairs; retained separate assessments."
                    )
                    continue
            planned_and_executed = kept["type"] == "T" and (
                (kept.get("candidate_kind") == "action")
                != (removed.get("candidate_kind") == "action")
            )
            if planned_and_executed and removed.get("candidate_kind") == "action":
                kept, removed = removed, kept
            if not planned_and_executed:
                kept["support"].extend(
                    s for s in removed["support"] if s not in kept["support"]
                )
            kept.setdefault("merged_nodes", []).append(removed["node_id"])
            for identity, representative in representatives.items():
                if representative is removed:
                    representatives[identity] = kept
        for node in nodes:
            if node["type"] != "J" or representatives[node["node_id"]] is not node:
                continue
            members = [key for key in originals if representatives[key] is node]
            if len(members) < 2:
                continue
            node["support"].sort(key=lambda s: (s["msg_idx"], s.get("start", 0)))
            node["text"] = normalize_whitespace(
                " ".join(s["quote"] for s in node["support"])
            )
            node["merged_nodes"] = [key for key in members if key != node["node_id"]]
            node["judgment_consolidation"] = {
                "method": "model_same_assessment_v1",
                "original_nodes": [originals[key] for key in members],
                "decisions": [
                    {"nodes": [a["node_id"], b["node_id"]], "answer": answers[name]}
                    for name, (a, b) in pairs.items()
                    if a["node_id"] in members and b["node_id"] in members
                ],
            }
            # The representative fragment's type score is not a score for the
            # whole assessment. Preserve individual scores in the audit above.
            node.pop("jev", None)
        return [n for n in nodes if representatives[n["node_id"]] is n], warnings

    async def extract_edges(
        self,
        messages: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        model: str,
        window: int,
        overlap: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if self.client is None:
            raise AnnotationError("Annotation client is required for annotation")
        edges = []
        warnings = []
        seen = set()
        for start, end in iter_windows(len(messages), window, overlap):
            current = [n for n in nodes if start <= n["time"] < end]
            state = window_state(messages, start, end)
            state["edge_ontology"] = EDGE_ONTOLOGY
            state["nodes"] = {
                n["node_id"]: {
                    k: n[k]
                    for k in (
                        "type",
                        "time",
                        "text",
                        "support",
                        "evidence_origin",
                        "evidence_source_id",
                        "shared_support_id",
                        "epistemic_role",
                        "claim_scope",
                    )
                    if k in n
                }
                for n in current
            }
            questions = {}
            pairs = {}
            for i, src in enumerate(current):
                for j, dst in enumerate(current):
                    pair = (src["node_id"], dst["node_id"])
                    if i == j or pair in seen:
                        continue
                    relations = []
                    for rel in EDGE_RELATIONS:
                        if not relation_allowed(rel, src, dst):
                            continue
                        # Use the canonical directions described in Pass B.
                        if rel == "tests" and src["type"] == "T":
                            continue
                        if src["type"] == dst["type"] and i > j:
                            continue
                        relations.append(rel)
                    if not relations:
                        continue
                    seen.add(pair)
                    key = f"edge_{src['node_id']}_{dst['node_id']}"
                    pairs[key] = pair
                    questions[key] = graph_edge_question(pair[0], pair[1], relations)
            answers = await self.client.evaluate(model, state, questions)
            selected = {
                key: answer
                for key, answer in answers.items()
                if answer["choice"] != "unrelated"
                and answer["confidence"] >= self.min_confidence
            }
            # A second decision selects actual evidence for the chosen relation.
            # Question IDs are not model input, so instructions name both nodes.
            support_questions = {}
            for key, answer in selected.items():
                src_id, dst_id = pairs[key]
                support_questions[key] = graph_edge_support_question(
                    src_id,
                    dst_id,
                    answer["choice"],
                    [
                        m
                        for m, message in state["messages"].items()
                        if message.get("content")
                    ],
                )
            supports = await self.client.evaluate(model, state, support_questions)
            for key, answer in supports.items():
                if (
                    answer["choice"] == "none"
                    or answer["confidence"] < self.min_confidence
                ):
                    warnings.append(f"Omitted {key}: no confident supporting message.")
                    continue
                message = state["messages"][answer["choice"]]
                src_id, dst_id = pairs[key]
                edge = {
                    "src": src_id,
                    "dst": dst_id,
                    "relation": selected[key]["choice"],
                    "time": message["msg_idx"],
                    "support": message_support(messages, message["msg_idx"]),
                    "jev": {"relation": selected[key], "support": answer},
                }
                coerced, qc = coerce_edge(edge, messages, set(state["nodes"]))
                warnings.extend(qc)
                if coerced is not None:
                    edges.append(coerced)
        # Record candidate-pair coverage. Local windows never imply that distant
        # nodes are unrelated, and their omissions remain reviewable.
        coverage = edge_pair_coverage(nodes, window, overlap)
        context = current_context()
        if context is not None:
            context.diagnostics.append(coverage)
        if coverage["unexamined_node_pairs"]:
            issue = f"{coverage['unexamined_node_pairs']} eligible node pairs cross graph windows and were not examined."
            warnings.append(issue)
            if context is not None:
                context.review_issues.append(issue)
        return edges, warnings
