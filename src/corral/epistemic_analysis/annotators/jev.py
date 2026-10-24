"""Jev's span-classification workflow, backed by OpenAI's Decisions API."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

from corral.epistemic_analysis._utils import (
    iter_windows,
    normalize_whitespace,
    sha1_hex,
)
from corral.epistemic_analysis.annotators.base import AnnotationBackend
from corral.epistemic_analysis.budget import (
    RequestBudgetExceeded,
    estimate_request,
    is_context_error,
    numeric_usage,
    record_request,
)
from corral.epistemic_analysis.context import (
    SUBMIT_ANSWER_ACTION,
    current_context,
    message_support,
    observation_indices,
    view_instructions,
)
from corral.epistemic_analysis.epistemic_prompts import (
    EDGE_ONTOLOGY,
    NODE_ONTOLOGY,
    NODE_RULES,
    jev_duplicate_question,
    jev_edge_question,
    jev_edge_support_question,
    jev_node_question,
    trace_check_criteria,
)
from corral.epistemic_analysis.graph import coerce_edge, coerce_node, is_runtime_notice
from corral.epistemic_analysis.ontology import (
    ALLOWED_EDGE_TYPE_COMBOS,
    EDGE_RELATIONS,
    NODE_TYPES,
    PASS_A_INSTRUCTIONS,
    PASS_A_SYSTEM,
    PASS_B_INSTRUCTIONS,
)
from corral.epistemic_analysis.trace_checks import (
    TRACE_CHECK_DEFINITIONS,
    trace_check_prompt,
    trace_check_state,
)

from .base import record_model

DEFAULT_ENDPOINT = "https://api.openai.com/v1/decisions"
MAX_QUESTIONS = 200
MAX_CHOICES = 255
# Edge-support decisions offer each message plus a final "none" choice.
MAX_WINDOW = MAX_CHOICES - 1
TAGGED_UNIT = re.compile(
    r"<action>.*?</action>(?:\s*<action_input>.*?</action_input>)?"
    r"|<final_answer>.*?</final_answer>",
    re.DOTALL,
)


class JevError(RuntimeError):
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
        raise JevError("Jev response is missing the answers map")
    for key, question in questions.items():
        answer = answers.get(key)
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise JevError(f"Missing or mistyped Jev answer: {key}")
        if question["type"] == "noul":
            if not _probability(answer.get("noul")):
                raise JevError(f"Invalid Noul probability: {key}")
        else:
            probabilities = answer.get("probabilities")
            choice = answer.get("choice")
            if (
                not isinstance(choice, str)
                or choice not in question["criteria"]
                or not _probability(answer.get("confidence"))
                or not isinstance(probabilities, dict)
                or set(probabilities) != set(question["criteria"])
                or not all(_probability(p) for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.01)
            ):
                raise JevError(f"Invalid Choice answer: {key}")
    return {key: answers[key] for key in questions}


def _decisions_request(model, state, questions):
    """Translate the wire format without changing any prompts or criteria."""
    converted = []
    for name, question in questions.items():
        item = {
            "name": name,
            "instructions": question.get("instructions", ""),
            "type": "predicate" if question["type"] == "noul" else question["type"],
        }
        if question["type"] == "choice":
            item["choices"] = [
                {"value": value, "description": description}
                for value, description in question["criteria"].items()
            ]
            if not 2 <= len(item["choices"]) <= MAX_CHOICES:
                raise JevError(f"Decisions requires 2-{MAX_CHOICES} choices: {name}")
        converted.append(item)
    return {
        "model": model,
        "input": json.dumps(state, ensure_ascii=False),
        "questions": converted,
    }


def _decisions_answers(payload, questions):
    """Validate Decisions responses, then restore the extraction loop's format."""
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise JevError("Expected exactly one Decisions answer per question")
    converted = {}
    for (name, question), answer in zip(questions.items(), answers, strict=True):
        expected_type = "predicate" if question["type"] == "noul" else question["type"]
        if (
            not isinstance(answer, dict)
            or answer.get("name") != name
            or answer.get("type") != expected_type
        ):
            raise JevError(f"Missing, refused, or mistyped Decisions answer: {name}")
        if expected_type == "predicate":
            converted[name] = {"type": "noul", "noul": answer.get("probability")}
            continue
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, list):
            raise JevError(f"Invalid Decisions probabilities: {name}")
        distribution = {}
        for entry in probabilities:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("value"), str)
                or entry["value"] not in question["criteria"]
                or entry["value"] in distribution
            ):
                raise JevError(f"Invalid Decisions probabilities: {name}")
            distribution[entry["value"]] = entry.get("probability")
        converted[name] = {
            "type": "choice",
            "choice": answer.get("choice"),
            "confidence": answer.get("confidence"),
            "probabilities": distribution,
        }
    return validate_answers({"answers": converted}, questions)


class JevClient:
    """Decisions transport retaining Jev's batching, retries, and graph workflow."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        batch_size: int = 32,
        timeout_s: float = 120,
        max_retries: int = 5,
        max_request_tokens: int = 0,
    ):
        if batch_size < 1 or max_retries < 1:
            raise ValueError("batch_size and max_retries must be positive")
        self.api_key = (
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        )
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_request_tokens = max_request_tokens

    def _post(self, body: dict[str, Any]) -> Any:
        if not self.api_key:
            raise JevError("Set OPENAI_API_KEY to run the Jev workflow with Decisions")
        request = Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            return json.load(response)

    async def evaluate(
        self,
        model: str,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        *,
        single_shot: bool = False,
    ) -> dict[str, dict[str, Any]]:
        if questions and not self.api_key:
            raise JevError("Set OPENAI_API_KEY to run the Jev workflow with Decisions")
        answers: dict[str, dict[str, Any]] = {}
        items = list(questions.items())
        if single_shot and len(items) > MAX_QUESTIONS:
            raise JevError("Too many questions for a single-shot Decisions request")
        context = current_context()
        budget = (
            context.config.max_request_tokens if context else self.max_request_tokens
        )

        async def evaluate_batch(batch):
            body = _decisions_request(model, state, batch)

            def record_exchange(**result):
                if context is not None:
                    context.requests.append(
                        copy.deepcopy(
                            {
                                "request": body,
                                "effective_tool_output_max_chars": context.view.effective_cap,
                                **result,
                            }
                        )
                    )

            estimated, estimator = estimate_request(body)
            limit = budget
            diagnostic = {
                "backend": "jev",
                "api": "openai-decisions",
                "estimated_tokens": estimated,
                "is_estimate": True,
                "estimator": estimator,
                "budget_tokens": limit,
                "state_chars": len(json.dumps(state, ensure_ascii=False)),
                "request_chars": len(json.dumps(body, ensure_ascii=False)),
                "question_count": len(batch),
            }

            async def split_or_raise(reason):
                record_request(
                    {
                        **diagnostic,
                        "outcome": "split"
                        if len(batch) > 1 and not single_shot
                        else "oversized",
                        "reason": reason,
                    }
                )
                if len(batch) == 1 or single_shot:
                    raise RequestBudgetExceeded({**diagnostic, "reason": reason})
                entries = list(batch.items())
                middle = len(entries) // 2
                await evaluate_batch(dict(entries[:middle]))
                await evaluate_batch(dict(entries[middle:]))

            if limit and estimated > limit:
                await split_or_raise("request_budget")
                return
            for attempt in range(self.max_retries):
                try:
                    payload = await asyncio.to_thread(self._post, body)
                    record_exchange(attempt=attempt + 1, response=payload)
                    checked = _decisions_answers(payload, batch)
                    record_model(payload.get("model"))
                    for answer in checked.values():
                        answer["model"] = payload.get("model", model)
                    answers.update(checked)
                    record_request(
                        {
                            **diagnostic,
                            "outcome": "complete",
                            "attempt": attempt + 1,
                            "usage": numeric_usage(payload.get("usage")),
                        }
                    )
                    return
                except HTTPError as exc:
                    status = exc.code
                    record_exchange(attempt=attempt + 1, http_status=status)
                    try:
                        error_body = exc.read(65536).decode("utf-8", errors="replace")
                    finally:
                        exc.close()
                    if status in {400, 413, 422} and is_context_error(error_body):
                        await split_or_raise("max_tokens_exceeded")
                        return
                    record_request(
                        {
                            **diagnostic,
                            "outcome": "http_error",
                            "http_status": status,
                            "attempt": attempt + 1,
                        }
                    )
                    if status != 429 and status < 500:
                        raise JevError(
                            f"Decisions request failed (HTTP {status})"
                        ) from exc
                    error = f"HTTP {status}"
                except (URLError, TimeoutError, OSError) as exc:
                    error = type(exc).__name__
                    record_exchange(attempt=attempt + 1, error=error)
                except (ValueError, JevError) as exc:
                    error = str(exc)
                if attempt + 1 == self.max_retries:
                    raise JevError(
                        f"Decisions request failed after {self.max_retries} attempts: {error}"
                    )
                logger.warning(f"Decisions request retry {attempt + 1}: {error}")
                await asyncio.sleep(min(2 ** (attempt + 1), 30))

        batch_size = (
            max(1, len(items)) if single_shot else min(self.batch_size, MAX_QUESTIONS)
        )
        for offset in range(0, len(items), batch_size):
            await evaluate_batch(dict(items[offset : offset + batch_size]))
        return answers


@dataclass(frozen=True)
class Span:
    msg_idx: int
    offset: int
    quote: str
    kind: str

    @property
    def key(self) -> str:
        return f"m{self.msg_idx}_s{self.offset}"

    @property
    def support(self) -> list[dict[str, Any]]:
        return [
            {
                "msg_idx": self.msg_idx,
                "quote": self.quote,
                "start": self.offset,
                "end": self.offset + len(self.quote),
            }
        ]


def candidate_spans(messages: list[dict[str, Any]], max_chars: int) -> list[Span]:
    """Split prose into exact substrings; preserve actions and their arguments."""
    spans = []

    def add_prose(idx: int, text: str, offset: int, kind: str = "prose") -> None:
        for match in re.finditer(r"\S[\s\S]*?(?=\n|(?<=[.!?])\s+|$)", text):
            quote = match.group()
            for start in range(0, len(quote), max_chars):
                part = quote[start : start + max_chars]
                if part.strip():
                    spans.append(Span(idx, offset + match.start() + start, part, kind))

    observations = observation_indices(messages)
    task_idx = next(
        (
            i
            for i, m in enumerate(messages)
            if m.get("role", "").lower() == "user" and i not in observations
        ),
        None,
    )
    for idx, message in enumerate(messages):
        content = str(message.get("content", "") or "")
        if idx in observations:
            spans.append(Span(idx, 0, content, "observation"))
            continue
        if idx == task_idx and content.strip():
            spans.append(Span(idx, 0, content, "task"))
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


def window_state(
    messages: list[dict[str, Any]], start: int, end: int
) -> dict[str, Any]:
    state = {
        "messages": {f"m{i}": {"msg_idx": i, **messages[i]} for i in range(start, end)},
        "window": {"start": start, "end": end},
        "node_ontology": NODE_ONTOLOGY,
    }
    if view_instructions():
        state["annotation_view_instructions"] = view_instructions()
    return state


class JevAnnotator:
    def __init__(
        self,
        client: JevClient | None,
        min_confidence: float = 0.5,
        max_span_chars: int = 1200,
    ):
        self.client = client
        self.min_confidence = min_confidence
        self.max_span_chars = max_span_chars

    def backend(self) -> AnnotationBackend:
        return AnnotationBackend(
            extract_nodes=self.extract_nodes,
            extract_edges=self.extract_edges,
            extract_trace_checks=self.extract_trace_checks,
            provenance={
                "annotator": "jev",
                "api": "openai-decisions",
                "candidate_extractor": "multilabel_spans_v4",
                "node_postprocessing": "test_precedence_neutral_fallback_v1",
                "node_decisions": "independent_choices_v1",
                "min_confidence": self.min_confidence,
                "max_span_chars": self.max_span_chars,
                "endpoint": getattr(self.client, "endpoint", DEFAULT_ENDPOINT),
                "batch_size": getattr(self.client, "batch_size", 32),
                "timeout_s": getattr(self.client, "timeout_s", 120),
                "max_retries": getattr(self.client, "max_retries", 5),
                "decision_prompt_sha1": sha1_hex(
                    PASS_A_SYSTEM
                    + NODE_ONTOLOGY
                    + NODE_RULES
                    + EDGE_ONTOLOGY
                    + "".join(
                        trace_check_prompt(name) for name in TRACE_CHECK_DEFINITIONS
                    )
                ),
                "ontology_sha1": sha1_hex(PASS_A_INSTRUCTIONS + PASS_B_INSTRUCTIONS),
            },
        )

    async def extract_trace_checks(self, trace, nodes, edges, model):
        if self.client is None:
            raise JevError("Jev client is required for annotation")
        checks = []
        for name in TRACE_CHECK_DEFINITIONS:
            question = {
                "type": "choice",
                "instructions": trace_check_prompt(name),
                "criteria": trace_check_criteria(name),
            }
            answers = await self.client.evaluate(
                model,
                trace_check_state(trace, nodes, edges, name),
                {name: question},
                single_shot=True,
            )
            answer = answers[name]
            status = (
                answer["choice"]
                if answer["confidence"] >= self.min_confidence
                else "uncertain"
            )
            checks.append(
                {"name": name, "status": status, "support": [], "jev": answer}
            )
        return checks

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
            raise JevError("Jev client is required for annotation")
        nodes: list[dict[str, Any]] = []
        warnings: list[str] = []
        seen: set[str] = set()
        next_id = 1

        def review(message):
            warnings.append(message)
            context = current_context()
            if context is not None:
                context.review_issues.append(message)

        for start, end in iter_windows(len(messages), window, overlap):
            current = [
                s for s in spans if start <= s.msg_idx < end and s.key not in seen
            ]
            if not current:
                continue
            state = window_state(messages, start, end)
            state["annotation_rules"] = PASS_A_SYSTEM
            state["candidates"] = {
                s.key: {"msg_idx": s.msg_idx, "kind": s.kind, "quote": s.quote}
                for s in current
            }
            questions = {}
            for span in current:
                for node_type in NODE_TYPES:
                    questions[f"{span.key}_{node_type}"] = jev_node_question(
                        span.key, node_type
                    )
            answers = await self.client.evaluate(model, state, questions)
            extracted = []
            for span in current:
                seen.add(span.key)
                selected = {}
                submission = span.kind == "action" and SUBMIT_ANSWER_ACTION.match(
                    span.quote
                )
                test_over_neutral = (
                    span.kind == "action"
                    and not submission
                    and all(
                        answers[f"{span.key}_{kind}"]["choice"] == "present"
                        for kind in ("T", "N")
                    )
                )
                for node_type in NODE_TYPES:
                    answer = answers[f"{span.key}_{node_type}"]
                    if answer["choice"] != "present":
                        continue
                    if answer["confidence"] < self.min_confidence and not (
                        test_over_neutral and node_type == "T"
                    ):
                        warnings.append(
                            f"Low node confidence; omitted {span.key} ({node_type})."
                        )
                        continue
                    if (
                        (span.kind in {"observation", "task"} and node_type != "E")
                        or (
                            node_type == "E"
                            and span.kind not in {"observation", "task"}
                        )
                        or (node_type == "N" and span.kind != "action")
                        or (
                            node_type == "F" and span.kind != "final" and not submission
                        )
                    ):
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
            if len(extracted) > max_nodes_per_window:
                review(
                    f"PassA[{start}-{end}] Node cap reached; omitted {len(extracted) - max_nodes_per_window} nodes."
                )
            nodes.extend(extracted[:max_nodes_per_window])
        for idx in {
            s.msg_idx
            for s in spans
            if s.kind == "final"
            or (s.kind == "action" and SUBMIT_ANSWER_ACTION.match(s.quote))
        }:
            if not any(n["time"] == idx and n["type"] == "F" for n in nodes):
                review(f"Final-answer message {idx} has no F node.")
        nodes.sort(key=lambda n: n["time"])
        return nodes, warnings

    async def _merge_duplicates(self, model, state, nodes):
        """Merge repeated instances within a message, preserving distinct claims."""
        questions, pairs = {}, {}
        for i, first in enumerate(nodes):
            for second in nodes[i + 1 :]:
                if (first["time"], first["type"]) != (second["time"], second["type"]):
                    continue
                key = f"same_{first['node_id']}_{second['node_id']}"
                pairs[key] = (first, second)
                if first["type"] == "F":
                    continue  # One submission, with potentially several support spans.
                questions[key] = jev_duplicate_question(
                    first["node_id"], second["node_id"], first["type"]
                )
        state = {**state, "nodes": {n["node_id"]: n for n in nodes}}
        answers = (
            await self.client.evaluate(model, state, questions) if questions else {}
        )
        representatives = {n["node_id"]: n for n in nodes}
        warnings = []
        for key, (first, second) in pairs.items():
            if first["type"] != "F":
                answer = answers[key]
                if answer["choice"] != "same":
                    continue
                if answer["confidence"] < self.min_confidence:
                    warnings.append(
                        f"Uncertain duplicate {key}; retained both instances."
                    )
                    continue
            kept, removed = (
                representatives[first["node_id"]],
                representatives[second["node_id"]],
            )
            if kept is removed:
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
            raise JevError("Jev client is required for annotation")
        edges = []
        warnings = []
        seen = set()
        for start, end in iter_windows(len(messages), window, overlap):
            current = [n for n in nodes if start <= n["time"] < end]
            state = window_state(messages, start, end)
            state["edge_ontology"] = EDGE_ONTOLOGY
            state["nodes"] = {
                n["node_id"]: {k: n[k] for k in ("type", "time", "text", "support")}
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
                        if (
                            rel,
                            src["type"],
                            dst["type"],
                        ) not in ALLOWED_EDGE_TYPE_COMBOS:
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
                    questions[key] = jev_edge_question(pair[0], pair[1], relations)
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
                support_questions[key] = jev_edge_support_question(
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
        return edges, warnings
