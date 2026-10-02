"""Deterministic annotation excerpts and exact mappings to normalized sources."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import AnnotationConfig, Message, NormalizedTrace, content_hash

VIEW_VERSION = "head-tail-submit-answer-v2"
SUBMIT_ANSWER_ACTION = re.compile(r"<action>\s*submit_answer\s*</action>")
VIEW_INSTRUCTIONS = (
    "[ANNOTATION VIEW: ...] marks omitted, unavailable context. Excerpts of JSON "
    "and tables are incomplete text. Cite only contiguous verbatim retained text, "
    "never the marker or across its gap. Missing details in an excerpt do not "
    "prove their absence from the original output."
)


def is_tool_result(message: dict, source: dict | None = None) -> bool:
    source = source or {}
    return (
        message.get("role", "").lower() in {"tool", "function"}
        or str(source.get("role", source.get("original_role", ""))).lower()
        in {"tool", "function"}
        or bool(source.get("tool_call_id"))
        or message.get("content", "").startswith("Observation:")
    )


@dataclass(frozen=True)
class RetainedRange:
    """Half-open Unicode character offsets, in original and rendered text."""

    original_start: int
    original_end: int
    view_start: int
    view_end: int


@dataclass
class AnnotationView:
    messages: list[Message]
    mappings: list[dict[str, Any]]
    requested_cap: int
    effective_cap: int
    policy: str = "head-tail"
    policy_version: str = VIEW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return content_hash(self.to_dict())

    @property
    def coverage(self) -> dict[str, Any]:
        outputs = [m for m in self.mappings if m["is_tool_result"]]
        omitted = sum(m["omitted_chars"] for m in outputs)
        original = sum(m["original_length"] for m in outputs)
        return {
            "clipped_outputs": sum(m["omitted_chars"] > 0 for m in outputs),
            "original_tool_chars": original,
            "retained_tool_chars": original - omitted,
            "omitted_tool_chars": omitted,
            "effective_tool_output_max_chars": self.effective_cap,
            "reduced_context": omitted > 0,
        }

    def support(self, idx: int) -> list[dict[str, Any]]:
        content = self.messages[idx]["content"]
        return [
            {
                "msg_idx": idx,
                "quote": content[r["view_start"] : r["view_end"]],
                "start": r["original_start"],
                "end": r["original_end"],
            }
            for r in self.mappings[idx]["retained_ranges"]
            if r["view_end"] > r["view_start"]
        ]

    def map_support(self, support: list[dict], originals: list[Message]) -> list[dict]:
        """Reject hidden text/metadata even if it appears in the original trace."""
        mapped = []
        for item in support:
            idx, quote = item.get("msg_idx"), item.get("quote")
            if (
                not isinstance(idx, int)
                or isinstance(idx, bool)
                or not 0 <= idx < len(self.messages)
                or not isinstance(quote, str)
                or not quote
            ):
                raise ValueError("Invalid annotation support")
            for span in self.support(idx):
                offset = span["quote"].find(quote)
                if "start" in item:
                    offset = item["start"] - span["start"]
                if (
                    offset >= 0
                    and span["quote"][offset : offset + len(quote)] == quote
                    and offset + len(quote) <= len(span["quote"])
                ):
                    start = span["start"] + offset
                    end = start + len(quote)
                    if "end" in item and item["end"] != end:
                        continue
                    if originals[idx]["content"][start:end] != quote:
                        continue
                    mapped.append({**item, "start": start, "end": end})
                    break
            else:
                raise ValueError(
                    f"Support is outside a contiguous retained range in message {idx}"
                )
        return mapped


def cap_sequence(requested: int) -> list[int]:
    """At most two reductions; disabled clipping never opts in implicitly."""
    if not requested:
        return [0]
    return [requested, *[cap for cap in (6000, 3000, 1500) if cap < requested][:2]]


def build_annotation_view(
    trace: NormalizedTrace, config: AnnotationConfig, effective_cap: int | None = None
) -> AnnotationView:
    cap = config.tool_output_max_chars if effective_cap is None else effective_cap
    if cap not in cap_sequence(config.tool_output_max_chars):
        raise ValueError("Effective cap is outside the configured fallback sequence")
    messages, mappings = [], []
    for idx, (message, source) in enumerate(
        zip(trace.messages, trace.message_sources, strict=False)
    ):
        original = message["content"]
        tool = is_tool_result(message, source)
        ranges = [RetainedRange(0, len(original), 0, len(original))]
        content, omitted = original, 0
        if tool and cap and len(original) > cap:
            head = cap - cap // 4
            tail = cap // 4
            omitted = len(original) - cap
            marker = f"\n[ANNOTATION VIEW: {omitted:,} original characters omitted]\n"
            content = original[:head] + marker + original[-tail:]
            ranges = [
                RetainedRange(0, head, 0, head),
                RetainedRange(
                    len(original) - tail,
                    len(original),
                    head + len(marker),
                    len(content),
                ),
            ]
        messages.append({**message, "content": content})
        mappings.append(
            {
                "msg_idx": idx,
                "source": dict(source),
                "is_tool_result": tool,
                "tool_identity": {
                    k: source.get(k, message.get(k))
                    for k in ("tool_call_id", "tool_name", "name")
                    if source.get(k, message.get(k)) is not None
                },
                "original_content_hash": content_hash(original),
                "original_length": len(original),
                "retained_ranges": [asdict(r) for r in ranges],
                "omitted_chars": omitted,
            }
        )
        if message["role"].lower() == "assistant" and SUBMIT_ANSWER_ACTION.search(
            original
        ):
            break
    return AnnotationView(
        messages,
        mappings,
        config.tool_output_max_chars,
        cap,
        config.tool_output_truncation,
    )


@dataclass
class AnnotationContext:
    view: AnnotationView
    config: AnnotationConfig
    diagnostics: list[dict[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)
    review_issues: list[str] = field(default_factory=list)


_context: ContextVar[AnnotationContext | None] = ContextVar(
    "annotation_context", default=None
)


def current_context() -> AnnotationContext | None:
    return _context.get()


@contextmanager
def annotation_context(view, config, diagnostics, requests=None, review_issues=None):
    token = _context.set(
        AnnotationContext(
            view,
            config,
            diagnostics,
            requests if requests is not None else [],
            review_issues if review_issues is not None else [],
        )
    )
    try:
        yield
    finally:
        _context.reset(token)


def message_support(messages, idx):
    context = current_context()
    if context is not None and messages is context.view.messages:
        return context.view.support(idx)
    return [{"msg_idx": idx, "quote": messages[idx]["content"]}]


def observation_indices(messages):
    context = current_context()
    if context is not None and messages is context.view.messages:
        return {m["msg_idx"] for m in context.view.mappings if m["is_tool_result"]}
    return {i for i, message in enumerate(messages) if is_tool_result(message)}


def view_instructions() -> str:
    context = current_context()
    return (
        VIEW_INSTRUCTIONS
        if context and context.view.coverage["reduced_context"]
        else ""
    )
