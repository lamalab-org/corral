"""Validate direct inputs and text lists before contacting any annotator."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from corral.epistemic_analysis.decision_applicability import SAVED_FIELDS
from corral.epistemic_analysis.schema import (
    GroupingMetadata,
    NormalizedTrace,
    SourceReference,
    content_hash,
)

from .langfuse import LangfuseLocator, parse_langfuse_url


@dataclass(frozen=True)
class InputEntry:
    locator: str
    kind: str
    path: Path | None = None
    remote: LangfuseLocator | None = None
    list_path: str | None = None
    line_number: int | None = None

    @property
    def diagnostic(self) -> str:
        return (
            f"{self.list_path}:{self.line_number}" if self.list_path else self.locator
        )


def resolve_inputs(
    inputs: list[str] | None = None,
    input_list: str | None = None,
    source: str | None = None,
) -> list[InputEntry]:
    if bool(inputs) == bool(input_list):
        raise ValueError("Supply either --input (repeatable) or --input-list")
    if source not in {None, "path", "langfuse"}:
        raise ValueError("source must be path or langfuse")
    list_path = Path(input_list).expanduser().resolve() if input_list else None
    base = list_path.parent if list_path else Path.cwd()
    values = (
        list(enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1))
        if list_path
        else [(None, value) for value in inputs]
    )
    entries, errors = [], []
    for number, raw in values:
        locator = raw.strip() if list_path else raw
        if list_path and (not locator or locator.startswith("#")):
            continue
        where = f"{list_path}:{number}" if list_path else locator
        try:
            if not locator:
                raise ValueError("Empty input")
            is_url = bool(urlsplit(locator).scheme)
            kind = "langfuse" if is_url else "path"
            if source and kind != source:
                raise ValueError(f"Expected a {source} input")
            remote = parse_langfuse_url(locator) if is_url else None
            path = None if is_url else (base / Path(locator).expanduser()).resolve()
            if path and not path.exists():
                raise ValueError(f"Path does not exist: {path}")
            if path and path.is_file() and path.suffix.lower() != ".json":
                raise ValueError(
                    "Expected a JSON trace; use --input-list for text lists"
                )
            entries.append(
                InputEntry(
                    locator,
                    kind,
                    path,
                    remote,
                    str(list_path) if list_path else None,
                    number,
                )
            )
        except (ValueError, OSError) as exc:
            errors.append(f"{where}: {exc}")
    if errors:
        raise ValueError("Invalid analysis inputs:\n" + "\n".join(errors))
    if not entries:
        raise ValueError("No analysis inputs selected")
    return entries


CONTEXT_FIELDS = frozenset(
    {
        "recorded_initial_instructions",
        "recovered_task_context",
        "public_task_context",
        "configured_tools",
        "capture_limitations",
        "source_evidence",
    }
)


@dataclass
class CheckInput:
    trace: NormalizedTrace
    context: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.trace.messages:
            raise ValueError("A check input requires saved messages")
        # Do not accept entire task configurations or benchmark rows as evidence.
        if set(self.context) - CONTEXT_FIELDS:
            raise ValueError("Context must contain only explicit public context fields")
        self.context = copy.deepcopy(self.context)
        for key in ("recovered_task_context", "public_task_context"):
            value = self.context.get(key)
            if value is not None and (
                not isinstance(value, dict)
                or set(value) - {"text", "source_evidence"}
                or not isinstance(value.get("text"), str)
                or not value.get("source_evidence")
            ):
                raise ValueError(f"{key} requires public text and source evidence")

    @classmethod
    def from_document(cls, document, context=None):
        """Accept a normalized snapshot or a saved trial; never copy scores/targets."""
        if "message_sources" in document:
            trace = NormalizedTrace(
                copy.deepcopy(document["messages"]),
                SourceReference(**document["source"]),
                copy.deepcopy(document["message_sources"]),
                grouping=GroupingMetadata(**document.get("grouping", {})),
                completion_status=document.get("completion_status", "unspecified"),
                reconstruction_status=document.get(
                    "reconstruction_status", "unspecified"
                ),
            )
        else:
            messages, sources = [], []
            for i, message in enumerate(document["messages"]):
                value = message.get("content")
                text = (
                    value
                    if isinstance(value, str)
                    else json.dumps(value, ensure_ascii=False)
                )
                fields = {
                    k: message[k]
                    for k in (
                        "tool_calls",
                        "function_call",
                        "name",
                        "tool_call_id",
                        "agent_run_id",
                        "branch_id",
                    )
                    if k in message
                }
                if fields:
                    text += SAVED_FIELDS + json.dumps(fields, ensure_ascii=False)
                messages.append(
                    {"role": str(message.get("role", "unknown")), "content": text}
                )
                sources.append(
                    {
                        "index": i,
                        **{
                            k: v
                            for k, v in fields.items()
                            if k not in {"tool_calls", "function_call"}
                        },
                    }
                )
            trace = NormalizedTrace(
                messages,
                SourceReference(
                    document.get("execution_id", "saved-trace"),
                    "saved-trial",
                    content_hash(document),
                ),
                sources,
                grouping=GroupingMetadata(**document.get("grouping", {})),
                completion_status=document.get("completion_status", "unspecified"),
                reconstruction_status=document.get(
                    "reconstruction_status", "unspecified"
                ),
            )
        return cls(trace, context or {})

    def public_state(self):
        initial = []
        for i, message in enumerate(self.trace.messages):
            if message["role"] == "assistant":
                break
            if message["role"] in {"system", "user", "developer"}:
                initial.append(
                    {
                        "msg_idx": i,
                        "message_id": f"m{i}",
                        "start": 0,
                        "end": len(message["content"]),
                        **message,
                        "source": self.trace.message_sources[i],
                    }
                )
        # Exclude arbitrary SourceReference.details and grouping/score metadata from prompts.
        return {
            "context": self.context,
            "initial_instructions": initial,
            "instruction_authority": "Recorded runtime instructions take precedence over recovered task configuration.",
            "capture": {
                "completion_status": self.trace.completion_status,
                "reconstruction_status": self.trace.reconstruction_status,
                "limitations": self.context.get("capture_limitations", []),
            },
            "source_hashes": {
                "saved_source": self.trace.source.content_hash,
                "messages": content_hash(self.trace.messages),
                "mappings": content_hash(self.trace.message_sources),
                "context": content_hash(self.context),
            },
        }

    def snapshot(self):
        return {
            "grouping": asdict(self.trace.grouping),
            "messages": self.trace.messages,
            "message_sources": self.trace.message_sources,
            "source": asdict(self.trace.source),
            "completion_status": self.trace.completion_status,
            "reconstruction_status": self.trace.reconstruction_status,
        }

    @property
    def fingerprint(self):
        return content_hash({"trace": self.snapshot(), "context": self.public_state()})
