"""Langfuse downloads and loss-aware conversation reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import unquote, urlsplit


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _parse(value: Any) -> Any:
    # Observations v2 returns input/output as JSON-encoded strings.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if not isinstance(block, dict) or block.get("type") not in {
                "text",
                "output_text",
                "thinking",
                "reasoning",
            }:
                raise ValueError(
                    "Non-text content blocks cannot be annotated faithfully"
                )
            text = block.get("text", block.get("thinking"))
            if not isinstance(text, str):
                raise ValueError("Content block is missing its recorded text")
            parts.append(text)
        return "\n".join(parts)
    return _json(value)


def _tool_text(value: Any) -> str:
    # Scientific tool results can be arbitrary JSON arrays, not content blocks.
    return value if isinstance(value, str) else _json(value)


def _messages(value: Any) -> list[dict[str, Any]]:
    value = _parse(value)
    if isinstance(value, dict) and "messages" in value:
        value = value["messages"]
    if isinstance(value, dict) and "role" in value:
        value = [value]
    if not isinstance(value, list) or not all(
        isinstance(item, dict) and isinstance(item.get("role"), str) for item in value
    ):
        raise ValueError("Expected Corral chat messages in generation input/output")
    return value


def _units(message: dict[str, Any]) -> list[tuple[str, str]]:
    """Compare histories without depending on how continuation calls are grouped."""
    units = []
    role = message["role"]
    for key in ("reasoning_content", "content"):
        value = message.get(key)
        content = (
            _tool_text(value) if role == "tool" and key == "content" else _text(value)
        )
        if content:
            units.append((role, content))
    for call in message.get("tool_calls") or []:
        function = call["function"]
        units.append(
            (
                "action",
                _json(
                    {
                        "id": call.get("id"),
                        "name": function["name"],
                        "arguments": _parse(function.get("arguments", {})),
                    }
                ),
            )
        )
    if role == "tool":
        units.append(("tool_result", str(message.get("tool_call_id"))))
    return units


def _render(message: dict[str, Any]) -> dict[str, str]:
    role = message["role"]
    if role == "tool":
        content = _tool_text(message.get("content"))
        content = (
            content if content.startswith("Observation:") else f"Observation: {content}"
        )
        return {"role": "user", "content": content}
    parts = [_text(message.get("reasoning_content")), _text(message.get("content"))]
    content = "\n".join(part for part in parts if part)
    # ReAct messages already contain the same actions as their structured calls.
    tagged = re.findall(r"<action>(.*?)</action>", content, re.DOTALL)
    calls = message.get("tool_calls") or []
    if tagged:
        if calls and [name.strip() for name in tagged] != [
            call["function"]["name"] for call in calls
        ]:
            raise ValueError("ReAct tags and structured tool calls disagree")
    else:
        for call in calls:
            function = call["function"]
            arguments = function.get("arguments", {})
            arguments = arguments if isinstance(arguments, str) else _json(arguments)
            content += (
                f"\n<action>{function['name']}</action>"
                f"<action_input>{arguments}</action_input>"
            )
    return {"role": role, "content": content.lstrip("\n")}


def reconstruct_trace(
    observations: list[dict[str, Any]],
    session_id: str,
) -> dict[str, Any]:
    """Use the last recorded history, validating coverage of earlier generations.

    Corral exports cumulative generation inputs. Concatenating those inputs
    would repeatedly annotate the same conversation. Diverging agent histories
    are rejected instead of silently choosing one branch.
    """
    trace_ids = {row["traceId"] for row in observations}
    if len(trace_ids) != 1:
        raise ValueError("Expected observations from exactly one trace")
    trace_id = next(iter(trace_ids))
    ordered = sorted(
        observations,
        key=lambda row: (
            row.get("endTime") or row["startTime"],
            row["startTime"],
            row["id"],
        ),
    )
    roots = [row for row in ordered if not row.get("parentObservationId")]
    if len(roots) != 1 or not roots[0].get("endTime"):
        raise ValueError(
            f"Trace {trace_id}: expected one completed task root; retry after ingestion finishes"
        )
    generations = [row for row in ordered if row["type"].upper() == "GENERATION"]
    if not generations:
        raise ValueError(f"Trace {trace_id}: no recorded model generations")
    if any(not row.get("endTime") for row in generations):
        raise ValueError(f"Trace {trace_id}: a generation is still open")
    latest = generations[-1]
    inputs = _messages(latest.get("input"))
    outputs = _messages(latest.get("output"))
    messages = [*inputs, *outputs]
    sources = [
        {"observation_id": latest["id"], "field": field, "index": index}
        for field, items in (("input", inputs), ("output", outputs))
        for index in range(len(items))
    ]
    covered = Counter(unit for message in messages for unit in _units(message))
    for row in generations:
        history = [*_messages(row.get("input")), *_messages(row.get("output"))]
        required = Counter(unit for message in history for unit in _units(message))
        if required - covered:
            raise ValueError(
                f"Trace {trace_id}: generation {row['id']} is absent from the final history; "
                "branched or truncated conversations need separate reconstruction"
            )
    # A run may finish/fail after a tool result, without another model turn.
    tool_ids = {m.get("tool_call_id") for m in messages if m["role"] == "tool"}
    for row in ordered:
        if row["type"].upper() != "TOOL":
            continue
        for index, message in enumerate(_messages(row.get("output"))):
            if message["role"] != "tool" or not message.get("tool_call_id"):
                raise ValueError(f"Trace {trace_id}: tool output lacks a tool_call_id")
            if message["tool_call_id"] in tool_ids:
                continue
            if row["startTime"] < latest["startTime"]:
                raise ValueError(
                    f"Trace {trace_id}: an earlier tool result is missing from model history"
                )
            tool_ids.add(message["tool_call_id"])
            messages.append(message)
            sources.append(
                {"observation_id": row["id"], "field": "output", "index": index}
            )
    for message, source in zip(messages, sources, strict=False):
        if message["role"] == "tool":
            source.update(
                {
                    "original_role": "tool",
                    **{k: message[k] for k in ("tool_call_id", "name") if k in message},
                }
            )
    rendered = [_render(message) for message in messages]
    root = roots[0]
    outcome = _parse(root.get("output"))
    if isinstance(outcome, dict) and outcome.get("status") == "submitted":
        answer = _text(outcome.get("answer"))
        final = f"<final_answer>{answer}</final_answer>"
        if answer and not any(final in message["content"] for message in rendered):
            rendered.append({"role": "assistant", "content": final})
            sources.append({"observation_id": root["id"], "field": "output.answer"})
    return {
        "messages": rendered,
        "source": {
            "provider": "langfuse",
            "session_id": session_id,
            "trace_id": trace_id,
            "observation_ids": sorted(row["id"] for row in observations),
            "message_sources": sources,
            "messages_sha256": _hash(rendered),
            "conversion_version": "corral-langfuse-v1",
        },
    }


def fetch_observations(
    api: Any,
    session_id: str | None,
    from_time: str | None = None,
    to_time: str | None = None,
    *,
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    if (session_id is None) == (trace_id is None):
        raise ValueError("Select exactly one session or trace")
    column, identity = (
        ("traceId", trace_id) if trace_id is not None else ("sessionId", session_id)
    )
    filters = [{"type": "string", "column": column, "operator": "=", "value": identity}]
    bounds = []
    for value, operator in ((from_time, ">="), (to_time, "<")):
        if value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Time bounds must include a timezone")
            bounds.append(parsed)
            filters.append(
                {
                    "type": "datetime",
                    "column": "startTime",
                    "operator": operator,
                    "value": value,
                }
            )
    if len(bounds) == 2 and bounds[0] >= bounds[1]:
        raise ValueError("from_time must precede to_time")
    cursor = None
    cursors = set()
    rows = {}
    while True:
        page = api.observations.get_many(
            fields="core,basic,io,metadata,model,trace_context",
            filter=_json(filters),
            limit=100,
            cursor=cursor,
            request_options={"max_retries": 3},
        ).model_dump(mode="json", by_alias=True)
        for row in page["data"]:
            if row.get(column) != identity:
                raise ValueError(
                    f"Langfuse returned an observation from another {'trace' if trace_id else 'session'}"
                )
            rows[(row["traceId"], row["id"])] = row
        cursor = page["meta"].get("cursor")
        if not cursor:
            return sorted(
                rows.values(),
                key=lambda row: (row["traceId"], row["startTime"], row["id"]),
            )
        if cursor in cursors:
            raise RuntimeError("Langfuse repeated a pagination cursor")
        cursors.add(cursor)


@dataclass(frozen=True)
class LangfuseLocator:
    base_url: str
    project_id: str
    kind: str
    identity: str


def parse_langfuse_url(value: str) -> LangfuseLocator:
    """Accept /project/PROJECT/{sessions,traces}/ID, with query/fragment retained
    by the caller as provenance. Encoded IDs are decoded exactly once.
    """
    parsed = urlsplit(value)
    match = re.fullmatch(
        r"(.*)/project/([^/]+)/(sessions|traces)/([^/]+)/?", parsed.path
    )
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or not match
    ):
        raise ValueError(
            "Unsupported Langfuse URL; expected https://HOST/project/PROJECT/sessions/ID or /traces/ID"
        )
    prefix, project, kind, identity = match.groups()
    if any(part in {".", "..", ""} for part in (unquote(project), unquote(identity))):
        raise ValueError("Invalid Langfuse project or record ID")
    return LangfuseLocator(
        f"{parsed.scheme}://{parsed.netloc}{prefix}".rstrip("/"),
        unquote(project),
        kind,
        unquote(identity),
    )


def download_locator(locator: LangfuseLocator, from_time=None, to_time=None):
    """Validate deployment and credential project before fetching observations.

    API keys are project scoped: https://langfuse.com/docs/api-and-data-platform/features/public-api
    """
    configured = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
        "LANGFUSE_HOST", "https://cloud.langfuse.com"
    )
    if locator.base_url != configured.rstrip("/"):
        raise ValueError(
            f"Langfuse URL deployment {locator.base_url} does not match configured {configured}"
        )
    public, secret = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        raise ValueError("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    try:
        import httpx
        from langfuse.api.client import LangfuseAPI
    except ImportError as exc:
        raise RuntimeError("Install corral[langfuse] to read Langfuse inputs") from exc
    with httpx.Client(timeout=60) as http:
        api = LangfuseAPI(
            base_url=locator.base_url,
            username=public,
            password=secret,
            httpx_client=http,
        )
        projects = api.projects.get().model_dump(mode="json", by_alias=True)["data"]
        if locator.project_id not in {project["id"] for project in projects}:
            raise ValueError(
                "Langfuse URL project does not match the configured credentials"
            )
        rows = fetch_observations(
            api,
            locator.identity if locator.kind == "sessions" else None,
            from_time,
            to_time,
            trace_id=locator.identity if locator.kind == "traces" else None,
        )
        if any(row.get("projectId") != locator.project_id for row in rows):
            raise ValueError("Langfuse returned records from another project")
        return rows


def traces_from_observations(entry, observations):
    """Reconstruct each trace separately and retain raw observations for audit."""
    from collections import defaultdict

    from corral.epistemic_analysis.schema import (
        GroupingMetadata,
        SourceReference,
        content_hash,
    )

    from .files import normalize_document

    if not observations:
        raise ValueError(f"{entry.diagnostic}: no observations found")
    grouped = defaultdict(list)
    for row in observations:
        if row.get("projectId") != entry.remote.project_id:
            raise ValueError(
                f"{entry.diagnostic}: observations belong to another project"
            )
        if (
            entry.remote.kind == "sessions"
            and row.get("sessionId") != entry.remote.identity
        ):
            raise ValueError(
                f"{entry.diagnostic}: observations belong to another session"
            )
        if (
            entry.remote.kind == "traces"
            and row.get("traceId") != entry.remote.identity
        ):
            raise ValueError(
                f"{entry.diagnostic}: observations belong to another trace"
            )
        grouped[row["traceId"]].append(row)
    traces = []
    for trace_id, rows in sorted(grouped.items()):
        ordered_rows = sorted(rows, key=lambda row: (row["startTime"], row["id"]))
        generations = [
            row for row in ordered_rows if row["type"].upper() == "GENERATION"
        ]
        for key in ("agent_run_id", "branch_id"):
            identities = {
                metadata[key]
                for row in generations
                if isinstance(metadata := _parse(row.get("metadata")), dict)
                and metadata.get(key)
            }
            if len(identities) > 1:
                raise ValueError(
                    f"Trace {trace_id}: multiple {key} values require separate conversation reconstruction"
                )
        sessions = {row.get("sessionId") for row in rows}
        if len(sessions) != 1:
            raise ValueError(f"Trace {trace_id}: inconsistent session identities")
        document = reconstruct_trace(ordered_rows, next(iter(sessions)) or "")
        root = next(row for row in rows if not row.get("parentObservationId"))
        metadata = _parse(root.get("metadata")) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        generations = [row for row in rows if row["type"].upper() == "GENERATION"]
        models = {row.get("model") for row in generations if row.get("model")}
        grouping = GroupingMetadata(
            agent_model=next(iter(models)) if len(models) == 1 else None,
            environment=metadata.get("environment"),
            level=metadata.get("level"),
            task=metadata.get("task"),
            trial=metadata.get("trial"),
        )
        identity = {
            key: metadata.get(key)
            for key in ("execution_id", "commit_hash", "branch_id", "agent_run_id")
        }
        source = SourceReference(
            identity=f"{entry.remote.base_url}/project/{entry.remote.project_id}/traces/{trace_id}",
            locator=entry.locator,
            provider="langfuse",
            content_hash=content_hash(ordered_rows),
            normalization_version="corral-langfuse-v1+corral-messages-v1",
            list_path=entry.list_path,
            line_number=entry.line_number,
            **identity,
            details=document["source"],
        )
        trace = normalize_document(document, source, grouping)
        trace.completion_status = "complete"
        trace.snapshot = ordered_rows
        traces.append(trace)
    return traces
