"""Sources files for epistemic trace analysis."""

from __future__ import annotations

from pathlib import Path


def normalize_document(document, source, grouping=None):
    """Normalize message text and retain original message indices."""
    from corral.epistemic_analysis.schema import GroupingMetadata, NormalizedTrace

    if not isinstance(document, dict) or not isinstance(document.get("messages"), list):
        raise ValueError("Trace requires a messages list")
    messages, references = [], []
    original_sources = document.get(
        "message_sources", document.get("source", {}).get("message_sources", [])
    )
    for index, message in enumerate(document["messages"]):
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            import json

            content = json.dumps(content, ensure_ascii=False)
        else:
            content = "" if content is None else str(content)
        if (
            content.strip()
            == "Error: Maximum iterations reached without finding a final answer."
        ):
            continue
        messages.append({"role": str(message.get("role", "")), "content": content})
        references.append(
            {
                "index": index,
                **(
                    {"prompt_context": message["metadata"]["prompt_context"]}
                    if (message.get("metadata") or {}).get("prompt_context")
                    else {}
                ),
                **(
                    {"original_content": message["content"]}
                    if isinstance(message.get("content"), list)
                    else {}
                ),
                **{
                    k: message[k]
                    for k in ("tool_call_id", "tool_name", "name")
                    if k in message
                },
                **(original_sources[index] if index < len(original_sources) else {}),
            }
        )
    metadata = document.get("grouping", {})
    grouping = grouping or GroupingMetadata(
        **{
            k: str(v)
            for k, v in metadata.items()
            if k in GroupingMetadata.__dataclass_fields__ and v is not None
        }
    )
    trace = NormalizedTrace(
        messages,
        source,
        references,
        grouping,
        completion_status=document.get("completion_status", "unspecified"),
        reconstruction_status=document.get("reconstruction_status", "complete"),
        snapshot=document,
    )
    if "prompt_reconstruction" in document:
        from corral.epistemic_analysis.annotation_transcript import reconstruct_trace

        trace = reconstruct_trace(trace, document["prompt_reconstruction"])
    return trace


def load_path(entry):
    """Load JSON traces from a file or directory, using explicit grouping metadata."""
    import hashlib
    import json
    from dataclasses import replace

    from corral.epistemic_analysis.schema import SourceReference

    origin = entry.path
    paths = (
        [origin]
        if origin.is_file()
        else [
            path
            for path in sorted(origin.rglob("*.json"))
            if not path.name.endswith(".annotated.json")
            and path.name != "manifest.json"
            and not {
                "snapshots",
                "annotations",
                "analysis",
                "cache",
                "annotation_views",
                "annotation_requests",
            }.intersection(path.relative_to(origin).parts[:-1])
        ]
    )
    if not paths:
        raise ValueError(f"{entry.diagnostic}: no supported trace JSON found")
    traces = []
    for path in paths:
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
            if (
                isinstance(document, dict)
                and document.get("format") == "corral-epistemic-trace-v1"
            ):
                saved = document["source"]
                reference = SourceReference(
                    **{**saved, "snapshots": tuple(saved.get("snapshots", ()))}
                )
                reference = replace(
                    reference,
                    list_path=entry.list_path,
                    line_number=entry.line_number,
                    details={
                        **reference.details,
                        "loaded_from": {
                            "locator": entry.locator,
                            "list_path": entry.list_path,
                            "line_number": entry.line_number,
                        },
                    },
                )
                trace = normalize_document(document, reference)
                if "prompt_reconstruction" not in document:
                    trace.message_sources = document["message_sources"]
                trace.__post_init__()
                # Preserve the saved source, not a snapshot of a snapshot.
                if reference.snapshots and Path(reference.snapshots[0]).is_file():
                    trace.snapshot = Path(reference.snapshots[0]).read_bytes()
                traces.append(trace)
                continue
            identity = {
                k: document.get(k)
                for k in ("execution_id", "commit_hash", "branch_id", "agent_run_id")
            }
            reference = SourceReference(
                identity=path.as_uri(),
                locator=entry.locator,
                resolved_locator=str(path),
                content_hash=hashlib.sha256(raw).hexdigest(),
                list_path=entry.list_path,
                line_number=entry.line_number,
                **identity,
                details={"original_source": document.get("source", {})},
            )
            trace = normalize_document(document, reference)
            # Retain bytes verbatim, including formatting, for hash verification.
            trace.snapshot = raw
            traces.append(trace)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError(f"{entry.diagnostic} ({path}): {exc}") from exc
    return traces
