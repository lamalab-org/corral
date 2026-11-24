"""Sources files for epistemic trace analysis."""

from __future__ import annotations

from pathlib import Path


def normalize_document(document, source, grouping=None):
    """Normalize message text and retain original message indices."""
    from corral.epistemic_analysis.schema import GroupingMetadata, NormalizedTrace

    if not isinstance(document, dict) or not isinstance(document.get("messages"), list):
        raise ValueError("Trace requires a messages list")
    messages, references = [], []
    original_sources = document.get("source", {}).get("message_sources", [])
    for index, message in enumerate(document["messages"]):
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
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
    return NormalizedTrace(
        messages,
        source,
        references,
        grouping,
        completion_status=document.get("completion_status", "unspecified"),
        reconstruction_status=document.get("reconstruction_status", "complete"),
        snapshot=document,
    )


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


def load_annotations(entry, configuration=None):
    """Select only a manifest's files or explicitly supplied annotation files.

    A directory with multiple configurations requires --configuration. A saved
    invocation manifest is already an explicit selection and may contain them.
    """
    import json

    from corral.epistemic_analysis.schema import (
        read_graph,
    )

    origin = entry.path
    manifest = origin / "manifest.json" if origin.is_dir() else origin
    selected_by_manifest = False
    if manifest.is_file() and manifest.name == "manifest.json":
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("format") != "corral-epistemic-selection-v1":
            raise ValueError(f"{entry.diagnostic}: unsupported selection manifest")
        paths = [(manifest.parent / name).resolve() for name in data["annotations"]]
        selected_by_manifest = True
    elif origin.is_dir():
        paths = sorted(origin.rglob("*.annotated.json"))
    else:
        paths = [origin]
    graphs = []
    for path in paths:
        try:
            graph = read_graph(json.loads(path.read_text(encoding="utf-8")))
            if (
                configuration
                and graph.provenance.get("configuration_id") != configuration
            ):
                continue
            graphs.append(graph)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{entry.diagnostic} ({path}): {exc}") from exc
    if not graphs:
        raise ValueError(f"{entry.diagnostic}: no annotations selected")
    configs = {g.provenance.get("configuration_id") for g in graphs}
    if len(configs) > 1 and not selected_by_manifest:
        raise ValueError(
            "Multiple annotation configurations found; use --configuration or a selection manifest"
        )
    return graphs
