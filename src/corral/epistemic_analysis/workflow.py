"""Compose source adapters, in-memory annotation, and report writers."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

from loguru import logger

from .aggregation import aggregate
from .analysis import analyze_graph
from .budget import RequestBudgetExceeded
from .context import AnnotationView
from .pipeline import annotate_trace, annotation_fingerprint, can_reuse
from .schema import AnnotatedGraph, content_hash, read_graph
from .sources.files import load_annotations, load_path
from .sources.inputs import resolve_inputs
from .sources.langfuse import download_locator, traces_from_observations
from .storage import safe_write_json, save_annotation_artifacts


def _save_trace(trace, output):
    key = content_hash(
        [
            trace.source.identity,
            trace.source.content_hash,
            trace.source.normalization_version,
        ]
    )
    snapshot = output / "snapshots" / f"{key}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(trace.snapshot, bytes):
        snapshot.write_bytes(trace.snapshot)
    else:
        safe_write_json(snapshot, trace.snapshot)
    normalized = output / "traces" / f"{key}.json"
    trace.source = replace(trace.source, snapshots=(str(snapshot), str(normalized)))
    safe_write_json(
        normalized,
        {
            "format": "corral-epistemic-trace-v1",
            "messages": trace.messages,
            "source": asdict(trace.source),
            "message_sources": trace.message_sources,
            "grouping": asdict(trace.grouping),
            "completion_status": trace.completion_status,
            "reconstruction_status": trace.reconstruction_status,
        },
    )
    return str(normalized.relative_to(output))


async def run(args, config) -> int:
    from .cli import create_backend
    from .reporting import write_reports

    entries = resolve_inputs(args.input, args.input_list, args.source)
    output = Path(args.output).expanduser().resolve()
    if args.skip_annotate and any(entry.kind != "path" for entry in entries):
        raise ValueError(
            "--skip-annotate requires local saved annotations; no Langfuse download is performed"
        )
    if args.dry_run:
        logger.info(
            "Dry run: reads remote Langfuse data for URL inputs; no annotation requests or output writes."
        )
    graphs, traces, errors = [], [], []
    for entry in entries:
        try:
            if args.skip_annotate:
                graphs.extend(load_annotations(entry, args.configuration))
            elif entry.kind == "path":
                traces.extend(load_path(entry))
            else:
                rows = await asyncio.to_thread(
                    download_locator, entry.remote, args.from_time, args.to_time
                )
                # Retain evidence even if reconstruction rejects a branch or incomplete history.
                if not args.dry_run:
                    safe_write_json(
                        output / "snapshots" / f"langfuse-{content_hash(rows)}.json",
                        rows,
                    )
                traces.extend(traces_from_observations(entry, rows))
        except (ValueError, TypeError, OSError) as exc:
            errors.append(f"{entry.diagnostic}: {exc}")
    if errors:
        raise ValueError(
            "Input validation failed before annotation:\n" + "\n".join(errors)
        )
    unique = {}
    for trace in traces:
        key = (
            trace.source.identity,
            trace.source.content_hash,
            trace.source.normalization_version,
        )
        if key in unique:
            previous = unique[key]
            origins = previous.source.details.get("input_references", [])
            if not origins:
                origins.append(
                    {
                        "locator": previous.source.locator,
                        "list_path": previous.source.list_path,
                        "line_number": previous.source.line_number,
                    }
                )
            origins.append(
                {
                    "locator": trace.source.locator,
                    "list_path": trace.source.list_path,
                    "line_number": trace.source.line_number,
                }
            )
            previous.source = replace(
                previous.source,
                details={**previous.source.details, "input_references": origins},
            )
        else:
            unique[key] = trace
    traces = list(unique.values())
    versions = {}
    for trace in traces:
        version = (trace.source.content_hash, trace.source.normalization_version)
        if (
            trace.source.identity in versions
            and versions[trace.source.identity] != version
        ):
            raise ValueError(
                f"Multiple snapshots selected for {trace.source.identity}; select one snapshot explicitly"
            )
        versions[trace.source.identity] = version
    if graphs:
        # Validate snapshot selection before writing any reanalysis outputs.
        aggregate([analyze_graph(graph) for graph in graphs], tuple(args.group_by))
    if args.dry_run:
        logger.info(
            f"Validated {len(graphs) if args.skip_annotate else len(traces)} selected trace(s)."
        )
        return 0
    normalized_files = [_save_trace(trace, output) for trace in traces]
    if args.download_only:
        safe_write_json(
            output / "manifest.json",
            {"format": "corral-epistemic-download-v1", "traces": normalized_files},
        )
        logger.info(f"Saved {len(traces)} trace(s) to {output}")
        return 0
    annotations = []
    failed = 0
    if not args.skip_annotate:
        backend = create_backend(args)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def process(trace):
            nonlocal failed
            fingerprint = annotation_fingerprint(trace, backend, config)
            request_id = content_hash(fingerprint)
            cache_index = output / "cache" / f"{request_id}.json"
            async with semaphore:
                graph = None
                if cache_index.is_file() and not args.force:
                    try:
                        path = (
                            output
                            / json.loads(cache_index.read_text(encoding="utf-8"))[
                                "annotation"
                            ]
                        )
                        cached = read_graph(
                            json.loads(path.read_text(encoding="utf-8"))
                        )
                        if can_reuse(cached, trace, backend, config):
                            graph = cached
                            # Current input references/grouping do not affect extraction.
                            graph.source, graph.grouping, graph.input_file = (
                                trace.source,
                                trace.grouping,
                                trace.source.locator,
                            )
                    except (ValueError, TypeError, KeyError, OSError):
                        pass
                if graph is None:
                    try:
                        graph = await annotate_trace(trace, backend, config)
                    except Exception as exc:
                        failed += 1
                        graph = AnnotatedGraph(
                            nodes=[],
                            edges=[],
                            source=trace.source,
                            grouping=trace.grouping,
                            input_file=trace.source.locator,
                            annotation_status="failed",
                            messages=trace.messages,
                            message_sources=trace.message_sources,
                            completion_status=trace.completion_status,
                            reconstruction_status=trace.reconstruction_status,
                            provenance={
                                "fingerprint": fingerprint,
                                "configuration_id": content_hash(
                                    fingerprint["configuration"]
                                ),
                            },
                            qc={"warnings": [], "error": str(exc)},
                            annotation_requests=getattr(exc, "annotation_requests", []),
                        )
                        if isinstance(exc, RequestBudgetExceeded):
                            graph.annotation_view = exc.annotation_view
                            graph.qc.update(
                                {
                                    "annotation_attempts": exc.attempts,
                                    "request_diagnostics": exc.diagnostics,
                                    "size_error": exc.diagnostic,
                                }
                            )
                            if exc.annotation_view:
                                view = AnnotationView(**exc.annotation_view)
                                graph.qc["context_coverage"] = view.coverage
                                graph.provenance["effective_tool_output_max_chars"] = (
                                    view.effective_cap
                                )
                saved_id = graph.provenance.get(
                    "resolved_fingerprint_hash", request_id + ".failed"
                )
                path = output / "annotations" / f"{saved_id}.annotated.json"
                if graph.annotation_status == "incomplete":
                    failed += 1
                save_annotation_artifacts(graph, output)
                safe_write_json(path, graph.to_dict())
                if graph.annotation_status == "complete":
                    safe_write_json(
                        cache_index, {"annotation": str(path.relative_to(output))}
                    )
            return graph, str(path.relative_to(output))

        completed = await asyncio.gather(*(process(trace) for trace in traces))
        graphs = [graph for graph, _ in completed]
        annotations = [name for _, name in completed]
    else:
        # Copy the explicit selection to a portable local result set. Reanalysis
        # has no dependency on source paths, credentials, or a provider client.
        for graph in graphs:
            save_annotation_artifacts(graph, output)
            path = (
                output
                / "annotations"
                / f"{content_hash(graph.to_dict())}.annotated.json"
            )
            safe_write_json(path, graph.to_dict())
            annotations.append(str(path.relative_to(output)))
    summary = aggregate(
        [analyze_graph(graph) for graph in graphs], tuple(args.group_by)
    )
    safe_write_json(
        output / "manifest.json",
        {
            "format": "corral-epistemic-selection-v1",
            "annotations": annotations,
            "traces": normalized_files,
        },
    )
    if not args.annotate_only:
        write_reports(summary, output / "analysis")
    logger.info(
        f"Analyzed {summary.n_selected} trace(s); {failed} annotation failure(s). Outputs: {output}"
    )
    return 1 if failed else 0
