"""Arguments and handlers for graph annotation and behavioral trace checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from .annotators.jev import DEFAULT_ENDPOINT, MAX_WINDOW
from .decisions import DEFAULT_DECISIONS_MODEL
from .schema import AnnotationConfig


def add_arguments(parser: argparse.ArgumentParser) -> None:
    origins = parser.add_mutually_exclusive_group(required=True)
    origins.add_argument(
        "--input",
        action="append",
        help="Trace file, directory, saved annotation, or Langfuse URL; repeatable.",
    )
    origins.add_argument(
        "--input-list", help="Text file containing one path or URL per line."
    )
    parser.add_argument(
        "--source",
        choices=("path", "langfuse"),
        help="Validate every input against this source type.",
    )
    parser.add_argument(
        "--annotator",
        choices=("llm", "jev"),
        default="llm",
        help="Graph extraction: generative LLM or Jev's workflow via OpenAI Decisions.",
    )
    parser.add_argument(
        "--model",
        help="Annotation model, separate from the agent model in trace metadata.",
    )
    parser.add_argument(
        "--output",
        default="epistemic_analysis",
        help="Output directory (default: ./epistemic_analysis).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Save source snapshots and normalized traces without annotation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs; reads Langfuse for remote entries, never annotates or writes.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Reannotate the requested configuration."
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--window", type=int, default=AnnotationConfig.window)
    parser.add_argument("--overlap", type=int, default=AnnotationConfig.overlap)
    parser.add_argument(
        "--max-nodes-per-window",
        type=int,
        default=AnnotationConfig.max_nodes_per_window,
    )
    parser.add_argument(
        "--tool-output-max-chars",
        type=int,
        default=AnnotationConfig.tool_output_max_chars,
        help="Retained original characters per tool result (0 disables; otherwise >=16).",
    )
    parser.add_argument(
        "--tool-output-truncation",
        choices=("head-tail",),
        default=AnnotationConfig.tool_output_truncation,
    )
    parser.add_argument(
        "--max-request-tokens",
        type=int,
        default=AnnotationConfig.max_request_tokens,
        help="Estimated complete input budget (0 disables preflight enforcement).",
    )
    parser.add_argument(
        "--llm-output-reserve-tokens",
        type=int,
        default=AnnotationConfig.llm_output_reserve_tokens,
        help="Generation allowance reserved within LLM context when budgeting is enabled.",
    )
    parser.add_argument(
        "--strict-support",
        action=argparse.BooleanOptionalAction,
        default=AnnotationConfig.strict_support,
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-span-chars", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--decisions-endpoint",
        "--jev-endpoint",
        dest="jev_endpoint",
        default=DEFAULT_ENDPOINT,
        help="OpenAI Decisions endpoint for the Jev graph workflow.",
    )
    parser.add_argument(
        "--from-time",
        help="Langfuse observation start bound (ISO timestamp with timezone).",
    )
    parser.add_argument(
        "--to-time",
        help="Langfuse observation end bound (exclusive, ISO timestamp with timezone).",
    )
    parser.set_defaults(_handler=run_analysis)


def validate_options(args) -> AnnotationConfig:
    for name in ("concurrency", "max_span_chars", "batch_size", "max_retries"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("temperature must be nonnegative and finite")
    if not math.isfinite(args.min_confidence) or not 0 <= args.min_confidence <= 1:
        raise ValueError("Confidence thresholds must lie between 0 and 1")
    if args.annotator == "jev" and args.window > MAX_WINDOW:
        raise ValueError(f"Decisions graph annotation requires window <= {MAX_WINDOW}")
    return AnnotationConfig(
        model=args.model
        or (
            DEFAULT_DECISIONS_MODEL
            if args.annotator == "jev"
            else "anthropic/claude-sonnet-4-6"
        ),
        window=args.window,
        overlap=args.overlap,
        max_nodes_per_window=args.max_nodes_per_window,
        strict_support=args.strict_support,
        tool_output_max_chars=args.tool_output_max_chars,
        tool_output_truncation=args.tool_output_truncation,
        max_request_tokens=args.max_request_tokens,
        llm_output_reserve_tokens=args.llm_output_reserve_tokens,
    )


def create_backend(args):
    if args.annotator == "llm":
        from .annotators.llm import LLMAnnotator

        return LLMAnnotator(
            temperature=args.temperature,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
        ).backend()
    from .annotators.jev import JevAnnotator, JevClient

    client = JevClient(
        endpoint=args.jev_endpoint,
        batch_size=args.batch_size,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
    )
    return JevAnnotator(
        client, min_confidence=args.min_confidence, max_span_chars=args.max_span_chars
    ).backend()


async def run_analysis(args) -> int:
    from .workflow import run

    return await run(args, validate_options(args))


def add_check_arguments(parser):
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Normalized trace or saved trial JSON.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Independent decision/evidence provider JSON.",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="Public context JSON; never a whole task configuration.",
    )
    parser.add_argument(
        "--policy", type=Path, help="Reviewed applicability/obligation contract JSON."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checks", nargs="+", help="Selected check names; defaults to all checks."
    )
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--force-labels", action="store_true")
    parser.add_argument("--force-evidence", action="store_true")
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Explicit opt-in; whole-trace window conclusions require review.",
    )
    parser.add_argument(
        "--export-review",
        type=Path,
        help="Create a new frozen export for the existing review app.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/configuration without contacting model servers.",
    )
    parser.set_defaults(_handler=run_checks)


def _read_check_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


async def run_checks(args):
    from .annotators.local import GemmaClient, KnowLineClient, ProviderConfig
    from .decision_applicability import ValidationPolicy
    from .sources.inputs import CheckInput
    from .storage import export_review
    from .trace_checks import CheckPipeline, check_definitions

    config = _read_check_json(args.config)
    decision_config = ProviderConfig(**config["decision"])
    evidence_config = (
        ProviderConfig(**config["evidence"]) if config.get("evidence") else None
    )
    inputs = CheckInput.from_document(
        _read_check_json(args.input),
        _read_check_json(args.context) if args.context else {},
    )
    policy = (
        ValidationPolicy(**_read_check_json(args.policy))
        if args.policy
        else ValidationPolicy()
    )
    if args.evidence_only and (args.labels_only or not evidence_config):
        raise ValueError(
            "evidence-only requires an evidence provider and cannot use labels-only"
        )
    if args.force_labels and args.evidence_only:
        raise ValueError("evidence-only cannot force labels")
    decision = KnowLineClient(decision_config)
    evidence = (
        GemmaClient(evidence_config)
        if evidence_config and not args.labels_only
        else None
    )
    try:
        pipeline = CheckPipeline(
            args.output,
            decision,
            evidence,
            threshold=config.get("threshold", 0.6),
            windowed=args.windowed,
            repair_attempts=config.get("repair_attempts", 1),
            reconcile_windowed=config.get("reconcile_windowed", False),
            overflow_tool_output_max_chars=config.get(
                "overflow_tool_output_max_chars", 16000
            ),
        )
        definitions = check_definitions(args.checks)
        if not definitions:
            raise ValueError("Select at least one check")
        if args.dry_run:
            print(  # noqa: T201
                json.dumps(
                    {
                        "input": inputs.fingerprint,
                        "checks": list(definitions),
                        "state": "validated",
                    }
                )
            )
            return 0
        if not args.evidence_only:
            await decision.preflight()
        if evidence:
            await evidence.preflight()
        # One writer per output tree. A crashed/cancelled process releases flock.
        import fcntl

        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".pipeline.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run = await pipeline.run(
                inputs,
                policy=policy,
                checks=args.checks,
                evidence_only=args.evidence_only,
                force_labels=args.force_labels,
                force_evidence=args.force_evidence,
            )
            if args.export_review:
                export_review(args.output, run, args.export_review)
        print(json.dumps(run, indent=2))  # noqa: T201
        return 0 if run["state"] == "complete" else 2
    finally:
        await decision.close()
        if evidence:
            await evidence.close()


def checks_main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_check_arguments(parser)
    try:
        return asyncio.run(run_checks(parser.parse_args()))
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
