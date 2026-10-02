"""Arguments and async handler for ``corral analyze``."""

from __future__ import annotations

import argparse
import math
import os

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
    parser.add_argument("--annotator", choices=("llm", "jev"), default="llm")
    parser.add_argument(
        "--model",
        help="Annotation model, separate from the agent model in trace metadata.",
    )
    parser.add_argument(
        "--output",
        default="epistemic_analysis",
        help="Output directory (default: ./epistemic_analysis).",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--skip-annotate",
        action="store_true",
        help="Reanalyze explicitly selected local annotations, without provider clients.",
    )
    modes.add_argument(
        "--download-only",
        action="store_true",
        help="Save source snapshots and normalized traces without annotation.",
    )
    modes.add_argument(
        "--annotate-only",
        action="store_true",
        help="Save annotations without reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs; reads Langfuse for remote entries, never annotates or writes.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Reannotate the requested configuration."
    )
    parser.add_argument(
        "--configuration",
        help="Exact configuration ID to select from a saved annotation directory.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=["agent_model", "environment", "level"],
        choices=("agent_model", "environment", "level", "task", "trial"),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--max-nodes-per-window", type=int, default=100)
    parser.add_argument(
        "--tool-output-max-chars",
        type=int,
        default=4000,
        help="Retained original characters per tool result (0 disables; otherwise >=16).",
    )
    parser.add_argument(
        "--tool-output-truncation", choices=("head-tail",), default="head-tail"
    )
    parser.add_argument(
        "--max-request-tokens",
        type=int,
        default=0,
        help="Estimated complete input budget (0 disables preflight enforcement).",
    )
    parser.add_argument(
        "--llm-output-reserve-tokens",
        type=int,
        default=4096,
        help="Generation allowance reserved within LLM context when budgeting is enabled.",
    )
    parser.add_argument(
        "--strict-support", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-span-chars", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--jev-endpoint", default="https://api.typesafe.ai/v1/systemone"
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
    if args.annotator == "jev" and args.window > 252:
        raise ValueError("Jev requires window <= 252")
    if args.configuration and not args.skip_annotate:
        raise ValueError("--configuration requires --skip-annotate")
    if args.skip_annotate and args.force:
        raise ValueError("--force cannot be used with --skip-annotate")
    return AnnotationConfig(
        model=args.model
        or ("jev-latest" if args.annotator == "jev" else "anthropic/claude-sonnet-4-6"),
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
        os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY") or "",
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
