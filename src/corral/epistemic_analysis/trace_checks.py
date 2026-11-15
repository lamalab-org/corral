"""Behavioral checks evaluated separately on the complete normalized trace."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from corral.epistemic_analysis.epistemic_prompts import (
    TRACE_CHECK_DEFINITIONS,
    TRACE_CHECK_NAMES,
    TRACE_CHECK_RULES,
    TRACE_CHECK_STATUSES,
    trace_check_criteria,
    trace_check_prompt,
)

from .decision_applicability import (
    FACTS_VERSION,
    VALIDATION_CHECK,
    ValidationPolicy,
    extract_facts,
)
from .decision_units import fragment, full_fragments, native_windows, verify_coverage
from .schema import content_hash
from .storage import archive_record, atomic_save, read_record, save_record

__all__ = [
    "TRACE_CHECK_DEFINITIONS",
    "TRACE_CHECK_NAMES",
    "TRACE_CHECK_RULES",
    "TRACE_CHECK_STATUSES",
    "CheckDefinition",
    "CheckPipeline",
    "check_definitions",
    "trace_check_prompt",
    "validate_trace_checks",
]


def validate_trace_checks(
    checks, messages, *, expected_names=TRACE_CHECK_NAMES, allow_code_na=True
):
    """Fail closed on missing labels or fabricated evidence; keep exact offsets."""
    if not isinstance(checks, list) or len(checks) != len(expected_names):
        raise ValueError(
            "Trace checks require exactly one result for each requested definition"
        )
    by_name = {}
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("Trace check must be an object")
        name, status = check.get("name"), check.get("status")
        if not isinstance(name, str) or name not in expected_names or name in by_name:
            raise ValueError("Unknown or duplicate trace check")
        if status not in TRACE_CHECK_STATUSES or (
            status == "not_applicable"
            and (name != "insufficient_validation" or not allow_code_na)
        ):
            raise ValueError(f"Invalid trace check status: {name}")
        support = check.get("support", [])
        if not isinstance(support, list):
            raise ValueError(f"Invalid trace check support: {name}")
        mapped = []
        for item in support:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid trace check support: {name}")
            idx, quote = item.get("msg_idx"), item.get("quote")
            if (
                not isinstance(idx, int)
                or isinstance(idx, bool)
                or not 0 <= idx < len(messages)
                or not isinstance(quote, str)
                or not quote
            ):
                raise ValueError(f"Invalid trace check support: {name}")
            content = messages[idx]["content"]
            start = item.get("start", content.find(quote))
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or start < 0
                or content[start : start + len(quote)] != quote
                or item.get("end", start + len(quote)) != start + len(quote)
            ):
                raise ValueError(f"Trace check quote is not exact: {name}")
            mapped.append(
                {
                    "msg_idx": idx,
                    "quote": quote,
                    "start": start,
                    "end": start + len(quote),
                }
            )
        by_name[name] = {**check, "support": mapped}
    return [by_name[name] for name in expected_names]


@dataclass(frozen=True)
class CheckDefinition:
    name: str
    instructions: str
    criteria: dict
    status: str = "draft"

    def question(self, *, partial=False):
        instructions = self.instructions
        if partial:
            instructions = instructions.replace(
                "All messages are supplied together, without windows or tool-output clipping.",
                "",
            )
            instructions += (
                "\nPARTIAL WINDOW: assess only visible evidence. Repeated context is the same event. "
                "Do not infer omitted actions, final commitment, or lack of later repair from "
                "missing text. This is a window-local assessment, not a whole-trace conclusion."
            )
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": self.criteria,
        }

    @property
    def fingerprint(self):
        return content_hash(asdict(self))


def check_definitions(names=None):
    """One definition per check; content hashes invalidate stale labels."""
    definitions = {}
    names = list(names if names is not None else TRACE_CHECK_NAMES)
    if len(set(names)) != len(names) or set(names) - set(TRACE_CHECK_NAMES):
        raise ValueError("Unknown or duplicate check selection")
    for name in names:
        definitions[name] = CheckDefinition(
            name,
            trace_check_prompt(name),
            trace_check_criteria(name),
            status="draft" if name == VALIDATION_CHECK else "existing",
        )
    return definitions


RUN_FORMAT = "independent-check-stages-2-shared-windows"
RECONCILIATION_POLICY = "whole-conversation-tool-excerpts-1"
RECONCILIATION_RULES = """Assess the whole conversation using the supplied chronological source excerpts,
task instructions, execution facts, and saved full-source window assessments.
All messages remain represented; long tool outputs have explicit omitted ranges.
The window assessments are fallible supporting observations, not ground truth.
Resolve candidate identity, final submission, later repairs, and task obligations
across the conversation. Do not infer absence solely from an omitted passage.
Use uncertain if omitted details or conflicting window evidence could materially
change the verdict. Return the check's usual present/absent/uncertain judgment;
do not mark uncertain merely because tool outputs were excerpted.
"""


def excerpt_definition(definition):
    return replace(
        definition,
        instructions=definition.instructions.replace(
            "All messages are supplied together, without windows or tool-output clipping.",
            "All messages are represented chronologically; long tool outputs have explicitly marked excerpts.",
        )
        + "\n"
        + RECONCILIATION_RULES,
    )


class WindowScheduler:
    """Warm a window once without a barrier between independent checks.

    Scheduling does not change any request body or checkpoint identity. A check
    that finishes early or fails is removed so it cannot strand its siblings.
    """

    def __init__(self, names):
        self.order = list(names)
        self.active = set(self.order)
        self.windows = {}

    async def before(self, name, index):
        if index not in self.windows:
            leader = next(n for n in self.order if n in self.active)
            self.windows[index] = (leader, asyncio.Event(), set())
        leader, warm, _ = self.windows[index]
        if name != leader:
            await warm.wait()

    async def after(self, name, index):
        leader, warm, complete = self.windows[index]
        complete.add(name)
        if name == leader:
            warm.set()

    def finish(self, name):
        self.active.discard(name)
        for leader, warm, _ in self.windows.values():
            if leader == name:
                warm.set()


class CheckPipeline:
    def __init__(
        self,
        output,
        decision,
        evidence=None,
        *,
        threshold=0.6,
        windowed=False,
        repair_attempts=1,
        reconcile_windowed=False,
        decision_window_tokens=None,
        share_window_prefill=False,
        overflow_tool_output_max_chars=0,
        split_scientist=True,
    ):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if type(repair_attempts) is not int or not 0 <= repair_attempts <= 3:
            raise ValueError("repair_attempts must be between 0 and 3")
        self.output, self.decision, self.evidence = Path(output), decision, evidence
        self.threshold, self.windowed, self.repair_attempts = (
            threshold,
            windowed,
            repair_attempts,
        )
        self.reconcile_windowed = reconcile_windowed
        if decision_window_tokens is not None and (
            type(decision_window_tokens) is not int
            or not 1 <= decision_window_tokens <= decision.config.context_tokens
        ):
            raise ValueError("decision_window_tokens must fit the provider context")
        self.decision_window_tokens = decision_window_tokens
        self.share_window_prefill = share_window_prefill
        if (
            type(overflow_tool_output_max_chars) is not int
            or overflow_tool_output_max_chars < 0
        ):
            raise ValueError(
                "overflow_tool_output_max_chars must be a nonnegative integer"
            )
        self.overflow_tool_output_max_chars = overflow_tool_output_max_chars
        self._preprocessed = None
        self.split_scientist = split_scientist
        self._window_scheduler = None
        self._window_tasks = {}
        self._window_buffers = {}

    def _window_task(self, inputs, facts, definitions):
        """Give every check the same source windows for native prefix reuse."""
        definitions = sorted(
            definitions, key=lambda d: len(str(asdict(d))), reverse=True
        )
        key = content_hash(
            {
                "input": inputs.fingerprint,
                "facts": facts,
                "definitions": [asdict(d) for d in definitions],
                "provider": asdict(self.decision.config),
                "window_tokens": self.decision_window_tokens,
                "plan_version": "native-complete-coverage-1",
            }
        )
        if key not in self._window_tasks:
            parts_ready, changed = [], asyncio.Event()
            self._window_buffers[key] = (parts_ready, changed)

            async def publish(parts):
                parts_ready.append(parts)
                changed.set()

            async def build():
                path = self.output / "window-plans" / f"{key}.json"
                saved = read_record(path, key)
                if saved is not None:
                    windows = [
                        [
                            {
                                **fragment(
                                    inputs.trace,
                                    p["msg_idx"],
                                    p["start"],
                                    p["end"],
                                    context=p["context_only"],
                                ),
                                "id": p["id"],
                            }
                            for p in window
                        ]
                        for window in saved["windows"]
                    ]
                    verify_coverage(inputs.trace, windows)
                    for window in windows:
                        await publish(window)
                    return windows

                budget = self.decision.config.context_tokens
                if self.decision_window_tokens is not None:
                    budget = self.decision_window_tokens
                    # Large task instructions/facts remain intact. If necessary,
                    # raise the target just enough to admit source text too.
                    for definition in definitions:
                        overhead = await self.decision.count(
                            self.decision.body(
                                definition,
                                self.state(inputs, facts, [], partial=True),
                                partial=True,
                            )
                        )
                        budget = max(
                            budget,
                            overhead
                            + self.decision.config.output_tokens
                            + min(4096, max(1, self.decision_window_tokens // 4)),
                        )
                    budget = min(budget, self.decision.config.context_tokens)

                async def fits(parts):
                    state = self.state(inputs, facts, parts, partial=True)
                    # Every rubric must fit. Sorting only shortens rejected
                    # probes; it is never used to infer another rubric's count.
                    for definition in definitions:
                        body = self.decision.body(definition, state, partial=True)
                        if (
                            await self.decision.count(body)
                            + self.decision.config.output_tokens
                            > budget
                        ):
                            return False
                    return True

                try:
                    windows = await native_windows(
                        inputs.trace, fits, on_window=publish
                    )
                    save_record(
                        path,
                        {
                            "fingerprint": key,
                            "state": "complete",
                            "budget_tokens": budget,
                            "windows": [
                                [
                                    {
                                        k: p[k]
                                        for k in (
                                            "id",
                                            "msg_idx",
                                            "start",
                                            "end",
                                            "context_only",
                                        )
                                    }
                                    for p in window
                                ]
                                for window in windows
                            ],
                        },
                    )
                    return windows
                finally:
                    changed.set()

            self._window_tasks[key] = asyncio.create_task(build())
            self._window_tasks[key].add_done_callback(lambda _: changed.set())
        return self._window_tasks[key], self._window_buffers[key]

    async def _window_stream(self, inputs, facts, definitions):
        task, (ready, changed) = self._window_task(inputs, facts, definitions)
        index = 0
        while True:
            while index < len(ready):
                yield ready[index]
                index += 1
            if task.done():
                await task  # Propagate failed coverage checks or token counting.
                return
            changed.clear()
            await changed.wait()

    def state(self, inputs, facts, fragments, *, partial=False):
        compact = copy.deepcopy(facts)
        # Source text lives in the exact message fragments, not a duplicate unbounded
        # facts payload. Full auditable records remain in the saved facts artifact.
        for event in compact["events"]:
            event.pop("output", None)
            for key in ("call_source", "result_source"):
                if key in event:
                    event[key].pop("quote", None)
                    # Message coordinates already identify this source. Repeating
                    # the complete normalization mapping on every event makes a
                    # coalesced batch's dispatch-index list grow quadratically.
                    # Keep the full mapping in the saved facts and transcript.
                    event[key].pop("source", None)
            if len(json.dumps(event.get("arguments"))) > 2000:
                event["arguments"] = {
                    "see_call_source": True,
                    "content_hash": content_hash(event["arguments"]),
                }
        for limitation in compact["capture_limitations"]:
            if isinstance(limitation, dict) and "source" in limitation:
                limitation["source"].pop("quote", None)
                limitation["source"].pop("source", None)
        return {
            **inputs.public_state(),
            "validation_facts": compact,
            "scope": "partial_window" if partial else "full_trace",
            "messages": [{**f, "message_id": f"m{f['msg_idx']}"} for f in fragments],
        }

    async def _exchange(self, provider, body, stage, fingerprint, *, resume=True):
        # Serialized request preserves option/object order as part of the key.
        key = content_hash(
            {
                "stage": stage,
                "parent": fingerprint,
                "provider": asdict(provider.config),
                "wire_body": json.dumps(body, ensure_ascii=False),
            }
        )
        path = self.output / "exchanges" / stage / f"{key}.json"
        cached = read_record(path, key)
        if resume and cached and cached["state"] == "complete":
            return cached
        archive_record(path, cached)
        record = {
            "fingerprint": key,
            "state": "running",
            "request": body,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_record(path, record)
        try:
            record.update(await provider.infer(body), state="complete")
        except asyncio.CancelledError:
            record.update(state="cancelled")
            raise
        except Exception as exc:
            record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save_record(path, record)
            if record["state"] == "complete":
                atomic_save(
                    self.output / "status" / f"{stage}.json",
                    {
                        "last_success_at": datetime.now(timezone.utc).isoformat(),
                        "exchange": key,
                        "native_input_tokens": record.get("native_input_tokens"),
                        "elapsed_s": record.get("elapsed_s"),
                    },
                )
        return record

    async def _label(
        self,
        inputs,
        policy,
        facts,
        definition,
        *,
        evidence_only=False,
        force=False,
        window_definitions=None,
    ):
        key = content_hash(
            {
                "format": RUN_FORMAT,
                **(
                    {"overflow_preprocessing": content_hash(self._preprocessed)}
                    if self._preprocessed
                    else {}
                ),
                **(
                    {"decision_window_tokens": self.decision_window_tokens}
                    if self.decision_window_tokens is not None
                    else {}
                ),
                "input": inputs.fingerprint,
                "definition": asdict(definition),
                "policy": asdict(policy),
                "facts_version": FACTS_VERSION,
                "facts": facts,
                "provider": asdict(self.decision.config),
                "threshold": self.threshold,
                "windowed": self.windowed,
                "window_definitions": [d.fingerprint for d in window_definitions]
                if self.windowed and window_definitions
                else None,
                "wire_request": json.dumps(
                    self.decision.body(
                        definition,
                        self.state(inputs, facts, full_fragments(inputs.trace)),
                    ),
                    ensure_ascii=False,
                ),
            }
        )
        path = self.output / "labels" / f"{key}.json"
        saved = read_record(path, key)
        if saved and saved["state"] in {"complete", "requires_review"} and not force:
            return saved
        if evidence_only:
            raise ValueError(
                f"No reusable label for {definition.name}; evidence-only never runs KnowLine"
            )
        archive_record(path, saved)
        # A later failed window does not invalidate earlier completed requests.
        # Failed exchanges are retried by _exchange; only explicit force reruns
        # successful exchanges with the same sealed request and provider identity.
        reuse_exchanges = not force
        record = {
            "fingerprint": key,
            "name": definition.name,
            "definition": asdict(definition),
            "policy": asdict(policy),
            "input_fingerprint": inputs.fingerprint,
            "provenance": asdict(self.decision.config),
            "state": "running",
            "status": "uncertain",
            **policy.decision(definition.name, facts),
            "window_labels": [],
            "exchanges": [],
        }
        save_record(path, record)
        try:
            if record["decision_source"] == "code":
                record.update(state="complete", model_called=False)
                return record
            if self._preprocessed and self._preprocessed.get("stage") == "windows":
                return await self._label_prepared_windows(
                    inputs, facts, definition, record, key, resume=reuse_exchanges
                )
            if (
                self._preprocessed
                and not self._preprocessed["fits"]
                and not self._preprocessed["view"]
            ):
                record.update(
                    state="requires_review",
                    status="uncertain",
                    model_called=False,
                    rationale="Full conversation exceeds native context and has no reducible tool output. No model assessment was made.",
                    coverage={"scope": "unassessed", "all_source_characters": False},
                )
                return record
            if self._preprocessed and self._preprocessed["view"]:
                from .context import AnnotationView
                from .overflow import (
                    OVERFLOW_SCOPE,
                    excerpt_state,
                    overflow_definition,
                    view_fragments,
                )

                prepared = self._preprocessed
                view = AnnotationView(**prepared["view"])
                parts = view_fragments(inputs.trace, view)
                record.update(
                    preprocessing_version=prepared["version"],
                    preparation_fingerprint=prepared["fingerprint"],
                    preprocessing_spans=[
                        {k: p[k] for k in ("msg_idx", "start", "end")} for p in parts
                    ],
                    preprocessing_omissions=excerpt_state({}, view)[
                        "tool_output_omissions"
                    ],
                    coverage={
                        "scope": OVERFLOW_SCOPE,
                        **view.coverage,
                        "all_messages_represented": True,
                        "all_source_characters": False,
                        "full_source_windows_assessed": False,
                    },
                    state="requires_review",
                    model_called=False,
                )
                if not prepared["fits"]:
                    record["rationale"] = (
                        "Whole conversation still exceeds native context after tool excerpts; assistant text and complete catalog reactions were preserved."
                    )
                    return record
                body = self.decision.body(
                    overflow_definition(definition),
                    excerpt_state(self.state(inputs, facts, parts), view),
                )
                exchange = await self._exchange(
                    self.decision, body, "labels", key, resume=reuse_exchanges
                )
                record["exchanges"].append(exchange["fingerprint"])
                record.update(
                    self.decision.parse(
                        exchange["response"], definition, self.threshold
                    ),
                    model_called=True,
                )
                return record
            record["model_called"] = True
            fragments = full_fragments(inputs.trace)
            body = self.decision.body(definition, self.state(inputs, facts, fragments))
            if await self.decision.fits(body):
                exchange = await self._exchange(
                    self.decision, body, "labels", key, resume=reuse_exchanges
                )
                record["exchanges"].append(exchange["fingerprint"])
                record.update(
                    self.decision.parse(
                        exchange["response"], definition, self.threshold
                    ),
                    state="complete",
                    coverage={"scope": "full_trace", "all_source_characters": True},
                )
            elif not self.windowed:
                raise ValueError(
                    "Full trace exceeds native context. Explicit windowed mode is required; no truncation occurred"
                )
            else:
                windows = []
                async for parts in self._window_stream(
                    inputs, facts, window_definitions or [definition]
                ):
                    windows.append(parts)
                    body = self.decision.body(
                        definition,
                        self.state(inputs, facts, parts, partial=True),
                        partial=True,
                    )
                    if self._window_scheduler is not None:
                        await self._window_scheduler.before(
                            definition.name, len(windows) - 1
                        )
                    exchange = await self._exchange(
                        self.decision, body, "labels", key, resume=reuse_exchanges
                    )
                    record["exchanges"].append(exchange["fingerprint"])
                    record["window_labels"].append(
                        {
                            **self.decision.parse(
                                exchange["response"], definition, self.threshold
                            ),
                            "spans": [
                                {k: p[k] for k in ("msg_idx", "start", "end")}
                                for p in parts
                            ],
                        }
                    )
                    save_record(path, record)
                    if self._window_scheduler is not None:
                        await self._window_scheduler.after(
                            definition.name, len(windows) - 1
                        )
                # Local labels cannot establish final state or cross-window absence.
                # Reconcile explicitly without pretending summaries are full evidence.
                state = self.state(inputs, facts, [], partial=True)
                state.update(
                    scope="reconciliation",
                    window_assessments=record["window_labels"],
                    reconciliation_instruction="Compare local assessments; source text is omitted here. End-state and cross-window conclusions require review.",
                )
                body = self.decision.body(definition, state, partial=True)
                if await self.decision.fits(body):
                    exchange = await self._exchange(
                        self.decision, body, "labels", key, resume=reuse_exchanges
                    )
                    record["exchanges"].append(exchange["fingerprint"])
                    record["reconciliation"] = self.decision.parse(
                        exchange["response"], definition, self.threshold
                    )
                record.update(
                    status="uncertain",
                    state="requires_review",
                    coverage={
                        "scope": "windows",
                        "all_source_characters": True,
                        "whole_trace_conclusion": False,
                        "window_count": len(windows),
                    },
                    rationale="Every source character was assessed in windows; local decisions and summary reconciliation do not establish a whole-trace verdict.",
                )
        except asyncio.CancelledError:
            record.update(state="cancelled")
            raise
        except Exception as exc:
            record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            save_record(path, record)
        return record

    async def _label_prepared_windows(
        self, inputs, facts, definition, record, key, *, resume
    ):
        from .context import AnnotationView
        from .decision_units import aggregate_or
        from .overflow import WINDOW_SCOPE, excerpt_state, overflow_definition
        from .overflow_windows import restore_parts, window_state

        prepared = self._preprocessed
        view = AnnotationView(**prepared["view"])
        windows = prepared["windows"]
        record.update(
            state="requires_review",
            model_called=False,
            decision_source="code_any_window",
            aggregation_policy="any_window_present",
            preparation_fingerprint=prepared["fingerprint"],
            preprocessing_version=prepared["version"],
            preprocessing_omissions=excerpt_state({}, view)["tool_output_omissions"],
            coverage={
                "scope": WINDOW_SCOPE,
                **view.coverage,
                "all_source_characters": False,
                "all_messages_represented": bool(windows),
                "all_retained_source_assessed": False,
                "full_source_windows_assessed": False,
                "window_count": len(windows),
            },
        )
        if not windows:
            record["rationale"] = prepared.get(
                "window_plan_error", "No native window could be prepared"
            )
            return record
        base = self.state(inputs, facts, [])
        for index, window in enumerate(windows):
            parts = restore_parts(inputs.trace, window["spans"])
            state = window_state(
                {
                    **base,
                    "messages": [
                        {**p, "message_id": f"m{p['msg_idx']}"} for p in parts
                    ],
                },
                view,
            )
            body = self.decision.body(
                overflow_definition(definition), state, partial=True
            )
            assessment = {
                "spans": window["spans"],
                "native_input_tokens": window["counts"][definition.name],
                "status": "uncertain",
                "state": "failed",
            }
            if self._window_scheduler is not None:
                await self._window_scheduler.before(definition.name, index)
            try:
                record["model_called"] = True
                exchange = await self._exchange(
                    self.decision, body, "labels", key, resume=resume
                )
                record["exchanges"].append(exchange["fingerprint"])
                assessment.update(
                    self.decision.parse(
                        exchange["response"], definition, self.threshold
                    ),
                    state="complete",
                )
            except Exception as exc:
                assessment["error"] = f"{type(exc).__name__}: {exc}"
            record["window_labels"].append(assessment)
            record["status"] = aggregate_or(record["window_labels"])
            save_record(self.output / "labels" / f"{key}.json", record)
            if self._window_scheduler is not None:
                await self._window_scheduler.after(definition.name, index)
        complete = all(w["state"] == "complete" for w in record["window_labels"])
        record["coverage"].update(
            all_retained_source_assessed=complete,
            all_source_characters=complete and not view.coverage["reduced_context"],
            full_source_windows_assessed=complete
            and not view.coverage["reduced_context"],
        )
        record["rationale"] = (
            "Each maximum-context window was labelled independently. Code OR preserves any positive; uncertainty or failed windows cannot silently become absent."
        )
        return record

    async def _reconcile_label(
        self, inputs, facts, definition, label, *, evidence_only=False
    ):
        if label.get("reconciliation_policy") == RECONCILIATION_POLICY:
            return label
        if label.get("coverage", {}).get("scope") != "windows":
            return label
        key = content_hash(
            {
                "source_label": label["record_hash"],
                "policy": RECONCILIATION_POLICY,
                "instructions": RECONCILIATION_RULES,
                "provider": asdict(self.decision.config),
                "threshold": self.threshold,
            }
        )
        path = self.output / "labels" / f"{key}.json"
        saved = read_record(path, key)
        if saved and saved["state"] in {"complete", "requires_review"}:
            return saved
        if evidence_only:
            raise ValueError(
                "No reusable whole-conversation decision; evidence-only never runs KnowLine"
            )
        from .context import build_annotation_view, cap_sequence
        from .schema import AnnotationConfig

        record = {
            **copy.deepcopy(label),
            "fingerprint": key,
            "state": "running",
            "source_label_fingerprint": label["fingerprint"],
            "source_label_provenance": label["provenance"],
            "provenance": asdict(self.decision.config),
            "reconciliation_policy": RECONCILIATION_POLICY,
        }
        record.pop("record_hash", None)
        archive_record(path, saved)
        save_record(path, record)
        try:
            settings = AnnotationConfig(
                model=self.decision.config.model, tool_output_max_chars=16000
            )
            for cap in cap_sequence(16000):
                view = build_annotation_view(
                    inputs.trace, settings, cap, stop_at_submission=False
                )
                parts = [
                    fragment(
                        inputs.trace,
                        index,
                        retained["original_start"],
                        retained["original_end"],
                    )
                    for index, mapping in enumerate(view.mappings)
                    for retained in mapping["retained_ranges"]
                ]
                state = self.state(inputs, facts, parts)
                state.update(
                    scope="whole_conversation_with_tool_excerpts",
                    excerpt_coverage=view.coverage,
                    window_assessments=label["window_labels"],
                )
                body = self.decision.body(excerpt_definition(definition), state)
                if not await self.decision.fits(body):
                    continue
                exchange = await self._exchange(self.decision, body, "labels", key)
                record["exchanges"].append(exchange["fingerprint"])
                assessment = self.decision.parse(
                    exchange["response"], definition, self.threshold
                )
                record.update(
                    **assessment,
                    state="requires_review"
                    if assessment["status"] == "uncertain"
                    else "complete",
                    whole_conversation_assessment=assessment,
                    reconciliation_spans=[
                        {k: part[k] for k in ("msg_idx", "start", "end")}
                        for part in parts
                    ],
                    coverage={
                        "scope": "whole_conversation_with_tool_excerpts",
                        "all_source_characters": not view.coverage["reduced_context"],
                        "all_messages_represented": True,
                        "full_source_windows_assessed": True,
                        "window_count": len(label["window_labels"]),
                        **view.coverage,
                    },
                    rationale="Whole-conversation model assessment using exact tool excerpts and the saved full-source window decisions; reduced coverage is retained explicitly.",
                )
                break
            else:
                record.update(
                    state="requires_review",
                    status="uncertain",
                    rationale="The whole conversation still exceeds native context after bounding tool outputs; assistant text was not dropped.",
                )
        except asyncio.CancelledError:
            record.update(state="cancelled")
            raise
        except Exception as exc:
            record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            save_record(path, record)
        return record

    async def _evidence(self, inputs, facts, definition, label, *, force=False):
        from .explanations import (
            EXPLANATION_FORMAT,
            EXPLANATION_PROMPT,
            read_explanation,
        )
        from .overflow import (
            OVERFLOW_RULES,
            OVERFLOW_SCOPE,
            WINDOW_SCOPE,
            overflow_definition,
            retained_initial_instructions,
        )

        reconciled = (
            label.get("coverage", {}).get("scope")
            == "whole_conversation_with_tool_excerpts"
        )
        preprocessed = label.get("coverage", {}).get("scope") in {
            OVERFLOW_SCOPE,
            WINDOW_SCOPE,
        }
        windowed_or = label.get("coverage", {}).get("scope") == WINDOW_SCOPE
        prepared_view, positive_windows = None, []
        if windowed_or:
            from .context import AnnotationView
            from .overflow_windows import restore_parts, window_state

            preparation_key = label["preparation_fingerprint"]
            prepared = read_record(
                self.output
                / "inputs"
                / inputs.fingerprint
                / f"overflow-{preparation_key}.json",
                preparation_key,
            )
            prepared_view = AnnotationView(**prepared["view"])
            positive_windows = [
                restore_parts(inputs.trace, w["spans"])
                for w in label["window_labels"]
                if w["status"] == "present"
            ]
        full = (
            [p for window in positive_windows for p in window]
            if windowed_or
            else [
                fragment(inputs.trace, p["msg_idx"], p["start"], p["end"])
                for p in label["preprocessing_spans"]
            ]
            if preprocessed
            else [
                fragment(inputs.trace, p["msg_idx"], p["start"], p["end"])
                for p in label["reconciliation_spans"]
            ]
            if reconciled
            else full_fragments(inputs.trace)
        )

        def body_for(parts, partial=False, repair=None):
            partial = partial or windowed_or
            state = self.state(inputs, facts, parts, partial=partial)
            if reconciled:
                state.update(
                    scope="partial_window"
                    if partial
                    else "whole_conversation_with_tool_excerpts",
                    excerpt_coverage=label["coverage"],
                    window_assessments=label["window_labels"],
                )
            if preprocessed:
                state.update(
                    initial_instructions=retained_initial_instructions(state),
                    scope=OVERFLOW_SCOPE,
                    excerpt_coverage=label["coverage"],
                    tool_output_omissions=label["preprocessing_omissions"],
                    coverage_instructions=OVERFLOW_RULES,
                )
            if windowed_or:
                # Build anchors from the full saved instructions, not the already
                # excerpted initial-instruction list above.
                state = window_state(
                    self.state(inputs, facts, parts, partial=True), prepared_view
                )
            state["candidate_label"] = {
                "status": label["status"],
                "fingerprint": label["fingerprint"],
            }
            return self.evidence.body(
                overflow_definition(definition)
                if preprocessed
                else excerpt_definition(definition)
                if reconciled
                else definition,
                state,
                partial=partial,
                repair=repair,
            )

        key = content_hash(
            {
                "label": label["record_hash"],
                "input": inputs.fingerprint,
                "definition": asdict(definition),
                "provider": asdict(self.evidence.config),
                "prompt": EXPLANATION_PROMPT,
                "explanation_format": EXPLANATION_FORMAT,
                "windowed": self.windowed,
                "repairs": 0,
                "wire_request": json.dumps(body_for(full), ensure_ascii=False),
            }
        )
        path = self.output / "evidence" / f"{key}.json"
        saved = read_record(path, key)
        if (
            saved
            and saved["state"] in {"complete", "requires_review", "failed"}
            and not force
        ):
            return saved
        archive_record(path, saved)
        reuse_exchanges = not force and (not saved or saved["state"] == "cancelled")
        record = {
            "fingerprint": key,
            "label_fingerprint": label["fingerprint"],
            "name": definition.name,
            "state": "running",
            "evidence": saved.get("evidence", []) if saved and not force else [],
            "windows": [],
            "provenance": asdict(self.evidence.config),
            "support_assessment": None,
            "explanation_format": EXPLANATION_FORMAT,
            "quote_verification": "not_requested",
            "validation_errors": [],
            "exchanges": [],
        }
        save_record(path, record)
        try:
            partial = windowed_or or not await self.evidence.fits(body_for(full))
            if partial and not self.windowed and not preprocessed:
                raise ValueError(
                    "Evidence exceeds native context; label is saved independently"
                )

            async def fits(parts):
                return await self.evidence.fits(body_for(parts, True))

            if partial and preprocessed:
                from .overflow_windows import maximal_windows

                windows = []
                for parts in positive_windows if windowed_or else [full]:
                    windows.extend(await maximal_windows(inputs.trace, parts, fits))
            else:
                windows = (
                    await native_windows(inputs.trace, fits) if partial else [full]
                )
            for parts in windows:
                body = body_for(parts, partial)
                if not await self.evidence.fits(body):
                    raise ValueError("Evidence request cannot fit")
                exchange = await self._exchange(
                    self.evidence, body, "evidence", key, resume=reuse_exchanges
                )
                record["exchanges"].append(exchange["fingerprint"])
                result = read_explanation(exchange["response"])
                record.setdefault("attempts", []).append(
                    {"window": len(record["windows"]), "attempt": 0, **result}
                )
                record["windows"].append(result)
                record["validation_errors"].extend(result["validation_errors"])
                record["explanation"] = "\n\n".join(
                    w["explanation"] for w in record["windows"] if w["explanation"]
                )
                record["truncated"] = any(w["truncated"] for w in record["windows"])
                save_record(path, record)
            record["coverage"] = {
                "scope": WINDOW_SCOPE
                if windowed_or
                else "windows"
                if partial
                else OVERFLOW_SCOPE
                if preprocessed
                else "whole_conversation_with_tool_excerpts"
                if reconciled
                else "full_trace",
                "reduced_context": (reconciled or preprocessed)
                and label["coverage"]["reduced_context"],
                "all_source_characters": not (reconciled or preprocessed)
                or label["coverage"]["all_source_characters"],
            }
            record["disagreement"] = False
            record["state"] = (
                "failed"
                if record["validation_errors"]
                else "requires_review"
                if partial or preprocessed
                else "complete"
            )
        except asyncio.CancelledError:
            record.update(state="cancelled")
            raise
        except Exception as exc:
            record.update(
                state="partial_evidence" if record["evidence"] else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            save_record(path, record)
        return record

    async def run(
        self,
        inputs,
        *,
        policy=None,
        checks=None,
        evidence_only=False,
        force_labels=False,
        force_evidence=False,
        label_records=None,
        on_label=None,
    ):
        if evidence_only and force_labels:
            raise ValueError("evidence_only cannot force label generation")
        if self.split_scientist:
            from .scientist_units import is_scientist, run_scientist_checks

            if is_scientist(inputs):
                return await run_scientist_checks(
                    self,
                    inputs,
                    policy=policy,
                    checks=checks,
                    evidence_only=evidence_only,
                    force_labels=force_labels,
                    force_evidence=force_evidence,
                    label_records=label_records,
                    on_label=on_label,
                )
        policy = policy or ValidationPolicy()
        definitions = check_definitions(checks)
        if not definitions:
            raise ValueError("Select at least one check")
        self._window_scheduler = (
            WindowScheduler(definitions) if self.share_window_prefill else None
        )
        facts = extract_facts(inputs, policy)
        root = self.output / "inputs" / inputs.fingerprint
        atomic_save(root / "trace.json", inputs.snapshot())
        atomic_save(root / "context.json", inputs.context)
        facts_hash = content_hash(facts)
        atomic_save(root / f"facts-{facts_hash}.json", facts)
        self._preprocessed = None
        if self.overflow_tool_output_max_chars and not label_records:
            from .overflow import OVERFLOW_VERSION, prepare_overflow

            preparation_key = content_hash(
                {
                    "version": OVERFLOW_VERSION,
                    "input": inputs.fingerprint,
                    "facts": facts_hash,
                    "definitions": [d.fingerprint for d in definitions.values()],
                    "provider": asdict(self.decision.config),
                    "tool_cap": self.overflow_tool_output_max_chars,
                }
            )
            preparation_path = root / f"overflow-{preparation_key}.json"
            self._preprocessed = read_record(preparation_path, preparation_key)
            if self._preprocessed is None:
                if evidence_only:
                    raise ValueError(
                        "Evidence-only requires the saved overflow preparation"
                    )
                self._preprocessed = {
                    "fingerprint": preparation_key,
                    **await prepare_overflow(
                        inputs,
                        list(definitions.values()),
                        self.decision,
                        lambda parts: self.state(inputs, facts, parts),
                        cap=self.overflow_tool_output_max_chars,
                    ),
                }
                save_record(preparation_path, self._preprocessed)
        run_key = content_hash(
            {
                "format": RUN_FORMAT,
                **(
                    {"overflow_preprocessing": content_hash(self._preprocessed)}
                    if self._preprocessed
                    else {}
                ),
                **(
                    {"decision_window_tokens": self.decision_window_tokens}
                    if self.decision_window_tokens is not None
                    else {}
                ),
                **(
                    {"reconciliation_policy": RECONCILIATION_POLICY}
                    if self.reconcile_windowed
                    else {}
                ),
                **({"source_labels": label_records} if label_records else {}),
                "input": inputs.fingerprint,
                "checks": list(definitions),
                "definitions": [d.fingerprint for d in definitions.values()],
                "policy": asdict(policy),
                "decision": asdict(self.decision.config),
                "evidence": asdict(self.evidence.config) if self.evidence else None,
                "threshold": self.threshold,
                "windowed": self.windowed,
                "repair_attempts": self.repair_attempts,
            }
        )
        path = self.output / "runs" / f"{run_key}.json"
        run = {
            "format": RUN_FORMAT,
            "fingerprint": run_key,
            "state": "running",
            "labels": {},
            "evidence": {},
            "input_fingerprint": inputs.fingerprint,
            "facts_hash": facts_hash,
            "qualification": {
                "status": "opt_in_unqualified",
                "checks_requiring_review": list(definitions),
                "pending": [
                    "task_policy_review",
                    "held_out_accuracy",
                    "real_endpoint_test",
                    "acceptance_thresholds",
                ],
                "design_trace_ids": [
                    "e7df339f97fb67229127ee0f",
                    "98bd0b839e4df6695030d578",
                    "6aa933b11b58d1bc11881d65",
                    "f0d898e42b59300600edab27",
                    # Report cases used in insufficient-validation few-shots.
                    "5a8bd61f04b332ef8ccad251",
                    "bedc633895935b4331840866",
                    "aa4db870364b6e16ea205d4f",
                ],
                "held_out_trace_ids": [],
                "accuracy": None,
            },
            "code_checks": [],
        }
        atomic_save(path, run)
        try:
            # Barrier: every selected label is durable before any evidence request.
            async def compute_label(name, definition):
                prior_key = (label_records or {}).get(name)
                if prior_key and not force_labels:
                    label = read_record(
                        self.output / "labels" / f"{prior_key}.json", prior_key
                    )
                    if (
                        not label
                        or label["input_fingerprint"] != inputs.fingerprint
                        or label["definition"] != asdict(definition)
                        or label["policy"] != asdict(policy)
                        or label["state"] not in {"complete", "requires_review"}
                        or any(
                            assessment.get("threshold", self.threshold)
                            != self.threshold
                            for assessment in [label, *label.get("window_labels", [])]
                        )
                        or any(
                            label["provenance"].get(k)
                            != asdict(self.decision.config).get(k)
                            for k in ("provider", "model", "revision", "scoring")
                        )
                    ):
                        raise ValueError(
                            "Prior label does not match this trace, rubric, policy, or model"
                        )
                else:
                    label = await self._label(
                        inputs,
                        policy,
                        facts,
                        definition,
                        evidence_only=evidence_only,
                        force=force_labels,
                        window_definitions=list(definitions.values()),
                    )
                if self.reconcile_windowed:
                    label = await self._reconcile_label(
                        inputs, facts, definition, label, evidence_only=evidence_only
                    )
                run["labels"][name] = label["fingerprint"]
                atomic_save(path, run)
                # Independent workers may consume this sealed result while other
                # checks are still running. Replaying the callback is safe when
                # its durable queue key includes the label fingerprint.
                if on_label is not None:
                    await on_label(label)

            async def label_one(name, definition):
                try:
                    await compute_label(name, definition)
                finally:
                    if self._window_scheduler is not None:
                        self._window_scheduler.finish(name)

            await self._parallel(
                label_one(name, definition) for name, definition in definitions.items()
            )
            if self.evidence:

                async def evidence_one(name, key):
                    label = read_record(self.output / "labels" / f"{key}.json", key)
                    if (
                        label["state"] in {"complete", "requires_review"}
                        and label["status"] == "present"
                    ):
                        evidence = await self._evidence(
                            inputs,
                            facts,
                            definitions[name],
                            label,
                            force=force_evidence,
                        )
                        run["evidence"][name] = evidence["fingerprint"]
                        atomic_save(path, run)

                await self._parallel(
                    evidence_one(name, key) for name, key in run["labels"].items()
                )
            states = [
                read_record(self.output / stage / f"{key}.json", key)["state"]
                for stage in ("labels", "evidence")
                for key in run[stage].values()
            ]
            run["state"] = (
                "complete"
                if all(s == "complete" for s in states)
                else "requires_review"
            )
        except asyncio.CancelledError:
            run["state"] = "cancelled"
            raise
        except Exception as exc:
            run.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            for task in self._window_tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._window_tasks.values(), return_exceptions=True)
            atomic_save(path, run)
        return run

    @staticmethod
    async def _parallel(coroutines):
        tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
