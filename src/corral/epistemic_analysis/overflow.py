"""Fit whole conversations by excerpting tool results, with original offsets.

Full native requests are counted first. Source transcripts are never rewritten.
Catalog excerpts retain complete records, rather than cutting through reactions.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import asdict, replace
from itertools import pairwise

from .context import AnnotationView, RetainedRange, is_tool_result, task_message_indices
from .decision_units import fragment, full_fragments
from .schema import content_hash

OVERFLOW_VERSION = "overflow-exact-instructions-v3"
OVERFLOW_SCOPE = "whole_conversation_with_preprocessed_tool_outputs"
WINDOW_SCOPE = "preprocessed_windows_with_code_or"
OVERFLOW_RULES = """Source excerpts preserve chronological order. Tool outputs and,
only after tool trimming proved insufficient, long messages can have marked
omissions. Only retained text and explicitly measured repetition metadata are
evidence; omitted catalog reactions, data rows, and prose have NOT been assessed. Do not infer
absence of a behavior or lack of validation from omitted text. Choose uncertain
when the omitted details could change the conclusion. Repetition metadata records
actual source run lengths and counts, not a model summary: use it together with
the retained samples when checking generation_breakdown. Annotation omission
markers are not agent-generated text."""


def overflow_definition(definition):
    return replace(
        definition,
        instructions=definition.instructions.replace(
            "All messages are supplied together, without windows or tool-output clipping.",
            OVERFLOW_RULES,
        ),
    )


def catalog_records(text):
    """Locate complete dictionaries in a JSON or Python-repr reaction list.

    The lexical pass handles nested structures and escaped quotes. Parsing each
    record with literal_eval validates the structure without executing tool text.
    """
    start = len(text) - len(text.lstrip())
    if text[start : start + 1] != "[":
        return []
    records, stack, quote, escaped, record_start = [], ["["], None, False, None
    for i in range(start + 1, len(text)):
        char = text[i]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "[{(":
            if len(stack) == 1:
                if char != "{":
                    return []
                record_start = i
            stack.append(char)
        elif char in "]})":
            if not stack or stack.pop() != {"]": "[", "}": "{", ")": "("}[char]:
                return []
            if not stack:
                return records
            if len(stack) == 1:
                try:
                    record = ast.literal_eval(text[record_start : i + 1])
                except (ValueError, SyntaxError, RecursionError):
                    # JSON true/null values are legitimate catalog metadata.
                    try:
                        record = json.loads(text[record_start : i + 1])
                    except ValueError:
                        return []
                if not isinstance(record, dict) or not (
                    {"reaction_id", "template_id"} & record.keys()
                ):
                    return []
                records.append((record_start, i + 1))
        elif len(stack) == 1 and not char.isspace() and char != ",":
            return []
    return []


def output_spans(text, cap, keep):
    records = catalog_records(text)
    catalog = (
        {"available": len(records), "retained": min(keep, len(records))}
        if records
        else None
    )
    if len(records) > keep:
        return [(0, records[keep - 1][1]), (records[-1][1], len(text))], catalog
    if records or len(text) <= cap:
        return [(0, len(text))], catalog
    return [(0, cap - cap // 4), (len(text) - cap // 4, len(text))], None


def embedded_tool_results(text):
    """Exact spans of AI Scientist's serialized Observation.result strings.

    Only the recorded Observation schema is accepted. Task instructions, action
    arguments, success/error fields, and surrounding user prose are untouched.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{\s*"action_index"\s*:', text):
        try:
            observation, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if (
            not isinstance(observation, dict)
            or not {
                "action_index",
                "purpose",
                "tool_name",
                "arguments",
                "success",
                "result",
                "error",
            }
            >= observation.keys()
            or not all(
                k in observation
                for k in ("action_index", "tool_name", "success", "result")
            )
        ):
            continue
        decoded = observation["result"]
        if not isinstance(decoded, str) or not isinstance(observation["success"], bool):
            continue
        cursor = match.start() + 1
        while True:
            while text[cursor].isspace() or text[cursor] == ",":
                cursor += 1
            if text[cursor] == "}":
                break
            key, cursor = decoder.raw_decode(text, cursor)
            while text[cursor].isspace() or text[cursor] == ":":
                cursor += 1
            start = cursor
            _, cursor = decoder.raw_decode(text, cursor)
            if key != "result":
                continue
            # Map decoded Unicode character boundaries back into the literal
            # JSON string, including escaped newlines, quotes and surrogate pairs.
            offset, boundaries = start + 1, [start + 1]
            while offset < cursor - 1:
                if text[offset] == "\\":
                    width = 6 if text[offset + 1] == "u" else 2
                    if (
                        width == 6
                        and 0xD800 <= int(text[offset + 2 : offset + 6], 16) <= 0xDBFF
                        and text[offset + 6 : offset + 8] == "\\u"
                    ):
                        width = 12
                    offset += width
                else:
                    offset += 1
                boundaries.append(offset)
            if len(boundaries) == len(decoded) + 1:
                yield decoded, boundaries
            break


def tool_excerpt_view(trace, cap=16000, catalog_keep=3):
    """Keep all non-tool messages; preserve the first 2/3 ranked reactions."""
    if cap < 1 or catalog_keep not in {2, 3}:
        raise ValueError(
            "Positive tool cap and two or three catalog reactions required"
        )
    messages, mappings = [], []
    instructions = task_message_indices(trace.messages)
    for index, (message, source) in enumerate(
        zip(trace.messages, trace.message_sources, strict=True)
    ):
        text = message["content"]
        tool = message["role"] not in {
            "assistant",
            "user",
            "system",
            "developer",
        } and is_tool_result(message, source)
        spans = [(0, len(text))]
        catalog = None
        embedded_catalogs, embedded_chars = [], 0
        if tool:
            spans, catalog = output_spans(text, cap, catalog_keep)
        elif (
            source.get("scientist_unit")
            and message["role"] == "user"
            and index not in instructions
        ):
            omitted_spans = []
            for decoded, offsets in embedded_tool_results(text):
                embedded_chars += offsets[-1] - offsets[0]
                kept, info = output_spans(decoded, cap, catalog_keep)
                if info:
                    embedded_catalogs.append(info)
                for previous, following in pairwise(kept):
                    omitted_spans.append((offsets[previous[1]], offsets[following[0]]))
            if omitted_spans:
                spans, cursor = [], 0
                for start, end in sorted(omitted_spans):
                    if start < cursor:
                        raise ValueError("Overlapping embedded observations")
                    spans.append((cursor, start))
                    cursor = end
                spans.append((cursor, len(text)))
        omitted = len(text) - sum(end - start for start, end in spans)
        rendered, ranges = "", []
        for number, (start, end) in enumerate(spans):
            if number:
                note = (
                    f"; {catalog['available'] - catalog['retained']} catalog reactions omitted"
                    if catalog
                    else ""
                )
                gap = start - spans[number - 1][1]
                rendered += (
                    f"\n[ANNOTATION VIEW: {gap:,} original characters omitted{note}]\n"
                )
            offset = len(rendered)
            rendered += text[start:end]
            ranges.append(asdict(RetainedRange(start, end, offset, len(rendered))))
        messages.append({**message, "content": rendered})
        mappings.append(
            {
                "msg_idx": index,
                "is_tool_result": tool or bool(embedded_chars),
                "source": dict(source),
                "original_length": len(text),
                "original_content_hash": content_hash(text),
                "omitted_chars": omitted,
                "retained_ranges": ranges,
                **({"catalog_reactions": catalog} if catalog else {}),
                **(
                    {"embedded_catalog_reactions": embedded_catalogs}
                    if embedded_catalogs
                    else {}
                ),
                **({"tool_original_chars": embedded_chars} if embedded_chars else {}),
            }
        )
    return AnnotationView(
        messages,
        mappings,
        cap,
        cap,
        "head-tail-and-complete-catalog-records",
        OVERFLOW_VERSION,
    )


def view_fragments(trace, view):
    return [
        fragment(trace, m["msg_idx"], span["original_start"], span["original_end"])
        for m in view.mappings
        for span in m["retained_ranges"]
    ]


def _subtract(spans, cuts):
    result = []
    for start, end in spans:
        cursor = start
        for low, high in sorted(cuts):
            if high <= cursor or low >= end:
                continue
            if cursor < low:
                result.append((cursor, low))
            cursor = min(end, max(cursor, high))
        if cursor < end or start == end:
            result.append((cursor, end))
    return result


def repetition_runs(text):
    """Find verifiable long runs without interpreting the agent's claims."""
    runs = [
        {
            "start": m.start(),
            "end": m.end(),
            "kind": "repeated_character",
            "sample": m[1],
            "repetitions": len(m[0]),
        }
        for m in re.finditer(r"(.)\1{4095,}", text, re.DOTALL)
    ]
    offset, previous, start, count = 0, None, 0, 0
    for line in [*text.splitlines(keepends=True), None]:
        if line != previous:
            if previous and previous.strip() and count >= 8 and offset - start >= 4096:
                runs.append(
                    {
                        "start": start,
                        "end": offset,
                        "kind": "repeated_line",
                        "sample": previous[:200],
                        "repetitions": count,
                    }
                )
            previous, start, count = line, offset, 0
        if line is not None:
            offset += len(line)
            count += 1
    # A character run inside a repeated line is already represented by that line.
    selected, end = [], -1
    for run in sorted(runs, key=lambda r: (r["start"], -r["end"])):
        if run["start"] >= end:
            selected.append(run)
            end = run["end"]
    return selected


def long_message_view(trace, tool_view, *, cap=None):
    """Keep repetition samples first; optionally bound prose at 64k/32k/16k chars.

    Complete retained catalog records and embedded tool results are protected.
    Every range maps directly to the original source, never an earlier excerpt.
    """
    view = copy.deepcopy(tool_view)
    view.policy = "tool-excerpts-then-repetition-and-long-message-excerpts"
    for index, mapping in enumerate(view.mappings):
        text = trace.messages[index]["content"]
        if trace.messages[index]["role"] in {"system", "developer", "user"}:
            continue  # Never shorten the instructions used to assess behavior.
        if mapping["is_tool_result"] and "tool_original_chars" not in mapping:
            continue
        spans = [
            (r["original_start"], r["original_end"]) for r in mapping["retained_ranges"]
        ]
        protected = (
            [(offsets[0], offsets[-1]) for _, offsets in embedded_tool_results(text)]
            if mapping.get("tool_original_chars")
            else []
        )
        cuts, signals = [], []
        for run in repetition_runs(text):
            low, high = run["start"], run["end"]
            if any(low < b and high > a for a, b in protected):
                continue
            if any(a <= low and high <= b for a, b in spans):
                # Hundreds of actual repetitions remain visible on both sides.
                cuts.append((low + 512, high - 512))
                signals.append(
                    {**run, "original_chars": high - low, "retained_sample_chars": 1024}
                )
        spans = _subtract(spans, cuts)
        if cap is not None and sum(b - a for a, b in spans) > cap:
            mutable = _subtract(spans, protected)
            length = sum(b - a for a, b in mutable)
            if length > cap:
                # Preserve beginning, middle and ending; do not trim down to tiny snippets.
                wanted = [
                    (0, cap // 2),
                    ((length - cap // 4) // 2, (length + cap // 4) // 2),
                    (length - cap // 4, length),
                ]
                kept, offset = [], 0
                for start, end in mutable:
                    for low, high in wanted:
                        a, b = max(low, offset), min(high, offset + end - start)
                        if a < b:
                            kept.append((start + a - offset, start + b - offset))
                    offset += end - start
                # Keep repetition samples even if they fall outside the prose samples.
                kept.extend((r["start"], r["start"] + 512) for r in signals)
                kept.extend((r["end"] - 512, r["end"]) for r in signals)
                kept.extend(
                    (max(a, low), min(b, high))
                    for a, b in spans
                    for low, high in protected
                    if max(a, low) < min(b, high)
                )
                merged = []
                for start, end in sorted(kept):
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                    else:
                        merged.append((start, end))
                spans = merged
        omitted = len(text) - sum(b - a for a, b in spans)
        additional = omitted - mapping["omitted_chars"]
        if additional <= 0:
            continue
        rendered, ranges = "", []
        for number, (start, end) in enumerate(spans):
            if number:
                rendered += f"\n[ANNOTATION VIEW: {start - spans[number - 1][1]:,} original characters omitted; repetition counts are recorded separately]\n"
            offset = len(rendered)
            rendered += text[start:end]
            ranges.append(asdict(RetainedRange(start, end, offset, len(rendered))))
        mapping.update(
            tool_omitted_chars=mapping["omitted_chars"]
            if mapping["is_tool_result"]
            else 0,
            message_omitted_chars=additional,
            omitted_chars=omitted,
            retained_ranges=ranges,
            repetition_signals=signals,
            long_message_cap=cap,
        )
        view.messages[index] = {**trace.messages[index], "content": rendered}
    return view


def retained_initial_instructions(state):
    """Retain exact instructions even when the current window starts later."""
    return copy.deepcopy(state.get("initial_instructions", []))


def excerpt_state(state, view):
    coverage = view.coverage
    coverage.pop("effective_tool_output_max_chars", None)
    return {
        **state,
        "initial_instructions": retained_initial_instructions(state),
        "scope": OVERFLOW_SCOPE,
        "excerpt_coverage": {
            **coverage,
            "all_messages_represented": True,
            "all_source_characters": False,
            "full_source_windows_assessed": False,
        },
        "tool_output_omissions": [
            {
                "msg_idx": m["msg_idx"],
                "omitted_chars": m["omitted_chars"],
                "message_omitted_chars": m.get("message_omitted_chars", 0),
                "original_chars": m["original_length"],
                "repetition_signals": m.get("repetition_signals", []),
                **(
                    {"catalog_reactions": m["catalog_reactions"]}
                    if "catalog_reactions" in m
                    else {}
                ),
                **(
                    {"embedded_catalog_reactions": m["embedded_catalog_reactions"]}
                    if "embedded_catalog_reactions" in m
                    else {}
                ),
            }
            for m in view.mappings
            if m["omitted_chars"]
        ],
        "coverage_instructions": OVERFLOW_RULES,
    }


async def prepare_overflow(inputs, definitions, decision, state_for, *, cap=16000):
    """Return a measured view, or an explicit still-oversized result.

    Every rubric must fit, including its output reserve; counting stops at the
    first overflow. Large reaction records are never shortened merely to satisfy
    the generic character cap.
    """

    template = state_for([])
    measured_states = {}

    def make_state(parts):
        return {
            **template,
            "messages": [{**p, "message_id": f"m{p['msg_idx']}"} for p in parts],
        }

    async def counts(parts, view=None):
        state = make_state(parts)
        if view:
            state = excerpt_state(state, view)
        identity = content_hash(state)
        if identity in measured_states:
            return measured_states[identity]
        result = {}
        for definition in definitions:
            d = overflow_definition(definition) if view else definition
            result[d.name] = await decision.count(decision.body(d, state))
            if (
                result[d.name] + decision.config.output_tokens
                > decision.config.context_tokens
            ):
                break  # One measured overflow proves the full request set cannot fit.
        measured_states[identity] = result
        return result

    def fits(values):
        return (
            max(values.values()) + decision.config.output_tokens
            <= decision.config.context_tokens
        )

    full_counts = await counts(full_fragments(inputs.trace))
    result = {
        "version": OVERFLOW_VERSION,
        "input_fingerprint": inputs.fingerprint,
        "context_tokens": decision.config.context_tokens,
        "output_tokens": decision.config.output_tokens,
        "full_counts": full_counts,
        "full_fits": fits(full_counts),
        "counts": full_counts,
        "fits": fits(full_counts),
        "view": None,
        "stage": "full",
        "windows": [],
    }
    if result["full_fits"]:
        return result
    minimal = tool_excerpt_view(inputs.trace, min(cap, 256), 2)
    caps = list(
        dict.fromkeys(
            [cap, *[n for n in (8000, 4000, 2000, 1000, 512, 256) if n < cap]]
        )
    )
    # Prefer three complete reactions; only reduce to two if all three cannot fit.
    best_view, best_counts = minimal, None
    if minimal.coverage["reduced_context"]:
        for keep in (3, 2):
            for current in caps:
                view = tool_excerpt_view(inputs.trace, current, keep)
                measured = await counts(view_fragments(inputs.trace, view), view)
                if best_counts is None or max(measured.values()) < max(
                    best_counts.values()
                ):
                    best_view, best_counts = view, measured
                if fits(measured):
                    result.update(
                        counts=measured,
                        fits=True,
                        view=view.to_dict(),
                        stage="tool_outputs",
                    )
                    return result
    else:
        best_counts = full_counts
    result.update(tool_only_counts=best_counts, tool_only_fits=False)
    # First retain all prose and reduce only long exact repetitions. General
    # message excerpts are deliberately bounded no lower than 16,000 characters.
    last_view = best_view
    for message_cap in (None, 64000, 32000, 16000):
        view = long_message_view(inputs.trace, best_view, cap=message_cap)
        measured = await counts(view_fragments(inputs.trace, view), view)
        last_view = view
        result.update(
            counts=measured,
            fits=fits(measured),
            view=view.to_dict(),
            stage="long_messages",
            long_message_cap=message_cap,
        )
        if result["fits"]:
            return result
    from .overflow_windows import prepare_windows

    result["stage"] = "windows"
    try:
        result["windows"] = await prepare_windows(
            inputs, last_view, definitions, decision, make_state
        )
    except ValueError as exc:
        result["window_plan_error"] = str(exc)
    return result
