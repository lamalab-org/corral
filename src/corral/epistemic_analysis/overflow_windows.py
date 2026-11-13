"""Native-context windows over retained original spans, followed by code OR."""

from __future__ import annotations

from .annotation_transcript import TASK_PROMPT_ORIGIN
from .decision_units import fragment
from .overflow import WINDOW_SCOPE, excerpt_state, overflow_definition


def span_refs(parts):
    return [
        {k: p[k] for k in ("msg_idx", "start", "end", "context_only")} for p in parts
    ]


def restore_parts(trace, refs):
    return [
        fragment(
            trace,
            p["msg_idx"],
            p["start"],
            p["end"],
            context=p.get("context_only", False),
        )
        for p in refs
    ]


def window_state(state, view):
    """Keep task anchors but scope bulky facts and omission metadata to this window."""
    parts = state["messages"]
    initial = []
    instructions = state.get("initial_instructions", [])
    has_task_anchor = any(
        i.get("source", {}).get("origin") == TASK_PROMPT_ORIGIN for i in instructions
    )
    for item in instructions:
        if (
            has_task_anchor
            and item["role"] == "user"
            and item.get("source", {}).get("origin") != TASK_PROMPT_ORIGIN
        ):
            continue
        mapping = view.mappings[item["msg_idx"]]
        remaining = 8192
        for span in mapping["retained_ranges"]:
            low, high = span["original_start"], span["original_end"]
            high = min(high, low + remaining)
            if high > low:
                initial.append(
                    {
                        **item,
                        "content": item["content"][low:high],
                        "start": low,
                        "end": high,
                        "context_only": True,
                    }
                )
                remaining -= high - low
            if not remaining:
                break
    visible = {p["msg_idx"] for p in [*parts, *initial]}
    state = excerpt_state(state, view)
    state["initial_instructions"] = initial
    state["scope"] = WINDOW_SCOPE
    state["excerpt_coverage"]["all_messages_represented"] = False
    state["tool_output_omissions"] = [
        m for m in state["tool_output_omissions"] if m["msg_idx"] in visible
    ]
    if facts := state.get("validation_facts"):
        facts = {**facts, "scope": "visible_window_only"}
        facts["events"] = [
            event
            for event in facts.get("events", [])
            if any(
                (event.get(key) or {}).get("msg_idx") in visible
                for key in ("call_source", "result_source")
            )
        ]
        # Relationships between omitted events cannot be inferred in a local window.
        facts["candidate_relationships"] = []
        state["validation_facts"] = facts
    state["window_aggregation"] = (
        "Each check is assessed independently in every window. Code assigns present if any window is present; without a positive, uncertainty or failure remains uncertain. Do not infer missing actions from a partial window."
    )
    state["window_anchor_policy"] = (
        "Repeated instruction anchors are bounded context excerpts. Their complete retained source is covered by the chronological windows."
    )
    return state


def verify_retained_coverage(trace, retained, windows):
    expected = {}
    for part in retained:
        expected.setdefault(part["msg_idx"], []).append((part["start"], part["end"]))
    actual = {i: [] for i in expected}
    for window in windows:
        for part in window:
            index, low, high = part["msg_idx"], part["start"], part["end"]
            if (
                index not in expected
                or not 0 <= low <= high <= len(trace.messages[index]["content"])
                or part["role"] != trace.messages[index]["role"]
                or trace.messages[index]["content"][low:high] != part["content"]
                or not any(a <= low <= high <= b for a, b in expected[index])
            ):
                raise ValueError(
                    "Window restored omitted text or lost original offsets"
                )
            actual[index].append((low, high))
    for index, spans in expected.items():
        for low, high in spans:
            cursor, found = low, False
            for a, b in sorted(actual[index]):
                if b < low or a > high:
                    continue
                if a <= cursor:
                    cursor, found = max(cursor, b), True
            if not found or cursor < high:
                raise ValueError(f"Uncovered retained span m{index}:{low}:{high}")


async def maximal_windows(
    trace,
    retained,
    fits,
    *,
    overlap_chars=256,
    on_window=None,
    token_budget=None,
    last_token_count=None,
):
    """Fill the complete native budget, even across message boundaries.

    Probe exponentially, then refine the boundary. Native token counts can guide
    interpolation, but every emitted request is verified. Empty messages consume
    a cursor position so they cannot disappear. Only already-retained spans enter
    the stream; omitted text is never restored by windowing.
    """
    from bisect import bisect_right

    ends, total = [], 0
    for part in retained:
        total += max(1, part["end"] - part["start"])
        ends.append(total)

    def pieces(start, end):
        result = []
        index = bisect_right(ends, start)
        while index < len(retained):
            low = ends[index - 1] if index else 0
            if low >= end:
                break
            part = retained[index]
            a = part["start"] + max(0, start - low)
            b = part["start"] + min(part["end"] - part["start"], end - low)
            result.append(fragment(trace, part["msg_idx"], a, b))
            index += 1
        return result

    windows, start, overlap = [], 0, []
    while start < total:

        async def probe(end, overlap=overlap, start=start):
            valid = await fits([*overlap, *pieces(start, end)])
            return valid, last_token_count() if last_token_count else None

        low, low_cost = start, 0
        high = min(total, start + 65536)
        valid, high_cost = await probe(high)
        while valid:
            low, low_cost = high, high_cost
            if high == total:
                break
            high = min(total, start + 2 * (high - start))
            valid, high_cost = await probe(high)
        if not valid:
            while high - low > 1:
                mid = (low + high) // 2
                if (
                    token_budget
                    and high_cost is not None
                    and low_cost is not None
                    and high_cost > low_cost
                ):
                    estimate = low + int(
                        (high - low)
                        * (token_budget - low_cost)
                        / (high_cost - low_cost)
                    )
                    margin = max(1, (high - low) // 10)
                    mid = max(low + margin, min(high - margin, estimate))
                valid, cost = await probe(mid)
                if valid:
                    low, low_cost = mid, cost
                else:
                    high, high_cost = mid, cost
        if low == start:
            if overlap:
                overlap = []
                continue
            raise ValueError(
                "Task context and rubric leave no room for a retained source character"
            )
        window = [*overlap, *pieces(start, low)]
        if not await fits(window):
            raise ValueError("Final window failed native verification")
        windows.append(window)
        if on_window:
            on_window(window)
        tail = window[-1]
        overlap = (
            [
                fragment(
                    trace,
                    tail["msg_idx"],
                    max(tail["start"], tail["end"] - overlap_chars),
                    tail["end"],
                    context=True,
                )
            ]
            if overlap_chars and tail["content"]
            else []
        )
        start = low
    verify_retained_coverage(trace, retained, windows)
    return windows


async def prepare_windows(inputs, view, definitions, decision, state_for):
    from .overflow import view_fragments

    last_counts, last_cost, result = None, None, []

    async def measure(parts):
        nonlocal last_cost
        state, counts = window_state(state_for(parts), view), {}
        for definition in definitions:
            body = decision.body(overflow_definition(definition), state, partial=True)
            counts[definition.name] = await decision.count(body)
            last_cost = max(counts.values())
            if (
                counts[definition.name] + decision.config.output_tokens
                > decision.config.context_tokens
            ):
                return None
        return counts

    async def fits(parts):
        nonlocal last_counts
        counts = await measure(parts)
        if counts is None:
            return False
        last_counts = counts
        return True

    def save_window(parts):
        result.append({"spans": span_refs(parts), "counts": last_counts})

    await maximal_windows(
        inputs.trace,
        view_fragments(inputs.trace, view),
        fits,
        on_window=save_window,
        token_budget=decision.config.context_tokens - decision.config.output_tokens,
        last_token_count=lambda: last_cost,
    )
    return result
