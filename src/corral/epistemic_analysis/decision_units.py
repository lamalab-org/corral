"""Exact source fragments, native context windows, and conservative label reduction."""

from __future__ import annotations

import re

NODE_NAME = re.compile(
    r"^(?:experiment_(node_\d+)_step_\d+|evaluate_(?:visual_)?(node_\d+))$"
)


def fragment(trace, index, start=0, end=None, *, context=False):
    message = trace.messages[index]
    end = len(message["content"]) if end is None else end
    return {
        "id": f"m{index}:{start}:{end}",
        "msg_idx": index,
        "role": message["role"],
        "content": message["content"][start:end],
        "start": start,
        "end": end,
        "message_chars": len(message["content"]),
        "source": trace.message_sources[index],
        "context_only": context,
    }


def aggregate_or(results):
    """User's node OR; errors/uncertainty never silently become an absent label."""
    if not results:
        return "uncertain"
    statuses = [r.get("status", "uncertain") for r in results]
    if "present" in statuses:
        return "present"
    if any(s not in {"absent", "not_applicable"} for s in statuses):
        return "uncertain"
    return (
        "not_applicable" if all(s == "not_applicable" for s in statuses) else "absent"
    )


def full_fragments(trace):
    return [fragment(trace, i) for i in range(len(trace.messages))]


def verify_coverage(trace, windows):
    ranges = {i: [] for i in range(len(trace.messages))}
    for window in windows:
        for part in window:
            i, start, end = part["msg_idx"], part["start"], part["end"]
            if (
                type(i) is not int
                or i not in ranges
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start <= end <= len(trace.messages[i]["content"])
                or trace.messages[i]["content"][start:end] != part["content"]
                or trace.messages[i]["role"] != part["role"]
            ):
                raise ValueError("Invalid original source fragment")
            ranges[i].append((start, end))
    for i, spans in ranges.items():
        covered = 0
        if not spans:
            raise ValueError(f"Missing message m{i}")
        for start, end in sorted(spans):
            if start > covered:
                raise ValueError(f"Gap in m{i} at {covered}")
            covered = max(covered, end)
        if covered != len(trace.messages[i]["content"]):
            raise ValueError(f"Uncovered tail of m{i}")


async def native_windows(trace, fits, *, overlap_chars=256, on_window=None):
    """Count complete native requests, including repeated context and output reserve.

    The supplied predicate builds the complete request for every probe. A huge
    message is split at Unicode character boundaries without dropping any text.
    Token counts need not be monotonic: every chosen prefix is checked explicitly.
    """
    windows, current = [], []
    for i, message in enumerate(trace.messages):
        start, length = 0, len(message["content"])
        while start < length or length == 0:
            whole = fragment(trace, i, start, length)
            if await fits([*current, whole]):
                current.append(whole)
                break
            if current:
                windows.append(current)
                if on_window:
                    await on_window(current)
                tail = current[-1]
                current = (
                    [
                        {
                            **tail,
                            "start": max(tail["start"], tail["end"] - overlap_chars),
                            "content": tail["content"][-overlap_chars:],
                            "context_only": True,
                        }
                    ]
                    if overlap_chars and tail["content"]
                    else []
                )
                while current and not await fits(
                    [*current, fragment(trace, i, start, min(start + 1, length))]
                ):
                    tail = current[0]
                    keep = len(tail["content"]) // 2
                    current = (
                        [
                            {
                                **tail,
                                "start": tail["end"] - keep,
                                "content": tail["content"][-keep:],
                            }
                        ]
                        if keep
                        else []
                    )
            lo, hi, best = start + 1, length, None
            while lo <= hi:
                mid = (lo + hi) // 2
                part = fragment(trace, i, start, mid)
                if await fits([*current, part]):
                    best, lo = part, mid + 1
                else:
                    hi = mid - 1
            if best is None:
                raise ValueError(
                    "Task context, rubric, and facts leave no room for a source character"
                )
            current.append(best)
            start = best["end"]
    if current:
        windows.append(current)
        if on_window:
            await on_window(current)
    verify_coverage(trace, windows)
    return windows
