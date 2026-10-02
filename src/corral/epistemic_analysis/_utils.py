"""_utils for epistemic trace analysis."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def iter_windows(n: int, window: int, overlap: int) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    if window <= 0:
        return [(0, n)]
    step = max(1, window - overlap)
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + window)
        out.append((start, end))
        if end == n:
            break
        start += step
    return out


def format_window(messages: list[dict[str, Any]], start: int, end: int) -> str:
    lines = []
    # Always prepend the first two messages (task / system context)
    preamble_end = min(2, len(messages))
    for i in range(preamble_end):
        role = messages[i].get("role", "")
        content = messages[i].get("content", "")
        if content is None:
            content = ""
        content = str(content)
        lines.append(f"[{i}] role={role}\n{content}\n")
    # For windows that don't start at the beginning, indicate omitted messages
    actual_start = max(start, preamble_end)
    if actual_start > preamble_end:
        lines.append(f"[... messages {preamble_end}-{actual_start - 1} omitted ...]\n")
    for i in range(actual_start, end):
        role = messages[i].get("role", "")
        content = messages[i].get("content", "")
        if content is None:
            content = ""
        content = str(content)
        lines.append(f"[{i}] role={role}\n{content}\n")
    return "\n".join(lines)
