"""Shared utility functions for corral benchmark scripts.

Functions used by more than one script (or that logically belong alongside the
constants declared in :mod:`constants`) are collected here.
"""

import json
import math
from pathlib import Path
from typing import Any

from constants import MODEL_CANONICAL, SKIP_DIR_PREFIXES
from loguru import logger


def read_json(path: Path) -> Any:
    """Read and return the contents of a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> float | None:
    """Convert *x* to float, returning ``None`` on failure."""
    try:
        return float(x)
    except Exception:
        return None


def logsumexp(logps: list[float]) -> float:
    if not logps:
        return float("-inf")
    m = max(logps)
    return m + math.log(sum(math.exp(lp - m) for lp in logps))


def entropy_from_top_logprobs(top_logprobs: list[dict[str, Any]]) -> float | None:
    """Compute entropy over the distribution implied by *top_logprobs*.

    *top_logprobs* is a list of dicts like ``{"token": "...", "logprob": -0.12, …}``.
    Normalises across the provided candidates (approximate entropy).
    """
    if not top_logprobs:
        return None

    lps: list[float] = []
    for cand in top_logprobs:
        lp = safe_float(cand.get("logprob"))
        if lp is not None:
            lps.append(lp)

    if not lps:
        return None

    z = logsumexp(lps)
    ps = [math.exp(lp - z) for lp in lps]
    return -sum(p * math.log(p + 1e-300) for p in ps)


def infer_model_from_str(s: str) -> str | None:
    """Return the canonical model key found in *s*, or ``None``."""
    s_norm = s.lower().replace("-", "_")
    # Longest-first ordering prevents shorter aliases (e.g. 'claude') from
    # shadowing more specific ones (e.g. 'claude_45').
    for pat in sorted(MODEL_CANONICAL, key=len, reverse=True):
        if pat.replace("-", "_") in s_norm:
            return MODEL_CANONICAL[pat]
    return None


def infer_agent_type(filename: str) -> str:
    """Return ``'tool_calling'`` or ``'react'`` based on the filename."""
    fl = filename.lower()
    if "tool_calling" in fl or "toolcalling" in fl:
        return "tool_calling"
    return "react"


def should_skip_dir(name: str) -> bool:
    """Return ``True`` if a directory with *name* should be ignored."""
    return any(name.startswith(p) for p in SKIP_DIR_PREFIXES)


def is_report_json(filepath: Path) -> bool:
    """Quick pre-filter: JSON, not hidden, not inside a skipped directory."""
    if filepath.suffix != ".json" or filepath.name.startswith("."):
        return False
    return all(not should_skip_dir(part) for part in filepath.parts)


def load_report(filepath: Path) -> dict | None:
    """Load a report JSON; return ``None`` if it cannot be read or is not a report."""
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Skipping {filepath}: {exc}")
        return None
    if "metrics" not in data or "task_results" not in data:
        return None
    return data
