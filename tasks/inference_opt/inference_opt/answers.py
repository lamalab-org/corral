"""Turn what a policy returns into the completion text the scorers parse."""

from __future__ import annotations

from typing import Any

from inference_opt.api import Answer

__all__ = ["ANSWER_PREFIX", "MAX_COMPLETION_CHARS", "completion_for"]

ANSWER_PREFIX = "ANSWER:"

#: Longer completions are cut; the answer marker belongs at the end of a short reply.
MAX_COMPLETION_CHARS = 8192


def completion_for(raw: Any, benchmark: str) -> str:
    """A structured :class:`Answer` gets the benchmark's marker; anything else is text."""
    if isinstance(raw, Answer):
        if benchmark == "chembench":
            completion = f"[ANSWER]{raw.final}[/ANSWER]"
        else:
            completion = f"{ANSWER_PREFIX} {raw.final}"
    else:
        completion = "" if raw is None else str(raw)
    return completion[:MAX_COMPLETION_CHARS]
