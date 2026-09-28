"""Condense evaluator tracebacks into what an agent or a report can use."""

from __future__ import annotations

__all__ = ["error_line", "error_tail"]


def error_tail(error: str, limit: int) -> str:
    """Keep the end of a traceback: the actual exception is at the bottom."""
    error = error.strip()
    return error if len(error) <= limit else "..." + error[-limit:]


def error_line(error: str) -> str:
    """The final ``SomeError: message`` line, even inside an ExceptionGroup."""
    lines = [line.strip().lstrip("|").strip() for line in error.splitlines()]
    lines = [line for line in lines if line and not line.startswith(("+-", "^"))]
    for line in reversed(lines):
        name = line.split(":", 1)[0]
        if name.endswith(("Error", "Exception", "Exhausted")) and " " not in name:
            return line[:300]
    return lines[-1][:300] if lines else ""
