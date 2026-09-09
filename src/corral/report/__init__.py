"""Reporting API, loaded lazily to keep benchmark metadata clients lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BenchmarkResult": ("corral.report.results", "BenchmarkResult"),
    "TaskTrialResult": ("corral.report.results", "TaskTrialResult"),
    "TaskTrialResults": ("corral.report.results", "TaskTrialResults"),
    "project_benchmark_result": (
        "corral.report.projection",
        "project_benchmark_result",
    ),
}

__all__ = [
    "BenchmarkResult",
    "TaskTrialResult",
    "TaskTrialResults",
    "project_benchmark_result",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
